"""Stop hook: refuse to finish a turn that changed code but not the changelog.

Compares the code fingerprint and CHANGELOG mtime against the snapshot the
UserPromptSubmit hook took. If code changed and docs/CHANGELOG.md did not, exits
2 with a message on stderr, which Claude Code feeds back to Claude so it updates
the docs before stopping.

Exits 0 immediately when `stop_hook_active` is set: that means Claude is already
continuing because of this hook, and blocking again could loop forever.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _fingerprint import STATE, changelog_mtime, code_fingerprint  # noqa: E402

MESSAGE = (
    "Code changed this turn but docs/CHANGELOG.md was not updated. Per the "
    "documentation policy in CLAUDE.md: append a CHANGELOG entry (date, the prompt "
    "from .claude/last_prompt.txt, files changed, what and why, metric before/after "
    "if applicable); update docs/TECHNICAL_REPORT.md and any Mermaid diagram if "
    "architecture, data flow, or a module's behavior changed; and regenerate figures "
    "with `python docs/scripts/make_figures.py` if metrics changed."
)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace") or "{}")
    except ValueError:
        payload = {}
    if payload.get("stop_hook_active"):
        return 0

    try:
        snap = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0            # no snapshot this turn: nothing to compare against

    try:
        code_changed = code_fingerprint() != snap.get("code_fingerprint")
        docs_changed = changelog_mtime() != snap.get("changelog_mtime")
    except Exception as exc:  # never trap the session because git hiccupped
        print(f"check_docs hook: {exc}", file=sys.stderr)
        return 0

    if code_changed and not docs_changed:
        print(MESSAGE, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
