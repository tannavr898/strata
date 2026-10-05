"""UserPromptSubmit hook: record the prompt and snapshot the code state.

Writes the prompt to .claude/last_prompt.txt (the changelog quotes it) and saves
a fingerprint of the code plus the CHANGELOG's mtime, which the Stop hook
compares against at the end of the turn. Never blocks the prompt.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from _fingerprint import PROMPT, STATE, changelog_mtime, code_fingerprint  # noqa: E402


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace") or "{}")
    except ValueError:
        payload = {}
    prompt = payload.get("prompt", "")

    try:
        PROMPT.write_text(prompt, encoding="utf-8")
        STATE.write_text(json.dumps({
            "saved_at": time.time(),
            "session_id": payload.get("session_id"),
            "code_fingerprint": code_fingerprint(),
            "changelog_mtime": changelog_mtime(),
        }), encoding="utf-8")
    except Exception as exc:  # a broken hook must never eat the user's prompt
        print(f"save_prompt hook: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
