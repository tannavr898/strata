"""Shared helpers for the documentation hooks (stdlib only).

`code_fingerprint()` hashes the state of every CODE path in the repo: HEAD, the
diff against HEAD, and the contents of untracked files. Documentation paths are
excluded, so writing docs never looks like a code change. Any code edit, new
file, or commit during a turn changes the fingerprint.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / ".claude" / "hooks" / ".turn_state.json"
PROMPT = ROOT / ".claude" / "last_prompt.txt"
CHANGELOG = ROOT / "docs" / "CHANGELOG.md"

# Paths that are documentation, not code. Editing these is what the Stop hook
# asks for, so they must not count as a code change themselves.
_EXCLUDE = [":(exclude)docs", ":(exclude).claude", ":(exclude)*.md"]


def _git(*args: str) -> bytes:
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, timeout=30)
    return out.stdout if out.returncode == 0 else b""


def code_fingerprint() -> str:
    h = hashlib.sha256()
    h.update(_git("rev-parse", "HEAD"))
    h.update(_git("diff", "HEAD", "--", ".", *_EXCLUDE))
    untracked = _git("ls-files", "--others", "--exclude-standard", "-z", "--", ".", *_EXCLUDE)
    for name in sorted(p for p in untracked.split(b"\0") if p):
        h.update(name)
        try:
            h.update((ROOT / name.decode("utf-8", "replace")).read_bytes())
        except OSError:
            pass
    return h.hexdigest()


def changelog_mtime() -> float:
    try:
        return CHANGELOG.stat().st_mtime
    except OSError:
        return 0.0
