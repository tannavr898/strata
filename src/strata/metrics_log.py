"""metrics_log.py — append one line per evaluation run to docs/metrics.jsonl.

The technical report can only show that a change improved the model if the
numbers were recorded when the change was made. Every evaluation entry point
(`scripts/build_models.py`, `python -m strata.main`) calls `append_run` once at
the end, so docs/metrics.jsonl becomes the history that
`docs/scripts/make_figures.py` plots metric-versus-change from.

One JSON object per line:

    {"timestamp": ..., "commit": "9d5a69e", "dirty": true,
     "kind": "build_models", "source": "run", "metrics": {...}}

`dirty` records uncommitted changes at run time: a number from a dirty tree
belongs to code that is not in any commit yet, and the log should say so rather
than attribute it to HEAD. Rows marked `"source": "backfill"` were transcribed
from earlier docs, not produced by a run; the figures script draws them
differently.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = _REPO_ROOT / "docs" / "metrics.jsonl"


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def append_run(kind: str, metrics: dict, note: Optional[str] = None,
               path: Path = METRICS_PATH) -> dict:
    """Append one evaluation record and return it.

    Never raises on a git failure: a missing commit hash is recorded as None
    rather than costing the caller the run it just spent minutes on.
    """
    status = _git("status", "--porcelain")
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "kind": kind,
        "source": "run",
        "metrics": metrics,
    }
    if note:
        record["note"] = note
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record
