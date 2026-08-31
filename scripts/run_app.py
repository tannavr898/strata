"""Run the Strata web app.

    python scripts/run_app.py              # http://127.0.0.1:8000
    python scripts/run_app.py --port 9000
    python scripts/run_app.py --reload      # auto-restart while developing

Needs two build steps first (each prints what to do if missing):
    python scripts/build_models.py    # fits the models -- slow, run once
    python scripts/seed_db.py         # creates data/strata.db with a demo field
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="restart on code changes")
    args = ap.parse_args()

    import uvicorn

    print(f"Strata -> http://{args.host}:{args.port}")
    uvicorn.run(
        "strata.web.app:app",
        host=args.host, port=args.port, reload=args.reload,
        reload_dirs=[str(_SRC)] if args.reload else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
