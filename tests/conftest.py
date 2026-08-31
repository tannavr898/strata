"""Shared fixtures.

The processed CSVs are gitignored (reproduced from the README provenance, not
committed), so every data-backed test SKIPS rather than fails when they are
absent. A fresh clone can therefore run the suite and get a meaningful result:
the contract tests below still run, because they assert on code, not data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

PROCESSED = _REPO_ROOT / "data" / "processed"


def _load_or_skip(name: str):
    import pandas as pd

    path = PROCESSED / name
    if not path.exists():
        pytest.skip(f"{path.relative_to(_REPO_ROOT)} not present; "
                    "reproduce it per the README provenance table")
    return pd.read_csv(path, dtype={"county_ansi": "string"}, low_memory=False)


@pytest.fixture(scope="session")
def analysis_table():
    """The assembled county-year-crop modeling table."""
    return _load_or_skip("analysis_county_year.csv")


@pytest.fixture(scope="session")
def ssurgo_soil():
    """County-level SSURGO topsoil summary, one row per county."""
    return _load_or_skip("ssurgo_pa_soil.csv")


@pytest.fixture(scope="session")
def nass_yields():
    """Raw NASS county-year yields, before the soil join."""
    return _load_or_skip("nass_pa_yields.csv")
