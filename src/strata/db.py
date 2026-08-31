"""db.py — SQLite persistence for fields, sites, probe readings, and lab tests.

Strata's analysis layer is stateless: CSVs in, models out. The app needs the
opposite -- somewhere to put a farmer's fields and the readings that accumulate
against them over a season. That is all this module is.

Stdlib `sqlite3`, no ORM. The schema is six tables; an ORM would add a dependency
and a layer of indirection to save nothing.

DESIGN DECISIONS WORTH KNOWING
------------------------------
*Single farm per instance.* There is no user or tenant table. The prototype runs
for one operation, which removes auth, sessions, and access control from the
critical path. Multi-farm later means adding a `farm` table and a foreign key,
which is why every table already has its own id rather than a composite key.

*Readings are long, not wide.* One row per (site, time, channel) rather than one
column per sensor. Firmware adding a channel is then a data change, not a schema
migration -- and it mirrors `probe.ProbeReading.values`, so the mapping is direct.

*Raw and calibrated are separate columns.* Same reason `ProbeReading` keeps them
apart: a raw sensor number must never be silently read as a lab-equivalent one.
`calibrated_value` stays NULL until a calibration exists, and NULL is meaningful
-- it means "uncalibrated", which the UI must show rather than hide.

*Lab tests are first class.* SSURGO carries pH/CEC/OM but never Mehlich-3 P/K or
buffer pH, and no field probe measures buffer pH either. A real AASL soil report
is the only thing that unlocks P/K advice, so it gets a real table -- and it makes
the app useful today with no working hardware at all.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = _REPO_ROOT / "data" / "strata.db"

SCHEMA = """
PRAGMA foreign_keys = ON;

-- A field the farmer works. `county_ansi` ties it to the county-scale public
-- data; `crop` selects which per-crop model applies.
CREATE TABLE IF NOT EXISTS field (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    county_ansi TEXT    NOT NULL,
    county_name TEXT,
    crop        TEXT    NOT NULL,
    acres       REAL,
    notes       TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- A spot within a field where readings are taken. Location is what lets Strata
-- line up repeat visits to the same place, and what lets a farmer find a device
-- left logging in tall corn.
CREATE TABLE IF NOT EXISTS site (
    id       INTEGER PRIMARY KEY,
    field_id INTEGER NOT NULL REFERENCES field(id) ON DELETE CASCADE,
    label    TEXT    NOT NULL,
    lat      REAL,
    lon      REAL,
    notes    TEXT
);

-- A fitted probe->lab mapping for one channel (see calibration.ProbeCalibration).
-- `synthetic` marks a fit made on generated pairs; it exists so the UI can never
-- present a synthetic calibration as a real one.
CREATE TABLE IF NOT EXISTS calibration (
    id         INTEGER PRIMARY KEY,
    channel    TEXT    NOT NULL,
    slope      REAL    NOT NULL,
    intercept  REAL    NOT NULL,
    r2         REAL,
    rmse       REAL,
    mean_bias  REAL,
    n_pairs    INTEGER NOT NULL,
    synthetic  INTEGER NOT NULL DEFAULT 0,
    fitted_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    notes      TEXT
);

-- One channel's value at one place and moment. Long/narrow on purpose.
CREATE TABLE IF NOT EXISTS reading (
    id               INTEGER PRIMARY KEY,
    site_id          INTEGER NOT NULL REFERENCES site(id) ON DELETE CASCADE,
    taken_at         TEXT    NOT NULL,
    channel          TEXT    NOT NULL,
    raw_value        REAL    NOT NULL,
    calibrated_value REAL,
    calibration_id   INTEGER REFERENCES calibration(id),
    device_id        TEXT,
    source           TEXT    NOT NULL DEFAULT 'device',
    UNIQUE (site_id, taken_at, channel)
);

-- A real laboratory soil test. The ONLY source of Mehlich-3 P/K and buffer pH,
-- and therefore the only thing that unlocks quantified P/K and lime advice.
CREATE TABLE IF NOT EXISTS lab_test (
    id             INTEGER PRIMARY KEY,
    field_id       INTEGER NOT NULL REFERENCES field(id) ON DELETE CASCADE,
    sampled_at     TEXT    NOT NULL,
    lab_name       TEXT,
    ph             REAL,
    buffer_ph      REAL,
    mehlich3_p_ppm REAL,
    mehlich3_k_ppm REAL,
    cec            REAL,
    om_pct         REAL,
    yield_goal     REAL,
    notes          TEXT
);

CREATE INDEX IF NOT EXISTS idx_reading_site_time ON reading(site_id, taken_at);
CREATE INDEX IF NOT EXISTS idx_reading_channel   ON reading(channel);
CREATE INDEX IF NOT EXISTS idx_site_field        ON site(field_id);
CREATE INDEX IF NOT EXISTS idx_labtest_field     ON lab_test(field_id, sampled_at);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open a connection with rows as dict-likes and foreign keys enforced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Create the schema if absent and return an open connection."""
    conn = connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# --------------------------------------------------------------------------- #
# Reads the app needs. Writes go through plain SQL at the call site; these exist
# because each one encodes a rule worth stating once.
# --------------------------------------------------------------------------- #
def get_fields(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM field ORDER BY name").fetchall()


def get_field(conn: sqlite3.Connection, field_id: int) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM field WHERE id = ?", (field_id,)).fetchone()


def latest_lab_test(conn: sqlite3.Connection, field_id: int) -> Optional[sqlite3.Row]:
    """The most recent lab test for a field.

    Most recent, not merged across tests: soil chemistry changes after liming and
    fertilizing, so combining a 2024 pH with a 2026 P reading would describe a
    field that never existed.
    """
    return conn.execute(
        "SELECT * FROM lab_test WHERE field_id = ? ORDER BY sampled_at DESC LIMIT 1",
        (field_id,),
    ).fetchone()


def sites_for_field(conn: sqlite3.Connection, field_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM site WHERE field_id = ? ORDER BY label", (field_id,)
    ).fetchall()


def readings_for_site(conn: sqlite3.Connection, site_id: int,
                      channel: Optional[str] = None) -> list[sqlite3.Row]:
    """A site's readings oldest-first, optionally one channel.

    Oldest-first because these are time series: the point of a probe is the shape
    over a season, not the newest value.
    """
    sql = "SELECT * FROM reading WHERE site_id = ?"
    params: list = [site_id]
    if channel is not None:
        sql += " AND channel = ?"
        params.append(channel)
    return conn.execute(sql + " ORDER BY taken_at", params).fetchall()


def reading_summary(conn: sqlite3.Connection, field_id: int) -> list[sqlite3.Row]:
    """Per-channel coverage across a whole field: how much data, over what span,
    and how much of it is still uncalibrated.

    `n_uncalibrated` is the honesty column. A field can look data-rich while every
    number in it is a raw sensor value no one has tied to a reference scale.
    """
    return conn.execute(
        """
        SELECT r.channel,
               COUNT(*)                                    AS n,
               MIN(r.taken_at)                             AS first_at,
               MAX(r.taken_at)                             AS last_at,
               SUM(CASE WHEN r.calibrated_value IS NULL
                        THEN 1 ELSE 0 END)                 AS n_uncalibrated
        FROM reading r
        JOIN site s ON s.id = r.site_id
        WHERE s.field_id = ?
        GROUP BY r.channel
        ORDER BY r.channel
        """,
        (field_id,),
    ).fetchall()
