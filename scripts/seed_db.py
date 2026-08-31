"""Create the app database and seed it with one demo field.

    python scripts/seed_db.py           # create + seed (refuses to clobber)
    python scripts/seed_db.py --reset   # delete and recreate

WHAT IS REAL AND WHAT IS NOT, because the distinction is the project's rule:

  REAL      The county context (Chester County FIPS 029) and the pH / CEC / OM
            below, which are this county's actual SSURGO topsoil summary.
  DEMO      The field, its acreage, its three sites, and their coordinates. A
            plausible Chester County farm, not a real one.
  DEMO      The Mehlich-3 P/K and buffer pH on the lab test. These CANNOT come
            from SSURGO -- only a real AASL report supplies them -- so they are
            illustrative values chosen to exercise the below-optimum rules.
  SYNTHETIC The probe readings. No device data exists yet. Every seeded reading
            is written with source='synthetic' so the app can mark it, and none
            of it is ever a finding.

Replace all of it with real data as it arrives; nothing here is load-bearing.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from strata.db import DB_PATH, init_db  # noqa: E402

_SEED = 42
_CHESTER_FIPS = "029"
_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

# Three sites in one field, spread far enough apart to differ in drainage --
# which is the whole reason to take more than one reading.
_SITES = [
    ("Upper bench",  39.9640, -75.6300, "Well drained, top of slope"),
    ("Mid slope",    39.9628, -75.6288, "Between the bench and the swale"),
    ("Low swale",    39.9615, -75.6275, "Holds water after heavy rain"),
]


def _real_chester_soil() -> dict:
    """Chester County's actual SSURGO topsoil summary."""
    df = pd.read_csv(_PROCESSED / "ssurgo_pa_soil.csv", dtype={"county_ansi": "string"})
    row = df[df["county_ansi"] == _CHESTER_FIPS]
    if row.empty:
        raise SystemExit(
            "No SSURGO row for Chester County (029). Reproduce data/processed/ "
            "per the README provenance table first."
        )
    r = row.iloc[0]
    return {"ph": float(r["topsoil_ph"]), "cec": float(r["topsoil_cec"]),
            "om_pct": float(r["topsoil_om_pct"])}


def _synthetic_readings(rng: random.Random, site_idx: int, start: datetime,
                        n_visits: int = 12) -> list[tuple]:
    """SYNTHETIC probe readings: weekly visits across one growing season.

    Shaped to be *plausible*, not accurate -- moisture dries down through summer
    and spikes after rain, CO2 rises as soil warms and biology wakes up, and the
    low swale sits wetter than the upper bench. It exists to give the UI realistic
    shapes to render (and to make a wrong axis or a broken sort obvious). It is
    labeled synthetic everywhere and is never a finding.
    """
    rows = []
    wetness = [-3.0, 0.0, +4.5][site_idx]           # swale holds water
    for visit in range(n_visits):
        taken = start + timedelta(days=7 * visit)
        season = visit / max(n_visits - 1, 1)        # 0 at planting, 1 at harvest
        rained = rng.random() < 0.25

        moisture = 26.0 + wetness - 9.0 * season + (6.0 if rained else 0.0)
        moisture += rng.gauss(0, 1.2)
        temp = 12.0 + 12.0 * math.sin(math.pi * season) + rng.gauss(0, 0.8)
        co2 = 800 + 1500 * math.sin(math.pi * season) + rng.gauss(0, 90)
        ph = 5.7 + wetness * 0.02 + rng.gauss(0, 0.06)
        n = 14 + 6 * (1 - season) + rng.gauss(0, 2.0)
        p = 21 + rng.gauss(0, 2.5)
        k = 88 + rng.gauss(0, 6.0)

        for channel, value in [
            ("soil_moisture", max(moisture, 3.0)), ("soil_temp", temp),
            ("co2", max(co2, 300)), ("ph", ph),
            ("nitrogen", max(n, 0)), ("phosphorus", max(p, 0)), ("potassium", max(k, 0)),
        ]:
            rows.append((taken.isoformat(timespec="seconds"), channel, round(value, 2)))
    return rows


def seed(db_path: Path) -> None:
    rng = random.Random(_SEED)
    soil = _real_chester_soil()
    conn = init_db(db_path)

    with conn:
        field_id = conn.execute(
            """INSERT INTO field (name, county_ansi, county_name, crop, acres, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("North Field", _CHESTER_FIPS, "Chester", "corn_grain_yield", 42.0,
             "DEMO field. County context is real Chester County; the field itself "
             "is illustrative."),
        ).lastrowid

        season_start = datetime(2026, 5, 4, 9, 0)
        for i, (label, lat, lon, note) in enumerate(_SITES):
            site_id = conn.execute(
                "INSERT INTO site (field_id, label, lat, lon, notes) VALUES (?,?,?,?,?)",
                (field_id, label, lat, lon, note),
            ).lastrowid
            conn.executemany(
                """INSERT INTO reading (site_id, taken_at, channel, raw_value, source)
                   VALUES (?, ?, ?, ?, 'synthetic')""",
                [(site_id, t, ch, v) for t, ch, v in
                 _synthetic_readings(rng, i, season_start)],
            )

        # The lab test. pH/CEC/OM are Chester County's real SSURGO values; P/K and
        # buffer pH are illustrative, because SSURGO does not carry them and only a
        # real AASL report can. Chosen below optimum so the rules have something to
        # say -- with these absent the engine correctly refuses to advise at all.
        conn.execute(
            """INSERT INTO lab_test (field_id, sampled_at, lab_name, ph, buffer_ph,
                                     mehlich3_p_ppm, mehlich3_k_ppm, cec, om_pct, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (field_id, "2026-04-12", "DEMO (not a real report)",
             soil["ph"], 6.50, 22.0, 85.0, soil["cec"], soil["om_pct"],
             "pH/CEC/OM are real Chester County SSURGO values; Mehlich-3 P/K and "
             "buffer pH are ILLUSTRATIVE -- replace with a real AASL report."),
        )

    # A SECOND demo field that exists to exercise the refusal states, because a
    # prototype where every path succeeds hides exactly the screens this product is
    # unusual for. Oats is below the product gate (LCO R^2 ~0.14), and this field
    # has no lab test, so it renders WITHHELD and NEEDS LAB TEST at once. Delete it
    # once there are real fields; keep one like it while designing the UI.
    with conn:
        conn.execute(
            """INSERT INTO field (name, county_ansi, county_name, crop, acres, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("Back Lot", _CHESTER_FIPS, "Chester", "oats_yield", 11.0,
             "DEMO field for the refusal states: oats falls below the product gate, "
             "and there is deliberately no lab test on file."),
        )

    n_readings = conn.execute("SELECT COUNT(*) FROM reading").fetchone()[0]
    print(f"Seeded {db_path}")
    print(f"  'North Field'  Chester County, corn -- {len(_SITES)} sites, "
          f"{n_readings} SYNTHETIC readings, 1 demo lab test")
    print("  'Back Lot'     Chester County, oats -- no lab test "
          "(exercises the WITHHELD + NEEDS LAB TEST states)")
    print(f"  real values used: pH {soil['ph']:.2f}, CEC {soil['cec']:.1f}, "
          f"OM {soil['om_pct']:.1f}% (Chester County SSURGO)")
    conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reset", action="store_true",
                    help="delete an existing database first")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    args = ap.parse_args()

    if args.db.exists():
        if not args.reset:
            print(f"{args.db} already exists. Use --reset to recreate it.",
                  file=sys.stderr)
            return 1
        args.db.unlink()
        print(f"Deleted {args.db}")

    seed(args.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
