"""data_ingest.py — source-agnostic data loaders for Strata.

This turn implements ONE source end to end: the USDA NASS Quick Stats API,
returning county-level crop yields for Pennsylvania as a tidy DataFrame.

Design constraints honored here:
  * Source-agnostic via config: no query parameters are hard-coded. Each loader
    reads `config/sources.yaml` and pulls a *named query* (e.g. "corn_grain_yield").
  * Documented schema: `NASSYieldLoader.SCHEMA` states every output column + type.
  * Observability: row counts and missingness are logged at every step, so we can
    see when real PA data is thin instead of silently masking it.
  * Reproducibility: deterministic transforms only; the API key is read from the
    environment (never embedded) so the code can be shared safely.

Later sources (SSURGO/SDA, KSSL pedons, Daymet) will subclass `DataSource` and
plug into the same logging + schema discipline. They are NOT implemented yet.
"""

from __future__ import annotations

import logging
import os
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter

# Load .env (NASS_API_KEY) if python-dotenv is installed. Optional so the module
# still imports in environments that inject env vars another way.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a convenience, not a requirement
    pass


# Module-level logger. main.py configures handlers/levels; we just emit here.
logger = logging.getLogger("strata.data_ingest")


# Repo-root-relative default config path. __file__ = src/strata/data_ingest.py,
# so three parents up is the repo root that holds config/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "sources.yaml"


def load_source_config(path: Path | str = _DEFAULT_CONFIG) -> Dict[str, Any]:
    """Read and parse config/sources.yaml.

    Kept as a free function (not a method) so every loader shares one parser and
    so tests can point it at a fixture file. Does NO network I/O.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source config not found at {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.debug("Loaded source config from %s with sections: %s", path, list(cfg))
    return cfg


class _SystemTrustAdapter(HTTPAdapter):
    """A requests adapter that pins a specific SSLContext onto the connection pool.

    We use it to inject an SSLContext that trusts the OPERATING-SYSTEM certificate
    store (via the `truststore` package). This is necessary on machines where AV
    or a corporate proxy intercepts HTTPS and re-signs server certs with a private
    root: that root is trusted by Windows/macOS but is absent from certifi's
    bundle, so default `requests` verification fails. Crucially, this keeps
    certificate verification ENABLED — it just points it at the right trust store.
    """

    def __init__(self, ssl_context: ssl.SSLContext, *args: Any, **kwargs: Any) -> None:
        self._ssl_context = ssl_context
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        kwargs["ssl_context"] = self._ssl_context
        return super().init_poolmanager(*args, **kwargs)


def build_session(use_system_trust_store: bool = True) -> requests.Session:
    """Create a requests.Session with the appropriate TLS trust configuration.

    If `use_system_trust_store` is True and `truststore` is importable, the
    session validates certificates against the OS trust store (the fix for the
    local Norton AV HTTPS interception). Otherwise it returns a plain session
    using certifi — and logs a clear warning so a failure here is never silent
    or mistaken for a code bug.
    """
    session = requests.Session()
    if not use_system_trust_store:
        return session
    try:
        import truststore  # imported lazily so the dep is only needed when enabled

        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        adapter = _SystemTrustAdapter(ctx)
        session.mount("https://", adapter)
        logger.debug("TLS configured to use the OS trust store (truststore).")
    except ImportError:
        logger.warning(
            "use_system_trust_store=True but `truststore` is not installed. "
            "Falling back to certifi; HTTPS may fail if your machine inspects TLS "
            "(e.g. Norton/corporate proxy). Install truststore to fix."
        )
    return session


class DataSource(ABC):
    """Abstract base for every Strata data loader.

    The contract is intentionally tiny so all sources behave the same:
      * fetch()  -> raw payload from the source (the only network step)
      * tidy()   -> a documented, analysis-ready DataFrame
      * load()   -> fetch() then tidy(), with logging around both

    WHY a base class: the planning constraints ask for one consistent ingest
    discipline (row-count + missingness logging, documented schemas) across
    several very different sources. Centralizing it here prevents drift.
    """

    #: Subclasses document their output columns here as {column: description}.
    SCHEMA: Dict[str, str] = {}

    @abstractmethod
    def fetch(self) -> Any:
        """Hit the source and return its raw payload. The ONLY network step."""
        raise NotImplementedError

    @abstractmethod
    def tidy(self, raw: Any) -> pd.DataFrame:
        """Transform a raw payload into a tidy DataFrame matching SCHEMA."""
        raise NotImplementedError

    def load(self) -> pd.DataFrame:
        """fetch() then tidy(), logging the shape and missingness of the result."""
        logger.info("[%s] fetching ...", type(self).__name__)
        raw = self.fetch()
        df = self.tidy(raw)
        self._log_frame(df)
        return df

    def _log_frame(self, df: pd.DataFrame) -> None:
        """Log row/column counts and per-column missingness.

        This is the 'observability' requirement: at every step we want to SEE
        how much real data we have and where it is thin, rather than discovering
        gaps later or hiding them behind fills.
        """
        logger.info("[%s] rows=%d cols=%d", type(self).__name__, len(df), df.shape[1])
        if df.empty:
            logger.warning("[%s] returned ZERO rows — check query parameters.", type(self).__name__)
            return
        missing = df.isna().mean().sort_values(ascending=False)
        worst = missing[missing > 0]
        if worst.empty:
            logger.info("[%s] no missing values in any column.", type(self).__name__)
        else:
            for col, frac in worst.items():
                logger.info("[%s] missing %5.1f%%  %s", type(self).__name__, frac * 100, col)


@dataclass
class NASSYieldLoader(DataSource):
    """Loader for USDA NASS Quick Stats county-level crop yields.

    Parameters
    ----------
    query_name:
        Name of a query block under `nass.queries` in sources.yaml
        (e.g. "corn_grain_yield", "soybean_yield", "hay_yield").
    config:
        Parsed sources.yaml. If None, it is loaded from the default path.
    api_key:
        NASS API key. If None, read from the env var named by `nass.api_key_env`
        (default "NASS_API_KEY"). Never hard-code it in source.
    timeout:
        Per-request timeout in seconds (defensive; NASS can be slow).
    """

    query_name: str
    config: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None
    timeout: int = 60
    session: Optional[requests.Session] = None  # reusable TLS-configured session

    # Documented output schema. Agronomists read these columns directly, so the
    # names stay human-meaningful and the yield units are explicit.
    SCHEMA: Dict[str, str] = field(
        default_factory=lambda: {
            "state_alpha": "State postal code (always 'PA' for our queries).",
            "county_name": "County name as reported by NASS (e.g. 'CHESTER').",
            "county_ansi": "3-digit county FIPS/ANSI code; stable join key across sources.",
            "year": "Calendar year of the yield estimate (int).",
            "commodity_desc": "Crop, e.g. 'CORN', 'SOYBEANS', 'HAY'.",
            "short_desc": "NASS's exact series label — proves which measurement this is.",
            "unit_desc": "Yield unit, e.g. 'BU / ACRE' or 'TONS / ACRE'.",
            "yield_value": "Numeric yield in unit_desc. NaN where NASS suppressed the value.",
        }
    )

    def __post_init__(self) -> None:
        # Resolve config and the specific named query once, up front, so any
        # misconfiguration fails loudly before we ever touch the network.
        if self.config is None:
            self.config = load_source_config()
        if "nass" not in self.config:
            raise KeyError("sources.yaml has no 'nass' section.")
        self._nass = self.config["nass"]

        queries = self._nass.get("queries", {})
        if self.query_name not in queries:
            raise KeyError(
                f"Query '{self.query_name}' not in nass.queries. "
                f"Available: {sorted(queries)}"
            )
        self._query = queries[self.query_name]

        # Resolve the API key from the environment if not passed explicitly.
        if self.api_key is None:
            env_name = self._nass.get("api_key_env", "NASS_API_KEY")
            self.api_key = os.environ.get(env_name)

        # Build a TLS-configured session if one wasn't supplied. The global
        # http.use_system_trust_store flag controls whether we trust the OS store.
        if self.session is None:
            use_store = self.config.get("http", {}).get("use_system_trust_store", True)
            self.session = build_session(use_system_trust_store=use_store)

    def build_params(self) -> Dict[str, Any]:
        """Merge shared defaults + this named query into the full param set.

        Separated from the network call so the exact request can be inspected,
        tested, and printed WITHOUT making a call. The API key is intentionally
        excluded here (added only at request time) to keep it out of logs.
        """
        params: Dict[str, Any] = {}
        params.update(self._nass.get("defaults", {}))  # shared: PA, COUNTY, SURVEY, ...
        params.update(self._query)                     # specific: commodity, short_desc, year__GE
        return params

    def build_query_url(self, include_key: bool = False) -> str:
        """Return the exact GET URL for inspection.

        By default the API key is shown as a placeholder so the URL is safe to
        paste into notes/logs. Pass include_key=True only for an actual request.
        """
        params = self.build_params()
        key = self.api_key if (include_key and self.api_key) else "YOUR_KEY"
        # Key first to mirror NASS's own examples; urlencode handles spaces/slashes.
        query_string = "key=" + key + "&" + urlencode(params)
        return self._nass["base_url"] + "?" + query_string

    def fetch(self) -> Dict[str, Any]:
        """Call the NASS Quick Stats API and return the parsed JSON dict.

        Raises a clear error if the API key is missing or NASS returns a non-200.
        This is the ONLY method that performs network I/O.
        """
        if not self.api_key:
            env_name = self._nass.get("api_key_env", "NASS_API_KEY")
            raise RuntimeError(
                f"No NASS API key. Set {env_name} in your environment / .env file."
            )
        params = self.build_params()
        params["key"] = self.api_key  # added only now, never persisted in build_params()

        logger.info("[NASSYieldLoader] GET %s", self.build_query_url(include_key=False))
        resp = self.session.get(self._nass["base_url"], params=params, timeout=self.timeout)

        # NASS returns 400 with a JSON {"error": [...]} body for bad/empty queries.
        if resp.status_code != 200:
            raise RuntimeError(
                f"NASS API returned HTTP {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json()

    def tidy(self, raw: Dict[str, Any]) -> pd.DataFrame:
        """Turn the raw NASS JSON into a tidy DataFrame matching SCHEMA.

        Steps, each deliberate:
          1. Pull the 'data' record list (NASS wraps results in {"data": [...]}).
          2. Keep only the documented columns (NASS returns ~40 we don't need).
          3. Coerce types: year -> int; the yield value -> float, treating NASS's
             suppression markers (e.g. '(D)', '(Z)') and thousands commas safely.
          4. Sort for stable, reproducible output.
        """
        records = raw.get("data", [])
        if not records:
            logger.warning("[NASSYieldLoader] NASS payload had no 'data' records.")
            return pd.DataFrame(columns=list(self.SCHEMA))

        df = pd.DataFrame.from_records(records)

        # NASS calls the numeric yield column 'Value'. Rename to our schema name.
        df = df.rename(columns={"Value": "yield_value"})

        # Keep only documented columns that are actually present.
        keep = [c for c in self.SCHEMA if c in df.columns]
        df = df[keep].copy()

        # Year as integer for clean joins/sorts.
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

        # Yield value: NASS uses '(D)'/'(Z)'/'(NA)' for suppressed/withheld values
        # and embeds thousands separators. Strip commas, then coerce; anything
        # non-numeric (the suppression markers) becomes NaN — surfaced, not faked.
        if "yield_value" in df.columns:
            cleaned = df["yield_value"].astype(str).str.replace(",", "", regex=False)
            df["yield_value"] = pd.to_numeric(cleaned, errors="coerce")

        # Deterministic ordering so repeated runs produce identical output.
        sort_cols = [c for c in ("commodity_desc", "county_name", "year") if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        return df

    @classmethod
    def load_crops(
        cls,
        query_names: list[str],
        config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load several named crop queries and stack them into one tidy frame.

        WHY a classmethod here: every crop shares the identical schema, so the
        per-crop frames concatenate cleanly into the long-format yield table the
        rest of Strata expects (one row per county-year-crop). This is the
        convenience seam for pulling 'all 8 crops' in one call.

        Each crop is loaded independently and tagged with `query_name` so its
        provenance is traceable, and a per-crop row count is logged so we can SEE
        which crops PA actually publishes at county level (thin crops show up as
        small or zero counts rather than being silently dropped).
        """
        # Parse the config ONCE and pass it into each loader, so we don't re-read
        # sources.yaml from disk for every crop.
        if config is None:
            config = load_source_config()

        # One shared, TLS-configured session for all crops (connection reuse).
        use_store = config.get("http", {}).get("use_system_trust_store", True)
        session = build_session(use_system_trust_store=use_store)

        frames: list[pd.DataFrame] = []
        for name in query_names:
            loader = cls(query_name=name, config=config, api_key=api_key, session=session)
            df = loader.load()
            # Tag provenance: which named query produced these rows.
            df.insert(0, "query_name", name)
            logger.info("[load_crops] %-20s -> %5d rows", name, len(df))
            frames.append(df)

        if not frames:
            return pd.DataFrame()  # pragma: no cover - empty query_names list

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "[load_crops] combined %d crops -> %d total rows", len(query_names), len(combined)
        )
        return combined


# A tiny manual entry point so you can run this one source in isolation later
# (after approval). Intentionally guarded so importing the module is side-effect free.
@dataclass
class SsurgoSoilLoader(DataSource):
    """Loader for USDA SSURGO soil properties via Soil Data Access (SDA).

    One instance = one SSURGO survey area (≈ one county). `load_areas(...)` loops
    all PA counties. The output is ONE tidy row per county: an area/component/
    depth-weighted summary of topsoil properties plus the dominant soil taxonomic
    order and drainage class. That county summary is static (SSURGO has no time
    dimension), so it joins to every NASS county-year yield row on county FIPS.

    GRANULARITY NOTE (stated up front, per the council's rigor rule): a single
    county-level soil summary is a coarse abstraction of spatially variable soil.
    It is a *context feature* for the county-year yield model, NOT a claim about
    any specific field. The future cheap-probe calibration layer is what will add
    field-specific resolution.

    Parameters
    ----------
    areasymbol : e.g. 'PA029' (Chester County).
    """

    areasymbol: str
    config: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None  # unused (SDA needs no key); kept for interface parity
    session: Optional[requests.Session] = None
    timeout: int = 180

    SCHEMA: Dict[str, str] = field(
        default_factory=lambda: {
            "county_ansi": "3-digit county FIPS parsed from areasymbol; join key to NASS.",
            "areasymbol": "SSURGO survey area code, e.g. 'PA029'.",
            "areaname": "Human-readable survey area name.",
            "n_mapunits": "Number of distinct map units summarized (data-density indicator).",
            "dominant_taxorder": "Area×component-weighted most common Soil Taxonomy order.",
            "dominant_drainage": "Area×component-weighted most common drainage class.",
            "topsoil_sand_pct": "Weighted mean sand % (0-25 cm).",
            "topsoil_silt_pct": "Weighted mean silt % (0-25 cm).",
            "topsoil_clay_pct": "Weighted mean clay % (0-25 cm).",
            "topsoil_ph": "Weighted mean 1:1 water pH (0-25 cm).",
            "topsoil_om_pct": "Weighted mean organic matter % (0-25 cm).",
            "topsoil_cec": "Weighted mean CEC at pH 7, meq/100g (0-25 cm).",
        }
    )

    # --- SQL templates. Kept here (not in YAML) because they ARE the loader's
    # logic, not a tunable knob. {asym} is the only substitution. ---

    # Per-mapunit RELATIVE area weight. STArea() is square-degrees (geometry is in
    # lat/lon); absolute units are irrelevant — we only need each map unit's area
    # RELATIVE to others in the same county, and lat/lon distortion across one
    # county is negligible.
    _SQL_AREA = (
        "SELECT mukey, SUM(mupolygongeo.STArea()) AS area "
        "FROM mupolygon WHERE areasymbol = '{asym}' GROUP BY mukey"
    )

    # Component + horizon properties. _r = SSURGO 'representative' value. We keep
    # every component and weight by comppct_r later (minor components self-downweight).
    _SQL_PROPS = (
        "SELECT mu.mukey, c.cokey, c.comppct_r, c.taxorder, c.drainagecl, "
        "ch.hzdept_r, ch.hzdepb_r, ch.sandtotal_r, ch.silttotal_r, ch.claytotal_r, "
        "ch.ph1to1h2o_r, ch.om_r, ch.cec7_r "
        "FROM legend l "
        "JOIN mapunit mu ON mu.lkey = l.lkey "
        "JOIN component c ON c.mukey = mu.mukey "
        "JOIN chorizon ch ON ch.cokey = c.cokey "
        "WHERE l.areasymbol = '{asym}' AND ch.hzdept_r IS NOT NULL"
    )

    # Numeric horizon property columns we depth-weight over the topsoil window.
    _PROP_COLS = [
        "sandtotal_r", "silttotal_r", "claytotal_r", "ph1to1h2o_r", "om_r", "cec7_r",
    ]
    _PROP_OUT = {
        "sandtotal_r": "topsoil_sand_pct", "silttotal_r": "topsoil_silt_pct",
        "claytotal_r": "topsoil_clay_pct", "ph1to1h2o_r": "topsoil_ph",
        "om_r": "topsoil_om_pct", "cec7_r": "topsoil_cec",
    }

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = load_source_config()
        if "ssurgo" not in self.config:
            raise KeyError("sources.yaml has no 'ssurgo' section.")
        self._cfg = self.config["ssurgo"]
        self._depth = float(self._cfg.get("topsoil_depth_cm", 25))
        if self.session is None:
            use_store = self.config.get("http", {}).get("use_system_trust_store", True)
            self.session = build_session(use_system_trust_store=use_store)

    def _post(self, sql: str) -> pd.DataFrame:
        """POST one SQL query to SDA and return its 'Table' as a DataFrame.

        SDA returns {"Table": [[header...], [row...], ...]} for JSON+COLUMNNAME;
        the first element is the column header row. An empty/!200 response yields
        an empty frame (logged by the caller), never a crash.
        """
        body = {"query": sql, "format": self._cfg.get("response_format", "JSON+COLUMNNAME")}
        resp = self.session.post(self._cfg["base_url"], json=body, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"SDA HTTP {resp.status_code}: {resp.text[:300]}")
        table = resp.json().get("Table", [])
        if not table:
            return pd.DataFrame()
        header, *rows = table
        return pd.DataFrame(rows, columns=header)

    def fetch(self) -> Dict[str, pd.DataFrame]:
        """Run the two SDA queries (area weights + properties) for this county."""
        asym = self.areasymbol
        logger.info("[SsurgoSoilLoader] querying SDA for %s ...", asym)
        area = self._post(self._SQL_AREA.format(asym=asym))
        props = self._post(self._SQL_PROPS.format(asym=asym))
        return {"area": area, "props": props}

    def tidy(self, raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aggregate component/horizon data into ONE county-level summary row.

        Three transparent weighting steps:
          (1) Horizons -> component: depth-weight each property over the 0-`depth`
              cm window by the thickness each horizon contributes to that window.
          (2) Components -> county weight: each component's weight = its map unit's
              area * comppct_r/100 (area-and-abundance weighting).
          (3) County summary: weighted mean of each property across components
              (skipping nulls), and the weight-argmax for the categorical class
              and drainage fields.
        """
        area_df, props = raw["area"], raw["props"]
        if props.empty:
            logger.warning("[SsurgoSoilLoader] %s returned no property rows.", self.areasymbol)
            return pd.DataFrame(columns=list(self.SCHEMA))

        # Coerce all numeric columns from SDA's string payload.
        num_cols = ["comppct_r", "hzdept_r", "hzdepb_r", *self._PROP_COLS]
        for col in num_cols:
            props[col] = pd.to_numeric(props[col], errors="coerce")
        area_df = area_df.copy()
        area_df["area"] = pd.to_numeric(area_df["area"], errors="coerce")

        # (1) Topsoil overlap thickness for each horizon: how many cm of this
        # horizon fall inside [0, depth]. Horizons entirely below the window get 0.
        top = self._depth
        overlap = (props["hzdepb_r"].clip(upper=top) - props["hzdept_r"].clip(lower=0))
        props["thick"] = overlap.clip(lower=0)
        props = props[props["thick"] > 0]  # keep only horizons touching the topsoil window

        # (1 cont.) Depth-weighted mean per component for each property. We weight
        # by thickness but only over horizons where that property is non-null, so a
        # missing value never silently pulls a component's mean toward zero.
        comp_rows = []
        group_keys = ["mukey", "cokey", "comppct_r", "taxorder", "drainagecl"]
        for keys, grp in props.groupby(["mukey", "cokey"], dropna=False):
            row = {"mukey": keys[0], "cokey": keys[1]}
            # carry component-level attributes (constant within a component)
            row["comppct_r"] = grp["comppct_r"].iloc[0]
            row["taxorder"] = grp["taxorder"].iloc[0]
            row["drainagecl"] = grp["drainagecl"].iloc[0]
            for col in self._PROP_COLS:
                sub = grp[["thick", col]].dropna()
                if sub.empty or sub["thick"].sum() == 0:
                    row[col] = pd.NA
                else:
                    row[col] = (sub[col] * sub["thick"]).sum() / sub["thick"].sum()
            comp_rows.append(row)
        comp = pd.DataFrame(comp_rows)

        # (2) County weight per component = map-unit area * component percent.
        comp = comp.merge(area_df, on="mukey", how="left")
        comp["area"] = comp["area"].fillna(0.0)
        comp["weight"] = comp["area"] * comp["comppct_r"].fillna(0.0)
        comp = comp[comp["weight"] > 0]  # drop zero-weight (no area or no comppct)

        # (3a) Weighted means of continuous properties, skipping nulls per property.
        out: Dict[str, Any] = {}
        for src, dst in self._PROP_OUT.items():
            sub = comp[["weight", src]].dropna()
            if sub.empty or sub["weight"].sum() == 0:
                out[dst] = pd.NA
            else:
                out[dst] = round((sub[src] * sub["weight"]).sum() / sub["weight"].sum(), 2)

        # (3b) Dominant categorical = class with the largest summed weight.
        def _dominant(colname: str) -> Any:
            sub = comp[["weight", colname]].dropna()
            if sub.empty:
                return pd.NA
            return sub.groupby(colname)["weight"].sum().idxmax()

        out["dominant_taxorder"] = _dominant("taxorder")
        out["dominant_drainage"] = _dominant("drainagecl")

        # Identity / provenance columns.
        out["county_ansi"] = self.areasymbol[2:]  # 'PA029' -> '029'
        out["areasymbol"] = self.areasymbol
        out["areaname"] = self._areaname or ""
        out["n_mapunits"] = int(comp["mukey"].nunique())

        # Order columns to the documented schema.
        return pd.DataFrame([{k: out.get(k, pd.NA) for k in self.SCHEMA}])

    # Filled in by load_areas (one cheap legend query) so each county row is named.
    _areaname: str = ""

    @classmethod
    def list_pa_areas(
        cls, config: Optional[Dict[str, Any]] = None, session: Optional[requests.Session] = None
    ) -> pd.DataFrame:
        """Return all SSURGO survey areas for the configured state (areasymbol, areaname)."""
        if config is None:
            config = load_source_config()
        cfg = config["ssurgo"]
        if session is None:
            use_store = config.get("http", {}).get("use_system_trust_store", True)
            session = build_session(use_system_trust_store=use_store)
        prefix = cfg.get("state_areasymbol_prefix", "PA")
        sql = (
            f"SELECT areasymbol, areaname FROM legend "
            f"WHERE areasymbol LIKE '{prefix}%' ORDER BY areasymbol"
        )
        body = {"query": sql, "format": cfg.get("response_format", "JSON+COLUMNNAME")}
        resp = session.post(cfg["base_url"], json=body, timeout=120)
        resp.raise_for_status()
        header, *rows = resp.json()["Table"]
        return pd.DataFrame(rows, columns=header)

    @classmethod
    def county_centroids(
        cls,
        config: Optional[Dict[str, Any]] = None,
        areasymbols: Optional[list[str]] = None,
        topn: int = 2000,
    ) -> pd.DataFrame:
        """Compute one representative lat/lon per county from SSURGO geometry.

        For each county we take the centroids of its largest `topn` map-unit
        polygons and average them weighted by polygon area. Using the largest
        polygons keeps the query cheap while covering most of the county's area;
        area-weighting pulls the point toward the bulk of the county. This reuses
        the SSURGO service (no new dataset) to feed Daymet's point-based API.

        Returns columns: county_ansi, areasymbol, areaname, lat, lon.
        """
        if config is None:
            config = load_source_config()
        cfg = config["ssurgo"]
        use_store = config.get("http", {}).get("use_system_trust_store", True)
        session = build_session(use_system_trust_store=use_store)

        areas = cls.list_pa_areas(config=config, session=session)
        name_by_sym = dict(zip(areas["areasymbol"], areas["areaname"]))
        if areasymbols is None:
            areasymbols = list(areas["areasymbol"])

        rows = []
        for asym in areasymbols:
            sql = (
                f"SELECT TOP {int(topn)} mupolygongeo.STCentroid().STY AS lat, "
                f"mupolygongeo.STCentroid().STX AS lon, mupolygongeo.STArea() AS a "
                f"FROM mupolygon WHERE areasymbol = '{asym}' "
                f"ORDER BY mupolygongeo.STArea() DESC"
            )
            body = {"query": sql, "format": cfg.get("response_format", "JSON+COLUMNNAME")}
            resp = session.post(cfg["base_url"], json=body, timeout=180)
            resp.raise_for_status()
            table = resp.json().get("Table", [])
            if not table or len(table) < 2:
                logger.warning("[county_centroids] %s: no polygons returned.", asym)
                continue
            header, *data = table
            poly = pd.DataFrame(data, columns=header).astype(float)
            wsum = poly["a"].sum()
            lat = (poly["lat"] * poly["a"]).sum() / wsum
            lon = (poly["lon"] * poly["a"]).sum() / wsum
            rows.append({
                "county_ansi": asym[2:], "areasymbol": asym,
                "areaname": name_by_sym.get(asym, ""),
                "lat": round(lat, 4), "lon": round(lon, 4),
            })
            logger.info("[county_centroids] %s -> (%.4f, %.4f)", asym, lat, lon)
        return pd.DataFrame(rows)

    @classmethod
    def load_areas(
        cls,
        areasymbols: Optional[list[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Summarize many counties into one tidy frame (one row per county).

        If `areasymbols` is None, every survey area for the configured state is
        pulled. Each county is logged so a slow/empty county is visible.
        """
        if config is None:
            config = load_source_config()
        use_store = config.get("http", {}).get("use_system_trust_store", True)
        session = build_session(use_system_trust_store=use_store)

        areas = cls.list_pa_areas(config=config, session=session)
        name_by_sym = dict(zip(areas["areasymbol"], areas["areaname"]))
        if areasymbols is None:
            areasymbols = list(areas["areasymbol"])

        frames: list[pd.DataFrame] = []
        for i, asym in enumerate(areasymbols, 1):
            loader = cls(areasymbol=asym, config=config, session=session)
            loader._areaname = name_by_sym.get(asym, "")
            df = loader.load()
            logger.info("[load_areas] %3d/%d %s -> %d row(s)", i, len(areasymbols), asym, len(df))
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        logger.info("[load_areas] summarized %d counties.", len(combined))
        return combined


@dataclass
class KSSLPedonLoader(DataSource):
    """Loader for NCSS/KSSL lab pedons via SDA (trains the soil classifier).

    One instance = one US state. `load_region(...)` loops the configured states
    and stacks them. Output is a LAYER-level tidy table: one row per measured
    horizon, carrying the parent pedon's Soil Taxonomy labels and coordinates.

    WHY layer-level (not pre-aggregated to pedons): collapsing a profile's
    horizons into per-pedon features is a *modeling* decision (which depths,
    which weighting) that belongs in soil_classifier.py, where it can be stated
    and varied. Ingest's job is to deliver clean, documented measurements.

    Join chain (all in lab_* tables on SDA):
        lab_combine_nasis_ncss (labels + lat/lon + state_key)
          -> lab_area        (state_key -> postal code, for the state filter)
          -> lab_layer       (pedon_key -> labsampnum + horizon depths)
          -> lab_physical_properties (sand/silt/clay)
          -> lab_chemical_properties (pH, CEC, organic carbon)
    """

    state_code: str
    config: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None  # unused; kept for interface parity
    session: Optional[requests.Session] = None
    timeout: int = 240

    SCHEMA: Dict[str, str] = field(
        default_factory=lambda: {
            "pedon_key": "Stable KSSL pedon (profile) id; groups horizons into one sample.",
            "state": "Two-letter state postal code (the pull's filter).",
            "lat": "Pedon latitude (decimal degrees).",
            "lon": "Pedon longitude (decimal degrees).",
            "tax_order": "Soil Taxonomy ORDER label (classifier target). Coalesced corr>SSL>samp.",
            "tax_greatgroup": "Great group label (alternative finer target).",
            "tax_subgroup": "Subgroup label (finest provided).",
            "labsampnum": "Lab sample number for this horizon (join key to lab properties).",
            "hzn_top": "Horizon top depth, cm.",
            "hzn_bot": "Horizon bottom depth, cm.",
            "hzn_desgn": "Horizon designation, e.g. 'Ap', 'Bt1'.",
            "sand_pct": "Total sand %, pipette/lab.",
            "silt_pct": "Total silt %.",
            "clay_pct": "Total clay %.",
            "ph_h2o": "pH in water (1:1).",
            "cec": "CEC, meq/100g. NH4OAc pH7 preferred; falls back to sum-of-cations pH8.2 where missing (method mix documented).",
            "organic_c_pct": "Organic carbon %, Walkley-Black (or estimated fallback).",
        }
    )

    # Per-state join. Labels are coalesced in pandas (clearer than nested SQL).
    _SQL = (
        "SELECT cn.pedon_key, a.area_code AS state, "
        "cn.latitude_decimal_degrees AS lat, cn.longitude_decimal_degrees AS lon, "
        "cn.corr_taxorder, cn.SSL_taxorder, cn.samp_taxorder, "
        "cn.corr_taxgrtgroup, cn.corr_taxsubgrp, "
        "ly.labsampnum, ly.hzn_top, ly.hzn_bot, ly.hzn_desgn, "
        "pp.sand_total, pp.silt_total, pp.clay_total, "
        "cp.ph_h2o, cp.cec_nh4_ph_7, cp.sum_of_cations_cec_pH_8_2, "
        "cp.organic_carbon_walkley_black, cp.estimated_organic_carbon "
        "FROM lab_combine_nasis_ncss cn "
        "JOIN lab_area a ON a.area_key = cn.state_key "
        "JOIN lab_layer ly ON ly.pedon_key = cn.pedon_key "
        "LEFT JOIN lab_physical_properties pp ON pp.labsampnum = ly.labsampnum "
        "LEFT JOIN lab_chemical_properties cp ON cp.labsampnum = ly.labsampnum "
        "WHERE a.area_code = '{st}'"
    )

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = load_source_config()
        if "kssl" not in self.config:
            raise KeyError("sources.yaml has no 'kssl' section.")
        self._cfg = self.config["kssl"]
        if self.session is None:
            use_store = self.config.get("http", {}).get("use_system_trust_store", True)
            self.session = build_session(use_system_trust_store=use_store)

    def fetch(self) -> pd.DataFrame:
        body = {"query": self._SQL.format(st=self.state_code),
                "format": self._cfg.get("response_format", "JSON+COLUMNNAME")}
        logger.info("[KSSLPedonLoader] querying SDA lab pedons for %s ...", self.state_code)
        resp = self.session.post(self._cfg["base_url"], json=body, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"SDA HTTP {resp.status_code}: {resp.text[:300]}")
        table = resp.json().get("Table", [])
        if not table:
            return pd.DataFrame()
        header, *rows = table
        return pd.DataFrame(rows, columns=header)

    def tidy(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            logger.warning("[KSSLPedonLoader] %s returned no rows.", self.state_code)
            return pd.DataFrame(columns=list(self.SCHEMA))
        df = raw.copy()

        # Coalesce the taxonomic ORDER label: prefer the corrected classification,
        # then the SSL, then the sampled. Lowercased+stripped for clean grouping.
        def _coalesce_lower(cols: list[str]) -> pd.Series:
            out = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
            for col in cols:
                if col in df.columns:
                    vals = df[col].astype("string").str.strip().str.lower()
                    out = out.where(out.notna(), vals)
            return out.replace({"": pd.NA})

        df["tax_order"] = _coalesce_lower(["corr_taxorder", "SSL_taxorder", "samp_taxorder"])
        df["tax_greatgroup"] = _coalesce_lower(["corr_taxgrtgroup"])
        df["tax_subgroup"] = _coalesce_lower(["corr_taxsubgrp"])

        # Organic carbon: prefer measured Walkley-Black, fall back to estimated.
        oc = pd.to_numeric(df.get("organic_carbon_walkley_black"), errors="coerce")
        oc_est = pd.to_numeric(df.get("estimated_organic_carbon"), errors="coerce")
        df["organic_c_pct"] = oc.where(oc.notna(), oc_est)

        # CEC: prefer NH4OAc pH7 (standard); fall back to sum-of-cations pH8.2 where
        # missing. cec_nh4_ph_7 is ~2/3 empty in KSSL, so this recovers coverage.
        cec = pd.to_numeric(df.get("cec_nh4_ph_7"), errors="coerce")
        cec_alt = pd.to_numeric(df.get("sum_of_cations_cec_pH_8_2"), errors="coerce")
        df["cec"] = cec.where(cec.notna(), cec_alt)

        # Rename property columns to the documented schema names.
        df = df.rename(columns={
            "sand_total": "sand_pct", "silt_total": "silt_pct", "clay_total": "clay_pct",
        })

        # Coerce numerics.
        for col in ["lat", "lon", "hzn_top", "hzn_bot", "sand_pct", "silt_pct",
                    "clay_pct", "ph_h2o", "cec"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Keep documented columns only, in order.
        keep = [c for c in self.SCHEMA if c in df.columns]
        df = df[keep].copy()

        # A labsampnum can occasionally yield duplicate property rows (multiple lab
        # preps); keep one row per (pedon, horizon) so layers aren't double-counted.
        before = len(df)
        df = df.drop_duplicates(subset=["pedon_key", "labsampnum"]).reset_index(drop=True)
        if len(df) < before:
            logger.info("[KSSLPedonLoader] %s dropped %d duplicate layer rows.",
                        self.state_code, before - len(df))
        return df

    @classmethod
    def load_region(
        cls,
        states: Optional[list[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Load lab pedons for each configured state and stack them (layer-level)."""
        if config is None:
            config = load_source_config()
        if states is None:
            states = config["kssl"].get("region_states", ["PA"])
        use_store = config.get("http", {}).get("use_system_trust_store", True)
        session = build_session(use_system_trust_store=use_store)

        frames: list[pd.DataFrame] = []
        for st in states:
            loader = cls(state_code=st, config=config, session=session)
            df = loader.load()
            logger.info("[load_region] %s -> %d layers, %d pedons",
                        st, len(df), df["pedon_key"].nunique() if not df.empty else 0)
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        logger.info("[load_region] %d layers across %d states.", len(combined), len(states))
        return combined


@dataclass
class DaymetWeatherLoader(DataSource):
    """Loader for Daymet daily weather, aggregated to county-year features.

    One instance = one county (its centroid lat/lon). A single Daymet Single
    Pixel call returns every day in the configured year range; `tidy` collapses
    those days into ONE row per year of growing-season agronomic features.

    GRANULARITY NOTE (council rule): one centroid point represents the whole
    county's weather. That is coarse for large/topographically varied counties,
    but it matches the county-year unit we already use for yields. Stated, not hidden.
    """

    county_ansi: str
    lat: float
    lon: float
    config: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None  # unused; interface parity
    session: Optional[requests.Session] = None
    timeout: int = 120

    SCHEMA: Dict[str, str] = field(
        default_factory=lambda: {
            "county_ansi": "3-digit county FIPS; join key to NASS/SSURGO.",
            "year": "Calendar year (int).",
            "gdd_base10": "Sum of growing-degree-days (base 10 C) over the season window.",
            "precip_total_mm": "Total precipitation (mm) over the season window.",
            "tmax_mean_c": "Mean daily max temperature (C) over the season.",
            "tmin_mean_c": "Mean daily min temperature (C) over the season.",
            "heat_days_30c": "Count of days with Tmax >= heat threshold (heat stress).",
            "n_season_days": "Days contributing (sanity check; ~214 for Apr-Oct).",
        }
    )

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = load_source_config()
        if "daymet" not in self.config:
            raise KeyError("sources.yaml has no 'daymet' section.")
        self._cfg = self.config["daymet"]
        if self.session is None:
            use_store = self.config.get("http", {}).get("use_system_trust_store", True)
            self.session = build_session(use_system_trust_store=use_store)

    def fetch(self) -> str:
        """GET the Daymet Single Pixel CSV for this county's centroid (all years)."""
        params = {
            "lat": self.lat, "lon": self.lon, "vars": self._cfg.get("vars", "tmax,tmin,prcp"),
            "start": f"{self._cfg['start_year']}-01-01",
            "end": f"{self._cfg['end_year']}-12-31",
        }
        logger.info("[DaymetWeatherLoader] county %s GET Daymet (%.3f, %.3f)",
                    self.county_ansi, self.lat, self.lon)
        resp = self.session.get(self._cfg["base_url"], params=params, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Daymet HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.text

    def tidy(self, raw: str) -> pd.DataFrame:
        """Parse the Daymet CSV and aggregate to one row per year.

        Daymet prepends ~6 metadata lines before the real 'year,yday,...' header,
        and uses a 365-day calendar (Dec 31 dropped in leap years) so a fixed
        yday→season window is valid every year.
        """
        import io

        lines = raw.splitlines()
        # Find the real CSV header (the line that starts with 'year,').
        start = next((i for i, ln in enumerate(lines) if ln.startswith("year,")), None)
        if start is None:
            logger.warning("[DaymetWeatherLoader] county %s: no data header found.", self.county_ansi)
            return pd.DataFrame(columns=list(self.SCHEMA))
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])))

        # Normalize Daymet's verbose column names to short ones.
        rename = {}
        for col in df.columns:
            low = col.lower()
            if low.startswith("prcp"): rename[col] = "prcp"
            elif low.startswith("tmax"): rename[col] = "tmax"
            elif low.startswith("tmin"): rename[col] = "tmin"
            elif low.startswith("yday"): rename[col] = "yday"
            elif low.startswith("year"): rename[col] = "year"
        df = df.rename(columns=rename)

        # Restrict to the growing-season window (inclusive yday bounds).
        s0, s1 = self._cfg["season_start_yday"], self._cfg["season_end_yday"]
        season = df[(df["yday"] >= s0) & (df["yday"] <= s1)].copy()

        # Daily growing-degree-days, base 10 C: max(0, mean(tmax,tmin) - base).
        base = float(self._cfg.get("gdd_base_c", 10))
        season["tmean"] = (season["tmax"] + season["tmin"]) / 2.0
        season["gdd"] = (season["tmean"] - base).clip(lower=0)
        heat = float(self._cfg.get("heat_threshold_c", 30))

        # Aggregate each year into one feature row.
        out = season.groupby("year").apply(
            lambda g: pd.Series({
                "gdd_base10": round(g["gdd"].sum(), 1),
                "precip_total_mm": round(g["prcp"].sum(), 1),
                "tmax_mean_c": round(g["tmax"].mean(), 2),
                "tmin_mean_c": round(g["tmin"].mean(), 2),
                "heat_days_30c": int((g["tmax"] >= heat).sum()),
                "n_season_days": int(len(g)),
            }),
            include_groups=False,
        ).reset_index()
        out.insert(0, "county_ansi", self.county_ansi)
        return out[list(self.SCHEMA)]

    @classmethod
    def load_counties(
        cls, centroids: pd.DataFrame, config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Pull Daymet features for every county in a centroids frame and stack them.

        `centroids` must have columns county_ansi, lat, lon (from
        SsurgoSoilLoader.county_centroids).
        """
        if config is None:
            config = load_source_config()
        use_store = config.get("http", {}).get("use_system_trust_store", True)
        session = build_session(use_system_trust_store=use_store)

        frames: list[pd.DataFrame] = []
        for _, row in centroids.iterrows():
            loader = cls(county_ansi=row["county_ansi"], lat=float(row["lat"]),
                         lon=float(row["lon"]), config=config, session=session)
            df = loader.load()
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        logger.info("[load_counties] %d county-year weather rows.", len(combined))
        return combined


def _print_coverage_report(df: pd.DataFrame) -> None:
    """Print a human-readable per-crop coverage report for the combined frame.

    This is the 'see where PA data is thin' view: for each crop it shows how many
    county-year rows exist, the year span, how many distinct counties appear, and
    what fraction of yield values NASS suppressed (NaN).
    """
    print("\n=== NASS PA county-year yield coverage ===")
    header = f"{'crop':<20}{'rows':>6}{'years':>14}{'counties':>10}{'%suppressed':>13}"
    print(header)
    print("-" * len(header))
    for name, grp in df.groupby("query_name"):
        years = f"{int(grp['year'].min())}-{int(grp['year'].max())}"
        counties = grp["county_name"].nunique()
        suppressed = grp["yield_value"].isna().mean() * 100
        print(f"{name:<20}{len(grp):>6}{years:>14}{counties:>10}{suppressed:>12.1f}%")
    print("-" * len(header))

    # Spotlight Chester County — the actual target area for this product.
    chester = df[df["county_name"].str.upper() == "CHESTER"]
    print(f"\nChester County rows: {len(chester)}")
    if not chester.empty:
        for name, grp in chester.groupby("query_name"):
            yrs = f"{int(grp['year'].min())}-{int(grp['year'].max())}"
            print(f"  {name:<20} {len(grp):>4} rows  ({yrs})")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # All 8 crops we configured. Adding/removing a crop here means editing this
    # list + sources.yaml — no other code changes (the loader is config-driven).
    CROPS = [
        "corn_grain_yield",
        "corn_silage_yield",
        "soybean_yield",
        "winter_wheat_yield",
        "barley_yield",
        "oats_yield",
        "hay_yield",
        "alfalfa_hay_yield",
    ]

    cfg = load_source_config()
    print("Planned request URL for corn (key hidden):")
    print(NASSYieldLoader("corn_grain_yield", config=cfg).build_query_url(include_key=False))

    combined = NASSYieldLoader.load_crops(CROPS, config=cfg)
    _print_coverage_report(combined)

    # Persist the tidy long-format table for downstream modules / inspection.
    out_path = _REPO_ROOT / "data" / "processed" / "nass_pa_yields.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nWrote {len(combined)} rows -> {out_path}")
    print("\nSample (Chester County corn grain, most recent years):")
    sample = combined[
        (combined["query_name"] == "corn_grain_yield")
        & (combined["county_name"].str.upper() == "CHESTER")
    ].tail(5)
    print(sample.to_string(index=False))
