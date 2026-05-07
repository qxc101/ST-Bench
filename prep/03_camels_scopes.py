"""
Resolve each CAMELS scope_id to (a) a basin ID list, and (b) a parquet of
scoped timeseries + attributes, written to scopes/CAMELS/<scope_id>/.

Scope membership rules are documented in prep/inventory.py CAMELS_SCOPES and
anchored on the Addor 2017 attribute files (camels_clim.txt, camels_hydro.txt,
camels_topo.txt, camels_vege.txt) and the Kratzert 531-basin list.

Output per scope (scopes/CAMELS/<scope_id>/):
    basin_ids.txt           newline-delimited gauge IDs in this scope
    attributes.parquet      per-basin attributes (merged from all attr tables),
                            filtered to basin_ids
    streamflow_daily.parquet  long-format (gauge_id, date, q_cms, q_mm_day) --
                              only written when the timeseries zip is already
                              extracted; otherwise deferred with a WARN
    forcing_daymet.parquet    long-format Daymet forcing per basin per day
    README.md                 summary of scope + row counts + file sizes

Run:
    conda activate stbench
    python prep/03_camels_scopes.py [--scope <scope_id> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw_data" / "CAMELS"
SCOPES_DIR = ROOT / "scopes" / "CAMELS"
SCOPE_LIB = ROOT / "data_tasks" / "mas_bench_scope_library.json"

BASIN_LIST_531_URL = (
    "https://raw.githubusercontent.com/kratzert/"
    "ealstm_regional_modeling/master/data/basin_list.txt"
)


# ----------------------------------------------------------------------------
# Attribute loading
# ----------------------------------------------------------------------------

def load_attributes() -> pd.DataFrame:
    """Load and merge Addor 2017 attribute tables on gauge_id (zero-padded 8-digit string)."""
    def read_attr(name: str) -> pd.DataFrame:
        df = pd.read_csv(RAW_DIR / name, sep=";", dtype={"gauge_id": str})
        df["gauge_id"] = df["gauge_id"].str.zfill(8)
        return df

    clim = read_attr("camels_clim.txt")
    hydro = read_attr("camels_hydro.txt")
    topo = read_attr("camels_topo.txt")
    soil = read_attr("camels_soil.txt")
    vege = read_attr("camels_vege.txt")
    geol = read_attr("camels_geol.txt")
    name = read_attr("camels_name.txt")  # has huc_02, gauge_name

    df = (
        clim.merge(hydro, on="gauge_id", how="outer")
            .merge(topo,  on="gauge_id", how="outer")
            .merge(soil,  on="gauge_id", how="outer")
            .merge(vege,  on="gauge_id", how="outer")
            .merge(geol,  on="gauge_id", how="outer")
            .merge(name,  on="gauge_id", how="outer")
    )
    # Keep huc_02 as zero-padded string (e.g. "01") and also an int column for filters
    df["huc_02"] = df["huc_02"].astype(str).str.zfill(2)
    df["huc_02_int"] = df["huc_02"].astype(int)
    return df


def load_basin_list_531() -> list[str]:
    """Fetch the Kratzert 531-basin list; cache at raw_data/CAMELS/basin_list_531.txt."""
    cache = RAW_DIR / "basin_list_531.txt"
    if not cache.exists():
        print(f"  fetching 531-basin list -> {cache}")
        cache.write_bytes(urllib.request.urlopen(BASIN_LIST_531_URL, timeout=30).read())
    ids = cache.read_text().strip().splitlines()
    return [i.zfill(8) for i in ids]


# ----------------------------------------------------------------------------
# Scope resolution rules — keyed by scope_id in mas_bench_scope_library.json
# Each rule produces a list[str] of gauge_ids (for CAMELS-US-derived scopes)
# or is marked `external` (sibling CAMELS datasets downloaded separately).
# ----------------------------------------------------------------------------

# Rules operate on a merged attributes DataFrame + basin_list_531
def resolve_scope(scope_id: str, attrs: pd.DataFrame, list_531: list[str]) -> dict[str, Any]:
    df531 = attrs[attrs["gauge_id"].isin(list_531)]

    # Exact base-list scopes
    if scope_id == "camels_us_531":
        return {"kind": "camels_us", "ids": list_531}
    if scope_id == "camels_us_671":
        return {"kind": "camels_us", "ids": sorted(attrs["gauge_id"].dropna().unique().tolist())}
    if scope_id == "camels_us_516_chem":
        # Approximation: the 516-basin CAMELS-US ∩ CAMELS-Chem intersection is the
        # 531 list minus the 15 sites with poor-chem coverage listed in Sterle 2024
        # Table S2. Without that list cached here, fall back to the 531 list with a
        # TODO marker — only 1 query uses this scope.
        return {"kind": "camels_us", "ids": list_531, "note": "approximated as 531 pending Sterle 2024 site list"}

    # External siblings — handled by separate downloaders
    if scope_id in ("camels_gb", "camels_cl", "camels_br", "camels_aus", "camels_ch",
                    "camels_de", "caravan_v1", "caravan_us_only", "hysets_subset", "lamah_ce"):
        return {"kind": "external_sibling", "sibling": scope_id}

    # Filter rules on attributes (531-anchored except camels_us_671)
    rules = {
        # clim
        "snow_dominated":        lambda d: d[d["frac_snow"] > 0.3],
        "snow_dominated_strict": lambda d: d[d["frac_snow"] > 0.5],
        "arid":                  lambda d: d[d["aridity"] > 1],
        "very_arid":             lambda d: d[d["aridity"] > 2],
        "humid":                 lambda d: d[d["aridity"] < 0.75],
        "humid_no_snow":         lambda d: d[(d["aridity"] < 1) & (d["frac_snow"] < 0.15)],
        "seasonal_summer_p":     lambda d: d[d["p_seasonality"] > 0],
        "seasonal_winter_p":     lambda d: d[d["p_seasonality"] < 0],
        # hydro
        "high_baseflow":     lambda d: d[d["baseflow_index"] > 0.5],
        "low_baseflow":      lambda d: d[d["baseflow_index"] < 0.3],
        "intermittent":      lambda d: d[d["zero_q_freq"] > 0],
        "high_runoff_ratio": lambda d: d[d["runoff_ratio"] > 0.5],
        "low_runoff_ratio":  lambda d: d[d["runoff_ratio"] < 0.3],
        # topo
        "small_basins":   lambda d: d[d["area_gages2"] < 250],
        "medium_basins":  lambda d: d[(d["area_gages2"] >= 250) & (d["area_gages2"] < 1000)],
        "large_basins":   lambda d: d[(d["area_gages2"] >= 1000) & (d["area_gages2"] < 2000)],
        "high_elevation": lambda d: d[d["elev_mean"] > 1500],
        "low_elevation":  lambda d: d[d["elev_mean"] < 500],
        "steep_slopes":   lambda d: d[d["slope_mean"] > 0.1],
        # vege
        "forested":       lambda d: d[d["frac_forest"] > 0.5],
        # HUC-2 regions (huc_02 is a zero-padded string; use huc_02_int for comparisons)
        "huc2_pacific_nw":     lambda d: d[d["huc_02_int"] == 17],
        "huc2_california":     lambda d: d[d["huc_02_int"] == 18],
        "huc2_upper_colorado": lambda d: d[d["huc_02_int"] == 14],
        "huc2_midwest":        lambda d: d[d["huc_02_int"].isin([5, 6, 7, 8, 10, 11])],
        "huc2_northeast":      lambda d: d[d["huc_02_int"].isin([1, 2, 3, 4])],
        "huc2_southeast":      lambda d: d[d["huc_02_int"].isin([3, 6, 8])],
    }
    if scope_id in rules:
        sub = rules[scope_id](df531)
        return {"kind": "camels_us", "ids": sorted(sub["gauge_id"].tolist())}

    # Temporal / fold-based scopes — keep the full 531 but tag the split semantics
    if scope_id == "kratzert_temporal":
        return {
            "kind": "camels_us_temporal",
            "ids": list_531,
            "temporal": {"train": ["1999-10-01", "2008-09-30"],
                         "test":  ["1989-10-01", "1999-09-30"]},
        }
    if scope_id == "newman_temporal":
        return {
            "kind": "camels_us_temporal",
            "ids": list_531,
            "temporal": {"train": ["1980-10-01", "1995-09-30"],
                         "test":  ["1995-10-01", "2014-09-30"]},
        }
    if scope_id == "pub_12fold":
        return {"kind": "camels_us_cv", "ids": list_531, "cv": "12-fold spatial (Kratzert 2019b)"}
    if scope_id == "huc2_loo":
        sub = df531.copy()
        folds = {h: sorted(sub.loc[sub["huc_02_int"] == h, "gauge_id"].tolist())
                 for h in sorted(sub["huc_02_int"].unique()) if len(sub[sub["huc_02_int"] == h]) > 0}
        return {"kind": "camels_us_cv", "ids": list_531, "cv": "HUC-2 leave-one-out", "folds": folds}

    return {"kind": "unresolved", "reason": f"no rule for scope_id {scope_id!r}"}


# ----------------------------------------------------------------------------
# Timeseries loader — reads the basin_dataset_public_v1p2 tree once it exists
# ----------------------------------------------------------------------------

TIMESERIES_ROOT = RAW_DIR / "basin_dataset_public_v1p2"


def load_streamflow(ids: list[str]) -> pd.DataFrame | None:
    """Load daily streamflow for given gauge_ids from usgs_streamflow/<huc>/<id>_streamflow_qc.txt.
    Returns long-format DataFrame (gauge_id, date, q_cfs, qc_flag). Returns None
    if the source tree isn't extracted yet.
    """
    usgs_root = TIMESERIES_ROOT / "usgs_streamflow"
    if not usgs_root.exists():
        return None

    # Build index of all available files (path per gauge) once per call
    files = {p.stem.split("_")[0].zfill(8): p
             for p in usgs_root.rglob("*_streamflow_qc.txt")}
    rows = []
    missing: list[str] = []
    for gid in ids:
        fp = files.get(gid)
        if not fp:
            missing.append(gid)
            continue
        # Format: YYYY MM DD Q QC_FLAG (whitespace-delimited), Q in cfs
        df = pd.read_csv(fp, sep=r"\s+", header=None,
                         names=["gauge_id_raw", "year", "month", "day", "q_cfs", "qc_flag"],
                         engine="python")
        df["gauge_id"] = gid
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        rows.append(df[["gauge_id", "date", "q_cfs", "qc_flag"]])
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    if missing:
        print(f"    !! streamflow missing for {len(missing)}/{len(ids)} basins (first: {missing[:3]})")
    return out


def load_forcing_daymet(ids: list[str]) -> pd.DataFrame | None:
    """Load Daymet daily forcing for given gauge_ids from
    basin_mean_forcing/daymet/<huc>/<id>_lump_cida_forcing_leap.txt.
    Returns long-format DataFrame (gauge_id, date, ...forcing vars...).
    """
    fr_root = TIMESERIES_ROOT / "basin_mean_forcing" / "daymet"
    if not fr_root.exists():
        return None

    files = {p.stem.split("_")[0].zfill(8): p
             for p in fr_root.rglob("*_lump_cida_forcing_leap.txt")}
    rows = []
    missing: list[str] = []
    for gid in ids:
        fp = files.get(gid)
        if not fp:
            missing.append(gid)
            continue
        # Skip 3-line header; columns: Year Mnth Day Hr Dayl(s) PRCP(mm/day) SRAD(W/m2) SWE(mm) Tmax(C) Tmin(C) Vp(Pa)
        df = pd.read_csv(fp, sep=r"\s+", skiprows=4, header=None,
                         names=["year", "month", "day", "hour", "dayl_s",
                                "prcp_mm", "srad_wm2", "swe_mm", "tmax_c", "tmin_c", "vp_pa"],
                         engine="python")
        df["gauge_id"] = gid
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        rows.append(df[["gauge_id", "date", "prcp_mm", "srad_wm2", "swe_mm",
                        "tmax_c", "tmin_c", "vp_pa", "dayl_s"]])
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    if missing:
        print(f"    !! daymet missing for {len(missing)}/{len(ids)} basins (first: {missing[:3]})")
    return out


# ----------------------------------------------------------------------------
# Per-scope writer
# ----------------------------------------------------------------------------

def write_scope(scope_id: str, scope_info: dict, attrs: pd.DataFrame) -> dict:
    out_dir = SCOPES_DIR / scope_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Unresolved / external: write a placeholder note and return
    if scope_info["kind"] in ("unresolved", "external_sibling"):
        (out_dir / "STATUS.txt").write_text(
            f"scope_id={scope_id}\nstatus={scope_info['kind']}\ndetails={json.dumps(scope_info)}\n"
        )
        return {"scope_id": scope_id, "status": scope_info["kind"], "n_ids": 0}

    ids = scope_info["ids"]

    # Write basin id list
    (out_dir / "basin_ids.txt").write_text("\n".join(ids) + "\n")

    # Write attributes parquet
    sub_attrs = attrs[attrs["gauge_id"].isin(ids)].copy()
    attrs_parquet = out_dir / "attributes.parquet"
    sub_attrs.to_parquet(attrs_parquet, index=False)

    result = {
        "scope_id": scope_id,
        "status": "ok",
        "n_ids": len(ids),
        "attributes_rows": len(sub_attrs),
    }

    # Try to write streamflow + forcing if raw timeseries tree is available
    q = load_streamflow(ids)
    if q is not None:
        q.to_parquet(out_dir / "streamflow_daily.parquet", index=False)
        result["streamflow_rows"] = len(q)

    f = load_forcing_daymet(ids)
    if f is not None:
        f.to_parquet(out_dir / "forcing_daymet.parquet", index=False)
        result["forcing_rows"] = len(f)

    # Add temporal/cv metadata if present
    if "temporal" in scope_info:
        result["temporal"] = scope_info["temporal"]
    if "cv" in scope_info:
        result["cv"] = scope_info["cv"]

    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", action="append", help="Restrict to specific scope_id(s)")
    ap.add_argument("--list", action="store_true", help="List CAMELS scopes and exit")
    args = ap.parse_args()

    scopes = json.loads(SCOPE_LIB.read_text())["CAMELS"]
    if args.list:
        for sid, meta in scopes.items():
            print(f"  {sid:24s} size={meta.get('size','?'):<6}  {meta.get('name','')}")
        return 0

    target_ids = args.scope if args.scope else list(scopes.keys())

    print("Loading CAMELS attributes + 531-basin list...")
    attrs = load_attributes()
    list_531 = load_basin_list_531()
    print(f"  attributes rows: {len(attrs)}  (expect ~671)")
    print(f"  531-basin list:  {len(list_531)}  (expect 531)")

    print(f"\nResolving {len(target_ids)} scopes...")
    summary = []
    for sid in target_ids:
        print(f"\n--- {sid} ({scopes[sid].get('name','')}) ---")
        info = resolve_scope(sid, attrs, list_531)
        res = write_scope(sid, info, attrs)
        summary.append(res)
        print(f"  -> {res.get('status')}: {res.get('n_ids')} basins"
              + (f", streamflow_rows={res['streamflow_rows']:,}" if "streamflow_rows" in res else "")
              + (f", forcing_rows={res['forcing_rows']:,}"      if "forcing_rows" in res    else ""))

    # Overall summary
    overall = {
        "total_scopes": len(summary),
        "ok":        [s["scope_id"] for s in summary if s["status"] == "ok"],
        "external":  [s["scope_id"] for s in summary if s["status"] == "external_sibling"],
        "unresolved":[s["scope_id"] for s in summary if s["status"] == "unresolved"],
    }
    (SCOPES_DIR / "_summary.json").write_text(json.dumps(overall, indent=2) + "\n")
    print(f"\nWrote {len(summary)} scopes under {SCOPES_DIR}")
    print(f"  ok:         {len(overall['ok'])}")
    print(f"  external:   {len(overall['external'])}  (sibling datasets, pending)")
    print(f"  unresolved: {len(overall['unresolved'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
