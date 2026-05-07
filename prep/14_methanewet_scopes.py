"""
Write MethaneWet scope-directory stubs so the stager will accept queries
for these scopes. Each stub directory contains a small summary and a
basin_ids.txt (actually site_ids.txt) listing which FLUXNET-CH4 sites
belong to the scope according to our synthesized site_metadata.csv.

The actual task data (flux timeseries, drivers, etc.) lives in
raw_data/MethaneWet/ as a global dataset. Per-scope filtering is up to the
agent to apply if it needs the class/region/records subset.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_data" / "MethaneWet"
SCOPES = ROOT / "scopes" / "MethaneWet"
SCOPES.mkdir(parents=True, exist_ok=True)
SCOPE_LIB = ROOT / "data_tasks" / "mas_bench_scope_library.json"


def resolve(meta: pd.DataFrame, sid: str) -> pd.DataFrame | None:
    """Return subset of sites belonging to the scope, or None if undefined."""
    cls_col = "SITE_CLASSIFICATION"

    # Canonical FLUXNET-CH4 subsets
    if sid == "fluxnet_ch4_full":
        return meta[meta["source_type"] == "ec"]
    if sid == "fluxnet_ch4_freshwater":
        return meta[meta[cls_col].isin(["bog", "fen", "marsh", "swamp", "wet_tundra"])]
    if sid == "fluxnet_ch4_rice":
        return meta[meta[cls_col] == "rice"]
    if sid == "fluxnet_ch4_brackish":
        return meta[meta[cls_col] == "brackish"]
    if sid == "fluxnet_ch4_uplands":
        return meta[meta[cls_col] == "upland"]
    if sid == "fluxnet_ch4_drained":
        return meta[meta[cls_col] == "drained"]

    # UpCH4 / Peltola / Irvin / Chen synthesis subsets
    if sid == "upch4_43":
        # 43 freshwater EC sites
        fw = meta[(meta["source_type"] == "ec") &
                  (meta[cls_col].isin(["bog", "fen", "marsh", "swamp", "wet_tundra"]))]
        return fw.head(43)
    if sid == "upch4_bogs":
        return meta[(meta["source_type"] == "ec") & (meta[cls_col] == "bog")]
    if sid == "upch4_fens":
        return meta[(meta["source_type"] == "ec") & (meta[cls_col] == "fen")]
    if sid == "upch4_marshes":
        return meta[(meta["source_type"] == "ec") & (meta[cls_col] == "marsh")]
    if sid == "upch4_swamps":
        return meta[(meta["source_type"] == "ec") & (meta[cls_col] == "swamp")]
    if sid == "upch4_wet_tundra":
        return meta[(meta["source_type"] == "ec") & (meta[cls_col] == "wet_tundra")]
    if sid == "peltola_2019":
        # >45°N freshwater EC sites
        return meta[(meta["source_type"] == "ec") &
                    (meta[cls_col].isin(["bog", "fen", "wet_tundra", "marsh"])) &
                    (meta["lat"] > 45)]
    if sid == "irvin_2021":
        # ~17 freshwater + 2 rice with ≥1 year
        fw = meta[(meta["source_type"] == "ec") &
                  (meta[cls_col].isin(["bog", "fen", "marsh", "swamp", "wet_tundra"]))]
        rice = meta[(meta["source_type"] == "ec") & (meta[cls_col] == "rice")]
        return pd.concat([fw.head(15), rice.head(2)])
    if sid == "knox_2019_bams":
        return meta[meta["source_type"] == "ec"].head(60)
    if sid == "chen_2024_82":
        return meta  # full combined panel

    # Climate zones
    if sid == "arctic_boreal":
        return meta[meta["climate_zone"].isin(["arctic", "boreal"])]
    if sid == "temperate_wetlands":
        return meta[(meta["climate_zone"] == "temperate") &
                    (meta[cls_col].isin(["bog", "fen", "marsh", "swamp"]))]
    if sid == "tropical_subtropical":
        return meta[meta["climate_zone"] == "tropical"]
    if sid == "arctic_above_665":
        return meta[meta["lat"] >= 66.5]

    # Permafrost
    if sid == "permafrost_continuous":
        return meta[(meta["lat"] >= 68) & (meta["climate_zone"].isin(["arctic", "boreal"]))]
    if sid == "permafrost_discontinuous":
        return meta[(meta["lat"].between(60, 68)) & (meta["climate_zone"] == "boreal")]
    if sid == "permafrost_none":
        return meta[meta["lat"] < 60]

    # Record length (we synthesized ANN_YEARS)
    if sid == "record_long_5yr":
        return meta[meta["ANN_YEARS"] >= 5]
    if sid == "record_medium_3_5yr":
        return meta[(meta["ANN_YEARS"] >= 3) & (meta["ANN_YEARS"] < 5)]
    if sid == "record_short_under3":
        return meta[meta["ANN_YEARS"] < 3]

    # Biome clusters
    if sid == "boreal_peatlands_3yr":
        return meta[(meta["climate_zone"] == "boreal") &
                    (meta[cls_col].isin(["bog", "fen"])) &
                    (meta["ANN_YEARS"] >= 3)]
    if sid == "temperate_marshes":
        return meta[(meta["climate_zone"] == "temperate") & (meta[cls_col] == "marsh")]
    if sid == "tropical_additions":
        return meta[meta["climate_zone"] == "tropical"].tail(5)

    # Region (by site_id prefix)
    if sid == "north_america_ch4":
        return meta[meta["SITE_ID"].str.startswith(("US-", "CA-", "BW-"))]
    if sid == "europe_ch4":
        return meta[meta["SITE_ID"].str.startswith(
            ("FI-", "SE-", "DE-", "NL-", "UK-", "IT-", "CZ-", "DK-", "CH-", "IL-", "FR-"))]
    if sid == "asia_oceania_ch4":
        return meta[meta["SITE_ID"].str.startswith(("JP-", "RU-", "CN-", "MY-", "AU-", "PH-"))]

    # BAWLD chamber subsets
    if sid == "bawld_ch4_full":
        return meta[meta["source_type"] == "chamber"]
    if sid == "bawld_bogs":
        return meta[(meta["source_type"] == "chamber") & (meta[cls_col] == "bog")]
    if sid == "bawld_permafrost_bogs":
        return meta[(meta["source_type"] == "chamber") & (meta[cls_col] == "permafrost_bog")]
    if sid == "bawld_tundra_wetlands":
        return meta[(meta["source_type"] == "chamber") & (meta[cls_col] == "wet_tundra")]

    return None


def main() -> int:
    meta_path = RAW / "site_metadata.csv"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found — run prep/09_methanewet_synthesis.py first")
        return 1
    meta = pd.read_csv(meta_path)
    lib = json.loads(SCOPE_LIB.read_text())["MethaneWet"]
    print(f"MethaneWet scopes in library: {len(lib)}")

    written = 0
    unresolved = 0
    for sid, meta_entry in lib.items():
        sub = resolve(meta, sid)
        d = SCOPES / sid
        d.mkdir(parents=True, exist_ok=True)
        if sub is None:
            # Fall back to full set so queries still stage
            sub = meta
            (d / "STATUS.txt").write_text(
                f"scope_id={sid}\nstatus=fallback_to_full\nlibrary_description={meta_entry.get('description','')}\n")
            unresolved += 1
        (d / "site_ids.txt").write_text("\n".join(sub["SITE_ID"].astype(str)) + "\n")
        sub.to_parquet(d / "site_metadata.parquet", index=False)
        (d / "summary.json").write_text(json.dumps({
            "scope_id": sid,
            "n_sites": int(len(sub)),
        }, indent=2) + "\n")
        written += 1
    print(f"Wrote {written}/{len(lib)} scopes ({unresolved} fell back to full set)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
