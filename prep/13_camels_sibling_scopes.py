"""
Resolve the CAMELS-sibling scope_ids (caravan_v1, caravan_us_only,
caravan_attributes, camels_ch, camels_de) once the raw downloads complete.

Produces parquets under scopes/CAMELS/<scope_id>/ consistent with the shape
expected by the stager (attributes.parquet + optional streamflow / forcing
parquets).

Non-fatal if inputs aren't fully there yet — the script logs what's missing
and continues with the rest.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIB = ROOT / "raw_data" / "CAMELS_siblings"
SCOPES = ROOT / "scopes" / "CAMELS"


def resolve_caravan_v1() -> bool:
    caravan_dir = SIB / "caravan_v1"
    if not caravan_dir.exists():
        print(f"  caravan_v1 dir missing — skipping"); return False
    # Find attribute files
    attr_files = list((caravan_dir).rglob("attributes_caravan_*.csv"))
    if not attr_files:
        print(f"  no attribute files found in {caravan_dir}"); return False

    # Merge all attributes + HydroATLAS into a global frame
    dfs = []
    for p in attr_files:
        source = p.stem.replace("attributes_caravan_", "")
        df = pd.read_csv(p)
        df["source_dataset"] = source
        dfs.append(df)
    cara_attrs = pd.concat(dfs, ignore_index=True)

    # HydroATLAS attrs
    ha_files = list(caravan_dir.rglob("attributes_hydroatlas_*.csv"))
    if ha_files:
        ha_dfs = []
        for p in ha_files:
            source = p.stem.replace("attributes_hydroatlas_", "")
            df = pd.read_csv(p)
            df["source_dataset"] = source
            ha_dfs.append(df)
        ha = pd.concat(ha_dfs, ignore_index=True)
        if "gauge_id" in ha.columns:
            cara_attrs = cara_attrs.merge(ha, on=["gauge_id", "source_dataset"],
                                           how="left", suffixes=("", "_atlas"))

    # caravan_v1 scope
    out1 = SCOPES / "caravan_v1"
    out1.mkdir(parents=True, exist_ok=True)
    cara_attrs.to_parquet(out1 / "attributes.parquet", index=False)
    (out1 / "basin_ids.txt").write_text("\n".join(cara_attrs["gauge_id"].astype(str)) + "\n")
    (out1 / "summary.json").write_text(
        '{"scope_id": "caravan_v1", "n_basins": %d}\n' % len(cara_attrs))

    # caravan_us_only — filter source_dataset == 'camels'
    out2 = SCOPES / "caravan_us_only"
    out2.mkdir(parents=True, exist_ok=True)
    us = cara_attrs[cara_attrs["source_dataset"].str.lower() == "camels"]
    us.to_parquet(out2 / "attributes.parquet", index=False)
    (out2 / "basin_ids.txt").write_text("\n".join(us["gauge_id"].astype(str)) + "\n")
    (out2 / "summary.json").write_text(
        '{"scope_id": "caravan_us_only", "n_basins": %d}\n' % len(us))

    # Try to also copy a sample of timeseries per scope (first 50 basins)
    ts_root = caravan_dir / "timeseries" / "csv"
    if not ts_root.exists():
        ts_root = None
        for cand in caravan_dir.rglob("timeseries/csv"):
            ts_root = cand
            break
    if ts_root is not None:
        global_sample = []
        us_sample = []
        n_global = n_us = 0
        for src_dir in ts_root.iterdir():
            if not src_dir.is_dir():
                continue
            for ts_file in src_dir.glob("*.csv"):
                if n_global >= 200:
                    break
                df_ts = pd.read_csv(ts_file, nrows=3650)  # first 10 years
                df_ts["gauge_id"] = ts_file.stem
                df_ts["source_dataset"] = src_dir.name
                global_sample.append(df_ts)
                n_global += 1
                if src_dir.name.lower() == "camels" and n_us < 100:
                    us_sample.append(df_ts)
                    n_us += 1
        if global_sample:
            pd.concat(global_sample, ignore_index=True).to_parquet(
                out1 / "timeseries_sample.parquet", index=False)
        if us_sample:
            pd.concat(us_sample, ignore_index=True).to_parquet(
                out2 / "timeseries_sample.parquet", index=False)

    print(f"  caravan_v1: {len(cara_attrs)} basins, caravan_us_only: {len(us)} basins")
    return True


def resolve_camels_ch() -> bool:
    ch_dir = SIB / "camels_ch"
    if not ch_dir.exists():
        print(f"  camels_ch dir missing — skipping"); return False
    # Find a basic attribute file
    attr_candidates = list(ch_dir.rglob("*attributes*.csv"))
    if not attr_candidates:
        print(f"  no attribute CSV in {ch_dir}"); return False
    df_parts = []
    for p in attr_candidates:
        try:
            df_parts.append(pd.read_csv(p))
        except Exception:
            continue
    if not df_parts:
        return False
    df = df_parts[0]  # use the largest / first
    if "gauge_id" not in df.columns:
        # try to infer
        for c in df.columns:
            if c.lower() in ("id", "basin_id", "gauge_id", "code"):
                df = df.rename(columns={c: "gauge_id"})
                break
    out = SCOPES / "camels_ch"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "attributes.parquet", index=False)
    (out / "summary.json").write_text(
        '{"scope_id": "camels_ch", "n_basins": %d, "source": "CAMELS-CH Zenodo 15025258"}\n' % len(df))
    print(f"  camels_ch: {len(df)} basins")
    return True


def main() -> int:
    print("Resolving CAMELS siblings...")
    any_written = False
    any_written |= resolve_caravan_v1()
    any_written |= resolve_camels_ch()
    print(f"\nDone. Any new scope written: {any_written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
