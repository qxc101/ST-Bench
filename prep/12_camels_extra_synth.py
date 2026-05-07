"""
Additional CAMELS synthesis for files not covered by prep/05_camels_synthesis.py
that require either blocked downloads (NID) or large public datasets (NLCD,
GAGES-II). These are small enough to synthesize reasonably from basin metadata.

Produces under raw_data/CAMELS/synth/:
  - nid.csv                   dams per basin (count, type, reservoir_size)
  - nlcd_land_use_change.csv  per-basin delta in major land-cover classes 2001-2019
  - gagesii_timeseries.csv    per-basin annual streamflow stats + static attrs
  - camels_streamflow_extended_1980_2020.csv   (same as 1980-2014 with trend extrapolation)
  - camels_extended_1980_2020.csv alias (wide format)
  - era5_reforecast.csv       synthetic ERA5-like forcing at basin centroids 2018-2020
  - era5_land_forcing.csv     alias, daily
  - caravan_attributes.csv    synthesized HydroATLAS-style attributes for CAMELS-US
  - caravan_global.csv        synthetic global basin attribute file (500 basins)
  - camels_us.csv             camels_us_531 panel wide CSV
  - camels_de.csv             synthetic CAMELS-DE-like subset (100 basins)
  - camels_us_harmonized.csv  harmonized column names for merge-with-siblings tasks
  - camels_gb_harmonized.csv  synthetic GB-like 100-basin panel in harmonized cols
  - camels_cl_harmonized.csv  synthetic CL-like 100-basin panel
  - camels_br_harmonized.csv  synthetic BR-like 100-basin panel
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SYNTH = ROOT / "raw_data" / "CAMELS" / "synth"
SYNTH.mkdir(parents=True, exist_ok=True)
BASE = ROOT / "scopes" / "CAMELS" / "camels_us_531"


def load_base():
    attrs = pd.read_parquet(BASE / "attributes.parquet")
    q = pd.read_parquet(BASE / "streamflow_daily.parquet")
    f = pd.read_parquet(BASE / "forcing_daymet.parquet")
    return attrs, q, f


def synth_nid(attrs: pd.DataFrame) -> None:
    rng = np.random.default_rng(42)
    rows = []
    for _, b in attrs.iterrows():
        area = float(b.get("area_gages2", 200) or 200)
        expected_dams = max(0, int(rng.poisson(area / 300)))  # more dams in bigger basins
        for i in range(expected_dams):
            rows.append({
                "basin_gauge_id": b["gauge_id"],
                "dam_id": f"{b['gauge_id']}_D{i:02d}",
                "dam_name": f"synth-dam-{b['gauge_id']}-{i}",
                "lat": b.get("gauge_lat", np.nan) + rng.normal(0, 0.1),
                "lon": b.get("gauge_lon", np.nan) + rng.normal(0, 0.1),
                "purpose": rng.choice(["flood_control", "irrigation", "hydro", "recreation", "water_supply"]),
                "height_m": float(rng.uniform(5, 80)),
                "reservoir_capacity_mcm": float(rng.uniform(0.5, 500)),
                "year_completed": int(rng.integers(1900, 2010)),
            })
    nid = pd.DataFrame(rows)
    nid.to_csv(SYNTH / "nid.csv", index=False)
    print(f"  nid.csv: {len(nid)} synthetic dams")


def synth_nlcd(attrs: pd.DataFrame) -> None:
    rng = np.random.default_rng(7)
    rows = []
    for _, b in attrs.iterrows():
        start_forest = float(b.get("frac_forest", 0.3) or 0.3)
        # Simulate a small delta over 18 years
        delta_forest = float(rng.normal(-0.02, 0.04))
        end_forest = float(np.clip(start_forest + delta_forest, 0.0, 1.0))
        start_urban = float(rng.uniform(0.0, 0.2))
        end_urban = float(np.clip(start_urban + rng.normal(0.01, 0.02), 0.0, 1.0))
        start_crop = float(rng.uniform(0.0, 0.5))
        end_crop = float(np.clip(start_crop + rng.normal(-0.005, 0.02), 0.0, 1.0))
        rows.append({
            "gauge_id": b["gauge_id"],
            "forest_2001": start_forest, "forest_2019": end_forest,
            "urban_2001": start_urban, "urban_2019": end_urban,
            "cropland_2001": start_crop, "cropland_2019": end_crop,
            "delta_forest": end_forest - start_forest,
            "delta_urban":  end_urban  - start_urban,
            "delta_crop":   end_crop   - start_crop,
        })
    pd.DataFrame(rows).to_csv(SYNTH / "nlcd_land_use_change.csv", index=False)
    print(f"  nlcd_land_use_change.csv: {len(rows)} basins")


def synth_gagesii(attrs: pd.DataFrame, q: pd.DataFrame) -> None:
    # Annual mean + peaks + min per basin-year
    q_ann = q.copy()
    q_ann["year"] = q_ann["date"].dt.year
    stats = (q_ann.groupby(["gauge_id", "year"])
             .agg(q_mean=("q_cfs", "mean"), q_max=("q_cfs", "max"),
                  q_min=("q_cfs", "min"), q_std=("q_cfs", "std"))
             .reset_index())
    # Merge with basic static attrs
    merged = stats.merge(
        attrs[["gauge_id", "gauge_lat", "gauge_lon", "elev_mean", "area_gages2", "huc_02"]],
        on="gauge_id", how="left")
    merged.to_csv(SYNTH / "gagesii_timeseries.csv", index=False)
    print(f"  gagesii_timeseries.csv: {len(merged):,} rows")


def synth_extended(attrs: pd.DataFrame, q: pd.DataFrame) -> None:
    # CAMELS covers 1980-2014. Extend to 2020 via seasonal-cycle + trend extrapolation
    rng = np.random.default_rng(99)
    out_rows = []
    q14 = q.copy()
    q14["year"] = q14["date"].dt.year
    # Fit seasonal mean per basin from 1980-2014
    seasonal = q14.assign(doy=q14["date"].dt.dayofyear).groupby(["gauge_id", "doy"])["q_cfs"].mean()
    new_dates = pd.date_range("2015-01-01", "2020-12-31", freq="D")
    for gid, grp in q14.groupby("gauge_id"):
        # Trend over years
        years = grp["year"].unique()
        annual_means = grp.groupby("year")["q_cfs"].mean()
        if len(annual_means) > 3:
            slope, intercept = np.polyfit(annual_means.index.values.astype(float),
                                           annual_means.values, 1)
        else:
            slope, intercept = 0.0, annual_means.mean() if len(annual_means) else 0
        for d in new_dates:
            doy = d.dayofyear
            seas = seasonal.get((gid, doy), np.nan)
            if np.isnan(seas):
                continue
            year = d.year
            # Extrapolated value = seasonal + trend adjustment + noise
            val = seas + slope * (year - annual_means.mean()) + rng.normal(0, seas * 0.15)
            val = max(0, val)
            out_rows.append({"gauge_id": gid, "date": d, "q_cfs": val, "qc_flag": "S"})
    extended_new = pd.DataFrame(out_rows)
    # Concat original through 2014 + synthetic 2015-2020
    full = pd.concat([q14[["gauge_id", "date", "q_cfs", "qc_flag"]], extended_new], ignore_index=True)
    # Write both long and wide forms
    full.to_csv(SYNTH / "camels_streamflow_extended_1980_2020.csv", index=False)
    # Wide form: gauge_id columns, date index — keep smallish (take 1 year per sample)
    wide_sample = full[full["date"].dt.year == 2020].pivot_table(
        index="date", columns="gauge_id", values="q_cfs")
    wide_sample.to_csv(SYNTH / "camels_extended_1980_2020.csv")
    print(f"  camels_streamflow_extended_1980_2020.csv: {len(full):,} rows")


def synth_era5_reforecast(attrs: pd.DataFrame) -> None:
    rng = np.random.default_rng(13)
    dates = pd.date_range("2018-01-01", "2020-12-31", freq="D")
    rows = []
    for _, b in attrs.iterrows():
        lat = b.get("gauge_lat", 40)
        amp = 15 if abs(lat) > 35 else 8
        base = 10 + (30 - abs(lat) * 0.4)
        for i, d in enumerate(dates):
            t = base + amp * np.cos(2 * np.pi * (d.dayofyear - 172) / 365) + rng.normal(0, 2)
            p = max(0, rng.gamma(0.5, 4.0))
            rows.append({"gauge_id": b["gauge_id"], "date": d,
                         "t2m_c": float(t), "prcp_mm": float(p),
                         "ssrd_wm2": float(rng.uniform(50, 300))})
    df = pd.DataFrame(rows)
    df.to_csv(SYNTH / "era5_reforecast.csv", index=False)
    df.to_csv(SYNTH / "era5_land_forcing.csv", index=False)
    print(f"  era5_reforecast.csv + era5_land_forcing.csv: {len(df):,} rows")


def synth_siblings(attrs: pd.DataFrame, q: pd.DataFrame) -> None:
    rng = np.random.default_rng(71)
    # camels_us.csv — wide 531-basin panel summary
    base = attrs.copy()
    base["dataset"] = "CAMELS-US"
    base.to_csv(SYNTH / "camels_us.csv", index=False)

    # Harmonized columns (common schema for xcountry tasks)
    harmonized_cols = ["gauge_id", "dataset", "lat", "lon", "area_km2",
                       "elev_mean", "forest_frac", "aridity", "p_mean", "q_mean"]
    def harmonize(df, ds):
        out = pd.DataFrame({
            "gauge_id": df["gauge_id"],
            "dataset": ds,
            "lat": df.get("gauge_lat", np.nan),
            "lon": df.get("gauge_lon", np.nan),
            "area_km2": df.get("area_gages2", np.nan),
            "elev_mean": df.get("elev_mean", np.nan),
            "forest_frac": df.get("frac_forest", np.nan),
            "aridity": df.get("aridity", np.nan),
            "p_mean": df.get("p_mean", np.nan),
            "q_mean": df.get("q_mean", np.nan),
        })
        return out

    us_h = harmonize(attrs, "CAMELS-US")
    us_h.to_csv(SYNTH / "camels_us_harmonized.csv", index=False)

    # Synthetic CAMELS-GB/CL/BR/DE panels — match the harmonized schema
    for ds, n_basins, (lat_rng, lon_rng), seed in [
        ("CAMELS-GB", 150, ((50, 60), (-6, 2)), 1),
        ("CAMELS-CL", 100, ((-55, -17), (-76, -66)), 2),
        ("CAMELS-BR", 100, ((-33, 5), (-74, -35)), 3),
        ("CAMELS-DE", 120, ((47, 55), (6, 15)), 4),
    ]:
        rng2 = np.random.default_rng(seed)
        out = pd.DataFrame({
            "gauge_id": [f"{ds.split('-')[1]}_{i:03d}" for i in range(n_basins)],
            "dataset": ds,
            "lat": rng2.uniform(*lat_rng, n_basins),
            "lon": rng2.uniform(*lon_rng, n_basins),
            "area_km2": rng2.uniform(50, 3000, n_basins),
            "elev_mean": rng2.uniform(50, 2500, n_basins),
            "forest_frac": rng2.uniform(0, 1, n_basins),
            "aridity": rng2.uniform(0.3, 2.5, n_basins),
            "p_mean": rng2.uniform(1, 10, n_basins),
            "q_mean": rng2.uniform(0.2, 8, n_basins),
        })
        suffix = ds.split("-")[1].lower()
        out.to_csv(SYNTH / f"camels_{suffix}_harmonized.csv", index=False)
        if suffix == "de":
            out.to_csv(SYNTH / "camels_de.csv", index=False)
    print("  sibling harmonized CSVs written")


def synth_caravan(attrs: pd.DataFrame) -> None:
    """Synthetic Caravan attributes + global panel if real Caravan hasn't finished
    downloading yet. These will be overwritten/augmented by the sibling resolver
    when Caravan extract finishes."""
    camels_ca = attrs.copy()
    camels_ca["dataset"] = "CAMELS-US"
    camels_ca["source_dataset"] = "camels"
    # Mock HydroATLAS-style columns
    rng = np.random.default_rng(5)
    for col, scale in [("mean_annual_runoff_mm", 1000), ("mean_annual_precip_mm", 1500),
                       ("mean_annual_temp_c", 25), ("slope_mean_degree", 10),
                       ("forest_frac_hydroatlas", 1.0)]:
        camels_ca[col] = rng.uniform(0, scale, len(camels_ca))
    camels_ca.to_csv(SYNTH / "caravan_attributes.csv", index=False)

    # Global Caravan-like 500-basin panel
    rng2 = np.random.default_rng(77)
    ids = [f"CAR_{i:05d}" for i in range(500)]
    datasets = rng2.choice(["camels", "camelsaus", "camelsbr", "camelscl", "camelsgb", "hysets", "lamah"], 500)
    df = pd.DataFrame({
        "gauge_id": ids,
        "source_dataset": datasets,
        "lat": rng2.uniform(-55, 70, 500),
        "lon": rng2.uniform(-170, 170, 500),
        "area_km2": rng2.uniform(100, 2000, 500),
        "elev_mean": rng2.uniform(50, 3000, 500),
        "aridity": rng2.uniform(0.3, 3.0, 500),
        "p_mean": rng2.uniform(1, 10, 500),
        "q_mean": rng2.uniform(0.2, 10, 500),
    })
    df.to_csv(SYNTH / "caravan_global.csv", index=False)
    print(f"  caravan_attributes.csv + caravan_global.csv written")


def main() -> int:
    attrs, q, f = load_base()
    print(f"attrs: {len(attrs)}, q rows: {len(q):,}")

    synth_nid(attrs)
    synth_nlcd(attrs)
    synth_gagesii(attrs, q)
    synth_extended(attrs, q)
    synth_era5_reforecast(attrs)
    synth_siblings(attrs, q)
    synth_caravan(attrs)

    print(f"\nAll extra synth files under {SYNTH}")
    print(f"Total files now: {len(list(SYNTH.iterdir()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
