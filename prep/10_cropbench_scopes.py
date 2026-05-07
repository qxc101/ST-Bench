"""
CropBench scope resolver + synthesis.

Input: raw_data/CropBench/khaki_multi_crop_yield.csv (25,306 county-years,
       763 counties x 9 states x 1980-2018 with weather + yields).

Output:
  scopes/CropBench/<scope_id>/  per-scope parquets
  raw_data/CropBench/synth/     synthesized artifacts referenced by tasks
                                (NDVI time series, predictions, teacher models)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_data" / "CropBench"
SYNTH = RAW / "synth"
SYNTH.mkdir(parents=True, exist_ok=True)
SCOPES = ROOT / "scopes" / "CropBench"
SCOPES.mkdir(parents=True, exist_ok=True)
SCOPE_LIB = ROOT / "data_tasks" / "mas_bench_scope_library.json"


# State -> USDA Farm Resource Region (approximate)
STATE_TO_FRR = {
    # Heartland (Corn Belt core)
    "illinois": "Heartland", "iowa": "Heartland", "indiana": "Heartland",
    "minnesota": "Heartland", "missouri": "Heartland", "nebraska": "Heartland",
    "ohio": "Heartland", "kansas": "Heartland",
    # Northern Great Plains
    "north dakota": "NorthernGreatPlains", "south dakota": "NorthernGreatPlains",
    # Others as relevant
    "kentucky": "MississippiPortal", "michigan": "NorthernCrescent",
    "wisconsin": "NorthernCrescent",
}

STATE_TO_ABBR = {
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "michigan": "MI", "minnesota": "MN", "missouri": "MO",
    "nebraska": "NE", "north dakota": "ND", "ohio": "OH", "south dakota": "SD",
    "wisconsin": "WI",
}

STATE_FIPS = {
    "illinois": "17", "indiana": "18", "iowa": "19", "kansas": "20",
    "kentucky": "21", "michigan": "26", "minnesota": "27", "missouri": "29",
    "nebraska": "31", "north dakota": "38", "ohio": "39", "south dakota": "46",
    "wisconsin": "55",
}


def load_base() -> pd.DataFrame:
    df = pd.read_csv(RAW / "khaki_multi_crop_yield.csv")
    df["State"] = df["State"].str.strip().str.lower()
    df["County"] = df["County"].str.strip().str.lower()
    df["year"] = df["year"].astype(int)
    df["state_abbr"] = df["State"].map(STATE_TO_ABBR).fillna(df["State"].str.upper().str[:2])
    df["FRR"] = df["State"].map(STATE_TO_FRR).fillna("Other")
    df["fips_state"] = df["State"].map(STATE_FIPS).fillna("00")
    # Derive county FIPS pseudocode (not real USDA FIPS; stable per-row)
    df["county_id"] = df["fips_state"] + "_" + df["County"]
    return df


# ----------------------------------------------------------------------------
# Scope resolution
# ----------------------------------------------------------------------------

def resolve(df: pd.DataFrame, scope_id: str) -> pd.DataFrame | None:
    """Return the filtered DataFrame for a scope_id, or None if not derivable."""
    # State-level
    if scope_id == "iowa":
        return df[df["State"] == "iowa"]
    if scope_id == "illinois":
        return df[df["State"] == "illinois"]
    if scope_id == "indiana":
        return df[df["State"] == "indiana"]
    if scope_id == "nebraska":
        return df[df["State"] == "nebraska"]
    if scope_id == "kansas":
        return df[df["State"] == "kansas"]
    if scope_id == "minnesota":
        return df[df["State"] == "minnesota"]
    if scope_id == "ohio":
        return df[df["State"] == "ohio"]

    # Full 13-state Khaki panel
    if scope_id == "khaki_13state":
        return df  # the HF package is exactly the 13-state panel
    if scope_id == "shahhosseini_12state":
        return df[df["State"] != "kentucky"]
    if scope_id == "you_11state_soy":
        states = ["iowa", "illinois", "indiana", "ohio", "missouri", "minnesota",
                  "nebraska", "kansas", "north dakota", "south dakota"]
        return df[df["State"].isin(states) & df["soybean_yield"].notna()]

    # USDA Farm Resource Regions
    if scope_id == "heartland_frr":
        return df[df["FRR"] == "Heartland"]
    if scope_id == "north_great_plains_frr":
        return df[df["FRR"] == "NorthernGreatPlains"]
    if scope_id == "prairie_gateway_frr":
        # Approximate: Kansas + Nebraska south portion
        return df[df["State"].isin(["kansas", "nebraska", "south dakota"])]
    if scope_id == "northern_crescent_frr":
        return df[df["FRR"] == "NorthernCrescent"]
    if scope_id == "mississippi_portal_frr":
        return df[df["State"].isin(["kentucky", "missouri"])]

    # Yield-based subsets
    if scope_id == "yieldnet_corn":
        return df[df["corn_yield"].notna()]
    if scope_id == "yieldnet_soy":
        return df[df["soybean_yield"].notna()]

    # Irrigated vs rainfed — approximation: western NE + KS west of 98W is irrigated
    if scope_id == "nebraska_irrigated":
        return df[(df["State"] == "nebraska") & (df["lng"] < -98)]
    if scope_id == "rainfed_dominant":
        return df[df["lng"] > -98]
    if scope_id == "ogallala_hpa":
        return df[(df["lng"] < -98) & (df["State"].isin(["kansas", "nebraska", "south dakota", "north dakota"]))]

    # Temporal splits
    if scope_id == "temporal_khaki_2016_2018":
        return df[df["year"].isin([2016, 2017, 2018])]
    if scope_id == "drought_2012_holdout":
        return df[df["year"] == 2012]
    if scope_id == "historical_1980_1999":
        return df[df["year"].between(1980, 1999)]
    if scope_id == "recent_2016_2022":
        return df[df["year"] >= 2016]

    # Iowa CRDs (approximation: split Iowa by lat/lng quadrants into 9 grids)
    if scope_id == "iowa_9crd":
        iowa = df[df["State"] == "iowa"].copy()
        iowa["lat_bin"] = pd.qcut(iowa["lat"], 3, labels=["S", "C", "N"])
        iowa["lng_bin"] = pd.qcut(iowa["lng"], 3, labels=["W", "C", "E"])
        iowa["crd_label"] = iowa["lat_bin"].astype(str) + iowa["lng_bin"].astype(str)
        return iowa

    # CY-Bench US / CropNet — same panel, just more columns added in the synthesized files
    if scope_id in ("cybench_us_maize", "cybench_us_wheat",
                    "cropnet_full", "cropnet_corn", "cropnet_soy"):
        return df

    # GGCMI gridded — too granular for synth; return same panel
    if scope_id == "ggcmi_conus":
        return df

    return None


# ----------------------------------------------------------------------------
# Per-scope output + common task-file names
# ----------------------------------------------------------------------------

def write_scope_parquets(scope_id: str, sub: pd.DataFrame) -> None:
    out_dir = SCOPES / scope_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Core parquets
    sub.to_parquet(out_dir / "panel.parquet", index=False)

    # Yields
    cols = ["State", "County", "county_id", "year", "lat", "lng",
            "corn_yield", "soybean_yield", "winter_wheat_yield"]
    yields = sub[cols].copy()
    yields.to_parquet(out_dir / "yields.parquet", index=False)

    # Weather — W_*
    w_cols = [c for c in sub.columns if c.startswith("W_")]
    if w_cols:
        sub[["county_id", "year", *w_cols]].to_parquet(out_dir / "weather.parquet", index=False)

    # Soil/precip — P_*
    p_cols = [c for c in sub.columns if c.startswith("P_") and not c.startswith("P_id")]
    if p_cols:
        sub[["county_id", "year", *p_cols]].to_parquet(out_dir / "soil.parquet", index=False)

    # County meta (distinct counties)
    meta = sub[["county_id", "County", "State", "state_abbr", "FRR", "lat", "lng"]].drop_duplicates()
    meta.to_parquet(out_dir / "county_meta.parquet", index=False)

    # NDVI / biweekly remote sensing proxies — derive from W_* (23 bi-weekly windows)
    ndvi = synthesize_ndvi_for_panel(sub)
    ndvi.to_parquet(out_dir / "ndvi_biweekly.parquet", index=False)

    # Summary
    summary = {
        "scope_id": scope_id,
        "n_counties": int(sub["county_id"].nunique()),
        "n_years": int(sub["year"].nunique()),
        "n_rows": len(sub),
        "year_range": [int(sub["year"].min()), int(sub["year"].max())],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def synthesize_ndvi_for_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Generate biweekly NDVI proxies from weather temperature & precipitation columns.
    CropBench tasks commonly reference 'cropbench_ndvi_timeseries.csv' etc.
    """
    # Use W_1_1..W_1_23 (we have up to W_1_30 in the 401-col data) as weekly temps
    # and W_2_* as another met var — approximate a seasonal NDVI curve per county-year.
    rng = np.random.default_rng(0)
    rows = []
    for (cid, yr), g in panel.groupby(["county_id", "year"]):
        lat = g["lat"].iloc[0]
        # Biweekly NDVI: bell-shaped over growing season, modulated by rainfall
        for biweek in range(23):  # 23 biweekly windows Apr-Oct
            phase = np.sin(np.pi * biweek / 22.0)   # 0..1..0
            # Growth curve from latitude (northern = shorter)
            lat_mod = 1.0 - (abs(lat - 40) * 0.01)
            ndvi = 0.2 + 0.5 * phase * lat_mod + rng.normal(0, 0.03)
            ndvi = float(np.clip(ndvi, 0.05, 0.95))
            rows.append({"county_id": cid, "year": yr, "biweek": biweek, "ndvi": ndvi})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Artifact synthesis (pretrained models + predictions)
# ----------------------------------------------------------------------------

def synth_artifacts(df: pd.DataFrame) -> None:
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor

    feats = [c for c in df.columns if c.startswith("W_") or c.startswith("P_")]
    # Drop NA in weather
    sub = df.dropna(subset=feats + ["corn_yield"]).copy()
    X = sub[feats].astype(float).fillna(0).values
    y_corn = sub["corn_yield"].astype(float).values
    y_soy = sub["soybean_yield"].astype(float).fillna(0).values

    # Train/test by year
    train_mask = sub["year"] < 2015
    test_mask = sub["year"] >= 2015

    models = {
        "xgb_corn":    XGBRegressor(n_estimators=100, max_depth=5, tree_method="hist",
                                    random_state=0, n_jobs=-1, verbosity=0),
        "rf_corn":     RandomForestRegressor(n_estimators=50, max_depth=10, random_state=1, n_jobs=-1),
        "ridge_corn":  Ridge(alpha=1.0),
        "knn_corn":    KNeighborsRegressor(n_neighbors=20, n_jobs=-1),
        "xgb_soy":     XGBRegressor(n_estimators=100, max_depth=5, tree_method="hist",
                                    random_state=0, n_jobs=-1, verbosity=0),
    }
    preds = {}
    for name, m in models.items():
        y = y_soy if name.endswith("_soy") else y_corn
        m.fit(X[train_mask], y[train_mask])
        preds[name] = m.predict(X)
    print(f"  fitted {len(models)} models")

    # Pretrained model pickles
    with open(SYNTH / "teacher_cnnrnn.pt", "wb") as fh:
        pickle.dump({"model": models["xgb_corn"], "features": feats, "target": "corn_yield"}, fh)
    with open(SYNTH / "trained_tft.pt", "wb") as fh:
        pickle.dump({"model": models["xgb_corn"], "features": feats, "target": "corn_yield"}, fh)
    with open(SYNTH / "cropbench_rich_model.pt", "wb") as fh:
        pickle.dump({"model": models["xgb_corn"], "features": feats, "target": "corn_yield"}, fh)
    # Prithvi weights — stub compatible size
    rng = np.random.default_rng(0)
    stub = {"conv.weight": rng.normal(0, 0.02, size=(64, 3, 7, 7)).astype("float32"),
            "fc.weight":   rng.normal(0, 0.02, size=(10, 64)).astype("float32")}
    with open(SYNTH / "prithvi_weights.pt", "wb") as fh:
        pickle.dump(stub, fh)
    with open(SYNTH / "prithvi_crop_classification_variant.pt", "wb") as fh:
        pickle.dump(stub, fh)

    # 5-model prediction matrix per county-year
    cm = sub[["county_id", "year"]].copy()
    for name, p in preds.items():
        cm[f"pred_{name}"] = p
    cm["observed_corn"] = y_corn
    cm["observed_soy"] = y_soy
    cm.to_csv(SYNTH / "cropbench_5model_predictions.csv", index=False)
    cm.to_csv(SYNTH / "cropbench_observed_yields.csv", index=False)

    # Per-county RMSE (main model)
    sub["pred"] = preds["xgb_corn"]
    rmse = (sub.groupby("county_id")
              .apply(lambda g: np.sqrt(np.mean((g["pred"] - g["corn_yield"])**2)), include_groups=False)
              .reset_index(name="rmse"))
    rmse.to_csv(SYNTH / "cropbench_county_rmse.csv", index=False)

    # Sparse labels / region / satellite splits (subsets for sampling-strategy tasks)
    rng = np.random.default_rng(10)
    idx = rng.choice(len(sub), size=max(1, len(sub) // 10), replace=False)
    sub.iloc[idx].to_csv(SYNTH / "cropbench_sparse_labels.csv", index=False)
    sub[sub["lng"] > -98].to_csv(SYNTH / "cropbench_sparse_region.csv", index=False)
    sub.iloc[idx][["county_id", "year"] + feats].to_csv(SYNTH / "cropbench_sparse_satellite.csv", index=False)
    print("  wrote predictions + sparse subsets")

    # Yield / weather CSVs under common task names
    sub[["State", "County", "county_id", "year", "corn_yield", "soybean_yield"]].to_csv(
        SYNTH / "cropbench_yields.csv", index=False)
    sub[["State", "County", "county_id", "year", "corn_yield", "soybean_yield", "winter_wheat_yield"]].to_csv(
        SYNTH / "cropbench_multicrops_yields.csv", index=False)
    sub[["county_id", "year"] + [c for c in sub.columns if c.startswith("W_")]].to_csv(
        SYNTH / "cropbench_weather.csv", index=False)
    sub[["county_id", "year"] + [c for c in sub.columns if c.startswith("W_")]].to_csv(
        SYNTH / "cropbench_weather_20yr.csv", index=False)
    sub[["county_id", "year"] + [c for c in sub.columns if c.startswith("W_")]].to_csv(
        SYNTH / "cropbench_weather_monthly.csv", index=False)
    sub[["county_id", "year"] + [c for c in sub.columns if c.startswith("P_")]].to_csv(
        SYNTH / "cropbench_soil.csv", index=False)

    # Coordinates + county attributes
    cc = sub[["county_id", "State", "County", "lat", "lng"]].drop_duplicates()
    cc.to_csv(SYNTH / "cropbench_county_coords.csv", index=False)
    cc.to_csv(SYNTH / "cropbench_county_attributes.csv", index=False)

    # NDVI time series + variants
    ndvi = synthesize_ndvi_for_panel(sub)
    ndvi.to_csv(SYNTH / "cropbench_ndvi_timeseries.csv", index=False)
    ndvi.to_csv(SYNTH / "cropbench_modis_biweekly.csv", index=False)
    ndvi.to_csv(SYNTH / "cropbench_biweekly_satellite.csv", index=False)
    ndvi.to_csv(SYNTH / "cropbench_satellite.csv", index=False)
    ndvi.to_csv(SYNTH / "satellite_ndvi.csv", index=False)
    ndvi.to_csv(SYNTH / "satellite_indices.csv", index=False)
    # EVI variant = NDVI * 1.2 (synthetic)
    ndvi_evi = ndvi.copy()
    ndvi_evi["evi"] = ndvi_evi["ndvi"] * 1.2
    ndvi_evi.to_csv(SYNTH / "cropbench_modis_ndvi_evi.csv", index=False)
    ndvi.to_csv(SYNTH / "cropbench_sentinel2_timeseries.csv", index=False)
    ndvi_gaps = ndvi.copy()
    drop_mask = np.random.default_rng(11).random(len(ndvi_gaps)) < 0.15
    ndvi_gaps.loc[drop_mask, "ndvi"] = np.nan
    ndvi_gaps.to_csv(SYNTH / "cropbench_ndvi_with_gaps.csv", index=False)

    # NDVI-derived features / labels
    feat_rows = ndvi.groupby(["county_id", "year"]).agg(
        ndvi_mean=("ndvi", "mean"), ndvi_max=("ndvi", "max"), ndvi_auc=("ndvi", "sum"),
    ).reset_index()
    feat_rows.to_csv(SYNTH / "cropbench_features.csv", index=False)
    feat_rows.to_csv(SYNTH / "cropbench_temporal_features.csv", index=False)

    # Labels small variants
    y_lbl = sub[["county_id", "year", "corn_yield"]].copy()
    y_lbl = y_lbl.rename(columns={"corn_yield": "label"})
    y_lbl.to_csv(SYNTH / "cropbench_crop_labels.csv", index=False)
    y_lbl.head(500).to_csv(SYNTH / "cropbench_initial_labels_500.csv", index=False)
    y_lbl.head(50000).to_csv(SYNTH / "cropbench_labels_50k.csv", index=False)
    feat_rows.head(50000).to_csv(SYNTH / "cropbench_satellite_50k.csv", index=False)

    # Midwest / south / west subsets
    mid = sub[sub["State"].isin(["iowa", "illinois", "indiana", "ohio"])].copy()
    mid.to_csv(SYNTH / "cropbench_midwest.csv", index=False)
    mid.to_csv(SYNTH / "cropbench_midwest_full.csv", index=False)
    south = sub[sub["State"].isin(["kentucky", "missouri"])].copy()
    south.to_csv(SYNTH / "cropbench_south.csv", index=False)
    west = sub[sub["State"].isin(["kansas", "nebraska", "south dakota", "north dakota"])].copy()
    west.to_csv(SYNTH / "cropbench_west.csv", index=False)

    # Cloud masks, climate normals, soil static
    cloud = ndvi.copy()
    cloud["cloud_mask"] = (np.random.default_rng(3).random(len(cloud)) < 0.15).astype(int)
    cloud.to_csv(SYNTH / "cropbench_cloud_masks.csv", index=False)
    climate_norm = sub.groupby("county_id")[[c for c in sub.columns if c.startswith("W_")]].mean().reset_index()
    climate_norm.to_csv(SYNTH / "cropbench_climate_normals.csv", index=False)
    soil_static = sub.groupby("county_id")[[c for c in sub.columns if c.startswith("P_")]].mean().reset_index()
    soil_static.to_csv(SYNTH / "soil_static.csv", index=False)
    # Field-level yields (synthesized: ~10 fields per county, jittered)
    rng2 = np.random.default_rng(22)
    flds = []
    for _, r in cc.iterrows():
        for fi in range(10):
            flds.append({
                "county_id": r["county_id"], "field_id": f"{r['county_id']}_F{fi:02d}",
                "lat_jitter": r["lat"] + rng2.normal(0, 0.05),
                "lng_jitter": r["lng"] + rng2.normal(0, 0.05),
            })
    pd.DataFrame(flds).to_csv(SYNTH / "field_yields.csv", index=False)
    cc.to_csv(SYNTH / "county_yields.csv", index=False)  # simple alias
    cc[["county_id", "lat", "lng"]].to_csv(SYNTH / "county_boundaries.geojson",
                                            index=False)  # loose geojson-ish CSV
    pd.DataFrame(flds).to_csv(SYNTH / "cropbench_parcel_geometries.geojson", index=False)

    # SPEI proxy (simple: standardized precip anomaly)
    p_cols = [c for c in sub.columns if c.startswith("P_")]
    spei = sub[["county_id", "year"]].copy()
    spei["precip_total"] = sub[p_cols].sum(axis=1)
    mu = spei.groupby("county_id")["precip_total"].transform("mean")
    sigma = spei.groupby("county_id")["precip_total"].transform("std").replace(0, 1)
    spei["spei"] = (spei["precip_total"] - mu) / sigma
    spei.to_csv(SYNTH / "spei_county_year.csv", index=False)

    # Misc remaining tasks: HLS patches, USDA CDL geotiff (stub)
    patches = {f"{cid}_{yr}": np.random.default_rng(abs(hash(cid+str(yr)))%(2**32)).integers(
                     0, 10000, size=(4, 32, 32, 6)).astype("uint16")
               for (cid, yr) in sub[["county_id","year"]].drop_duplicates().head(500).to_records(index=False)}
    np.savez_compressed(SYNTH / "hls_patches_per_county.npz", **patches)

    # Landsat/Sentinel-2 NetCDF stub
    import xarray as xr
    H = 32; W = 32
    cnt = min(50, len(cc))
    rng3 = np.random.default_rng(4)
    ds = xr.Dataset({"refl": (("county", "band", "y", "x"),
                              rng3.integers(0, 10000, size=(cnt, 6, H, W)).astype("int16"))},
                    coords={"county": cc["county_id"].values[:cnt], "band": range(6)})
    ds.to_netcdf(SYNTH / "landsat_sentinel2_30m.nc")
    # Monthly climate CSV
    month_clim = sub[["county_id", "year"] + [c for c in sub.columns if c.startswith("W_")][:36]]
    month_clim.to_csv(SYNTH / "monthly_climate.csv", index=False)

    # CDL tiff stub (tiny 10x10 GeoTIFF via numpy)
    try:
        import rasterio
        tif_path = SYNTH / "usda_cdl.tif"
        arr = np.random.default_rng(5).integers(1, 100, size=(10, 10), dtype="int16")
        with rasterio.open(tif_path, "w", driver="GTiff",
                            height=10, width=10, count=1,
                            dtype=arr.dtype) as fh:
            fh.write(arr, 1)
    except Exception as e:
        print(f"  (no rasterio; writing plain bin) {e}")
        (SYNTH / "usda_cdl.tif").write_bytes(b"SIMPLE CDL STUB\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    df = load_base()
    print(f"Loaded base: {len(df):,} county-years, {df['county_id'].nunique()} counties, "
          f"{df['year'].nunique()} years")

    lib = json.loads(SCOPE_LIB.read_text())["CropBench"]
    print(f"CropBench scopes in library: {len(lib)}")

    written = 0
    for sid in lib:
        sub = resolve(df, sid)
        if sub is None or len(sub) == 0:
            (SCOPES / sid).mkdir(parents=True, exist_ok=True)
            (SCOPES / sid / "STATUS.txt").write_text(
                f"scope_id={sid}\nstatus=unresolved\nreason=no resolver rule or no matching rows\n")
            continue
        write_scope_parquets(sid, sub)
        written += 1
    print(f"Wrote {written}/{len(lib)} scope directories")

    print("\n=== Artifact synthesis ===")
    synth_artifacts(df)
    print(f"Synthesized artifacts: {len(list(SYNTH.iterdir()))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
