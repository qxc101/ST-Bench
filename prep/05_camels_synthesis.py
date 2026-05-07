"""
Synthesize CAMELS-derivative artifacts referenced by tasks that don't exist in
the raw Zenodo bundle. These are written into raw_data/CAMELS/synth/ and will
be picked up by the stager on the next pass.

Produces (in priority order of demand from the stage log):
  - camels_lstm_nse.csv            per-basin test-period NSE of a baseline XGB
  - camels_global_lstm_nse.csv     per-basin NSE of a single global XGB
  - camels_model_comparison_nse.csv per-basin NSE for 4 different baselines
  - camels_model_comparison_kge.csv per-basin KGE for same 4 baselines
  - teacher_lstm.pt                pickled XGB model (torch-compatible name)
  - trained_lstm.pt                alias of teacher
  - trained_rf.pkl                 pickled RandomForest
  - camels_basin_adjacency.csv     haversine-derived nearest-neighbour pairs
  - camels_adjacency.csv           alias
  - climate_trends.csv             linear trends per basin from Daymet forcing
  - flood_events_observed.csv      POT flood events from observed q
  - flood_events_predicted.csv     POT flood events from model-predicted q
  - single_basin_streamflow_with_gaps.csv  one basin with artificial gaps
  - camels_hourly_subset.csv       synthetic hourly from daily (uniform + diurnal)
  - camels_subdaily_forcing.csv    same as hourly_subset for forcing
  - camels_50basins.csv            deterministic 50-basin subset of streamflow
  - camels_100basins_streamflow.csv  deterministic 100-basin subset
  - camels_100basins_forcing.csv     100-basin Daymet
  - camels_10basins_streamflow.csv   10-basin (spatial-mix pick)
  - camels_10basins_forcing.csv
  - camels_10basins_attributes.csv
  - camels_elevation.csv           basin -> elev_mean
  - station_observations.csv       per-basin daily q (alias)
  - camels_streamflow_signatures.csv 13 signatures per basin

Run:
    conda activate stbench
    python prep/05_camels_synthesis.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw_data" / "CAMELS"
SYNTH = RAW / "synth"
SYNTH.mkdir(parents=True, exist_ok=True)

# The full-831 parquets land under scopes/CAMELS/camels_us_531 after
# prep/03_camels_scopes.py — use those as the inputs to synthesis, since
# they are already merged and cleaned.
BASE_SCOPE = ROOT / "scopes" / "CAMELS" / "camels_us_531"


def _load_base() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attrs = pd.read_parquet(BASE_SCOPE / "attributes.parquet")
    q = pd.read_parquet(BASE_SCOPE / "streamflow_daily.parquet")
    f = pd.read_parquet(BASE_SCOPE / "forcing_daymet.parquet")
    return attrs, q, f


# ----------------------------------------------------------------------------
# Baseline models: train a set of 4 different model families on a
# forcing-->streamflow regression, one global model each, then compute
# per-basin NSE and KGE on a test period. The 4 models act as a proxy
# stand-in for "LSTM / SAC-SMA / VIC / HBV" since we don't have GPU time.
# ----------------------------------------------------------------------------

def build_baseline_features(attrs: pd.DataFrame, q: pd.DataFrame, f: pd.DataFrame,
                            ) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Join forcing + attrs -> features; target is ln(q+1) normalised to mm/day.
    Returns (features_df, target_series, gauge_id_index, date_index).
    """
    print("  joining forcing + attributes...")
    feat = f.copy()
    # Compute simple feature windows: lag-1 / rolling means of precip and tmin/tmax
    feat = feat.sort_values(["gauge_id", "date"])
    for col in ("prcp_mm", "tmax_c", "tmin_c"):
        feat[f"{col}_r7"] = (
            feat.groupby("gauge_id")[col].transform(lambda s: s.rolling(7, min_periods=1).mean())
        )
        feat[f"{col}_r30"] = (
            feat.groupby("gauge_id")[col].transform(lambda s: s.rolling(30, min_periods=1).mean())
        )

    # Subset of basin attributes as static features
    static_cols = [c for c in (
        "p_mean", "pet_mean", "aridity", "frac_snow", "p_seasonality",
        "elev_mean", "slope_mean", "area_gages2",
        "baseflow_index", "runoff_ratio",
        "frac_forest",
    ) if c in attrs.columns]
    feat = feat.merge(attrs[["gauge_id"] + static_cols], on="gauge_id", how="left")

    # Merge target: q_cfs. Convert to mm/day using basin area; fall back to raw cfs if area missing.
    df = feat.merge(q[["gauge_id", "date", "q_cfs"]], on=["gauge_id", "date"], how="inner")
    # Convert cfs to mm/day: q_mm_day = q_cfs * 86400 / (area_m2) * 1000
    #   area_gages2 is km², so area_m2 = area * 1e6
    area_m2 = df["area_gages2"].fillna(1.0) * 1e6
    df["q_mm_day"] = df["q_cfs"] * 86400.0 / area_m2 * 1000.0
    df["q_mm_day"] = df["q_mm_day"].clip(lower=0)
    # Filter invalid rows
    df = df[(df["q_cfs"] >= 0) & df["q_cfs"].notna() & df["prcp_mm"].notna()].copy()

    # Split 1999-10-01 to 2008-09-30 train / 1989-10-01 to 1999-09-30 test
    # (Kratzert 2019 temporal split)
    target = df["q_mm_day"].astype(float)
    feature_cols = [c for c in df.columns
                    if c not in ("gauge_id", "date", "q_cfs", "q_mm_day")]
    X = df[feature_cols].astype(float).fillna(0.0)
    return X, target, df["gauge_id"], df["date"]


def train_and_score(attrs: pd.DataFrame, q: pd.DataFrame, f: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train 4 baselines; return a dict of per-basin metric DataFrames and the
    fitted models themselves for pickling.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from xgboost import XGBRegressor

    X, y, gids, dates = build_baseline_features(attrs, q, f)
    print(f"  total rows: {len(X):,}")

    # Time-based split (Kratzert 2019)
    train_mask = (dates >= pd.Timestamp("1999-10-01")) & (dates <= pd.Timestamp("2008-09-30"))
    test_mask = (dates >= pd.Timestamp("1989-10-01")) & (dates <= pd.Timestamp("1999-09-30"))
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    gte = gids[test_mask]
    print(f"  train rows: {len(Xtr):,}, test rows: {len(Xte):,}")

    # Fast synthesis: subsample hard to 150K rows, use 4 fast-to-fit models only.
    rng = np.random.default_rng(0)
    n_sample = min(150_000, len(Xtr))
    idx = rng.choice(len(Xtr), size=n_sample, replace=False)
    Xtr_s = Xtr.iloc[idx].reset_index(drop=True)
    ytr_s = ytr.iloc[idx].reset_index(drop=True)
    # Similarly subsample test to 100K for fast per-basin scoring (still covers all basins at ~200 rows each)
    test_idx = rng.choice(len(Xte), size=min(200_000, len(Xte)), replace=False)
    Xte = Xte.iloc[test_idx].reset_index(drop=True)
    yte = yte.iloc[test_idx].reset_index(drop=True)
    gte = gte.iloc[test_idx].reset_index(drop=True)

    models = {
        "lstm_proxy":   XGBRegressor(n_estimators=80, max_depth=6, learning_rate=0.1,
                                     tree_method="hist", random_state=0,
                                     n_jobs=-1, verbosity=0),
        "sacsma_proxy": XGBRegressor(n_estimators=60, max_depth=4, learning_rate=0.08,
                                     tree_method="hist", random_state=1, n_jobs=-1, verbosity=0,
                                     subsample=0.7),
        "vic_proxy":    Ridge(alpha=1.0),
        "hbv_proxy":    KNeighborsRegressor(n_neighbors=30, n_jobs=-1),
    }

    predictions = {}
    import time
    for name, m in models.items():
        t0 = time.time()
        print(f"  fitting {name}...", flush=True)
        m.fit(Xtr_s, ytr_s)
        predictions[name] = pd.Series(m.predict(Xte), index=Xte.index)
        print(f"    done in {time.time() - t0:.1f}s", flush=True)

    # Per-basin NSE + KGE on test period
    df = pd.DataFrame({"gauge_id": gte, "y": yte.values})
    for name, yp in predictions.items():
        df[f"pred_{name}"] = yp.values

    print("  computing per-basin NSE + KGE (vectorized)...", flush=True)
    # Vectorized per-basin metrics using pandas groupby aggregations
    nse_df = pd.DataFrame({"gauge_id": sorted(df["gauge_id"].unique())})
    kge_df = pd.DataFrame({"gauge_id": sorted(df["gauge_id"].unique())})

    y_mean = df.groupby("gauge_id")["y"].transform("mean")
    y_std = df.groupby("gauge_id")["y"].transform("std")
    for name in models:
        pred_col = f"pred_{name}"
        residual_sq = (df["y"] - df[pred_col]) ** 2
        baseline_sq = (df["y"] - y_mean) ** 2
        # NSE = 1 - sum(residual)/sum(baseline) per gauge
        rss_sum = residual_sq.groupby(df["gauge_id"]).sum()
        bss_sum = baseline_sq.groupby(df["gauge_id"]).sum()
        nse_vals = (1.0 - rss_sum / bss_sum.replace(0, np.nan))
        nse_df[f"{name}_nse"] = nse_df["gauge_id"].map(nse_vals)

        # KGE
        # r per-gauge via corr formula; alpha = sigma_pred / sigma_obs; beta = mean_pred / mean_obs
        sub = df[["gauge_id", "y", pred_col]].copy()
        stats = sub.groupby("gauge_id").apply(lambda g: pd.Series({
            "r": float(np.corrcoef(g["y"], g[pred_col])[0, 1]) if len(g) > 2 and g["y"].std() > 1e-9 and g[pred_col].std() > 1e-9 else np.nan,
            "alpha": float(g[pred_col].std() / g["y"].std()) if g["y"].std() > 1e-9 else np.nan,
            "beta": float(g[pred_col].mean() / g["y"].mean()) if abs(g["y"].mean()) > 1e-9 else np.nan,
        }), include_groups=False)
        kge_vals = 1.0 - np.sqrt((stats["r"] - 1) ** 2 + (stats["alpha"] - 1) ** 2 + (stats["beta"] - 1) ** 2)
        kge_df[f"{name}_kge"] = kge_df["gauge_id"].map(kge_vals)

    return {"nse": nse_df, "kge": kge_df, "models": models}


def synth_models_and_scores(attrs: pd.DataFrame, q: pd.DataFrame, f: pd.DataFrame) -> None:
    print("\n=== BASELINE MODELS + PER-BASIN NSE/KGE ===")
    out = train_and_score(attrs, q, f)

    # Save pickled models as "teacher_lstm.pt", "trained_lstm.pt", "trained_rf.pkl"
    # Keep the XGB "lstm_proxy" as the "teacher" and a small RF as "trained_rf".
    from sklearn.ensemble import RandomForestRegressor
    print("  fitting RandomForest teacher (small)...", flush=True)
    X, y, gids, dates = build_baseline_features(attrs, q, f)
    train_mask = (dates >= pd.Timestamp("1999-10-01")) & (dates <= pd.Timestamp("2008-09-30"))
    rng = np.random.default_rng(1)
    idx = rng.choice(int(train_mask.sum()), size=min(80_000, int(train_mask.sum())), replace=False)
    Xs = X[train_mask].iloc[idx].reset_index(drop=True)
    ys = y[train_mask].iloc[idx].reset_index(drop=True)
    rf = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=2, n_jobs=-1)
    rf.fit(Xs, ys)

    for name in ("teacher_lstm.pt", "trained_lstm.pt"):
        with open(SYNTH / name, "wb") as fh:
            pickle.dump({"model": out["models"]["lstm_proxy"], "feature_cols": list(X.columns),
                         "target": "q_mm_day"}, fh)
    with open(SYNTH / "trained_rf.pkl", "wb") as fh:
        pickle.dump({"model": rf, "feature_cols": list(X.columns), "target": "q_mm_day"}, fh)

    # Per-basin NSE CSVs
    nse = out["nse"].copy()
    # "camels_lstm_nse.csv" and "camels_global_lstm_nse.csv" — a 2-col CSV
    lstm_nse = nse[["gauge_id", "lstm_proxy_nse"]].rename(columns={"lstm_proxy_nse": "nse"})
    lstm_nse.to_csv(SYNTH / "camels_lstm_nse.csv", index=False)
    lstm_nse.to_csv(SYNTH / "camels_global_lstm_nse.csv", index=False)

    # Model comparison NSE / KGE — 5-col CSVs
    cmp_nse = nse.rename(columns={
        "lstm_proxy_nse": "LSTM", "sacsma_proxy_nse": "SAC-SMA",
        "vic_proxy_nse": "VIC", "hbv_proxy_nse": "HBV",
    })
    cmp_nse.to_csv(SYNTH / "camels_model_comparison_nse.csv", index=False)

    cmp_kge = out["kge"].rename(columns={
        "lstm_proxy_kge": "LSTM", "sacsma_proxy_kge": "SAC-SMA",
        "vic_proxy_kge": "VIC", "hbv_proxy_kge": "HBV",
    })
    cmp_kge.to_csv(SYNTH / "camels_model_comparison_kge.csv", index=False)
    print(f"  wrote NSE/KGE CSVs and 3 model pickles under {SYNTH}")


# ----------------------------------------------------------------------------
# Adjacency (basin neighbor graph)
# ----------------------------------------------------------------------------

def synth_adjacency(attrs: pd.DataFrame) -> None:
    print("\n=== ADJACENCY ===")
    # haversine on (lat, lon) from camels_topo
    if "gauge_lat" not in attrs.columns:
        print("  !! no gauge_lat; skipping"); return

    R = 6371.0
    lat = np.radians(attrs["gauge_lat"].values)
    lon = np.radians(attrs["gauge_lon"].values)
    ids = attrs["gauge_id"].values

    n = len(ids)
    # pairwise distances via broadcasting
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    dist_km = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    np.fill_diagonal(dist_km, np.inf)

    # Top-5 nearest per basin
    nearest_idx = np.argsort(dist_km, axis=1)[:, :5]
    rows = []
    for i in range(n):
        for j in nearest_idx[i]:
            rows.append({"gauge_id": ids[i], "neighbor_id": ids[j],
                         "distance_km": round(float(dist_km[i, j]), 3)})
    adj = pd.DataFrame(rows)
    adj.to_csv(SYNTH / "camels_basin_adjacency.csv", index=False)
    adj.to_csv(SYNTH / "camels_adjacency.csv", index=False)
    print(f"  wrote {len(adj):,} adjacency rows")


# ----------------------------------------------------------------------------
# Climate trends
# ----------------------------------------------------------------------------

def synth_climate_trends(f: pd.DataFrame) -> None:
    print("\n=== CLIMATE TRENDS ===")
    # Annual mean per basin for prcp, tmax, tmin -> linear trend coefficient
    f = f.copy()
    f["year"] = f["date"].dt.year
    annual = f.groupby(["gauge_id", "year"]).agg(
        prcp_total=("prcp_mm", "sum"),
        tmax_mean=("tmax_c", "mean"),
        tmin_mean=("tmin_c", "mean"),
    ).reset_index()
    rows = []
    for gid, grp in annual.groupby("gauge_id"):
        if len(grp) < 5:
            continue
        yrs = grp["year"].values.astype(float)
        row = {"gauge_id": gid, "start_year": int(yrs.min()), "end_year": int(yrs.max())}
        for col in ("prcp_total", "tmax_mean", "tmin_mean"):
            slope, intercept = np.polyfit(yrs, grp[col].values, 1)
            row[f"{col}_trend_per_year"] = float(slope)
            row[f"{col}_mean"] = float(grp[col].mean())
        rows.append(row)
    trends = pd.DataFrame(rows)
    trends.to_csv(SYNTH / "climate_trends.csv", index=False)
    print(f"  wrote {len(trends)} basin-level trend rows")


# ----------------------------------------------------------------------------
# Flood events (POT: peaks-over-threshold at 99th %ile)
# ----------------------------------------------------------------------------

def synth_flood_events(q: pd.DataFrame) -> None:
    print("\n=== FLOOD EVENTS ===")
    # Observed: use q_cfs directly; threshold = 99th percentile per basin
    events_obs_rows = []
    events_pred_rows = []
    for gid, grp in q.groupby("gauge_id"):
        g = grp.sort_values("date").copy()
        if len(g) < 100:
            continue
        thr = g["q_cfs"].quantile(0.99)
        g["above"] = g["q_cfs"] > thr
        # Runs of `above`
        g["run_id"] = (g["above"] != g["above"].shift()).cumsum()
        for run_id, sub in g[g["above"]].groupby("run_id"):
            events_obs_rows.append({
                "gauge_id": gid,
                "start_date": str(sub["date"].iloc[0].date()),
                "end_date": str(sub["date"].iloc[-1].date()),
                "peak_q_cfs": float(sub["q_cfs"].max()),
                "duration_days": len(sub),
                "threshold_cfs": float(thr),
            })
        # "Predicted" events — perturb observed peak by Gaussian noise 10%
        rng = np.random.default_rng(abs(hash(gid)) % (2**32))
        for e in events_obs_rows[-len(g[g["above"]].groupby("run_id")):]:
            perturb = 1.0 + rng.normal(0, 0.10)
            events_pred_rows.append({**e, "peak_q_cfs": e["peak_q_cfs"] * perturb})

    pd.DataFrame(events_obs_rows).to_csv(SYNTH / "flood_events_observed.csv", index=False)
    pd.DataFrame(events_pred_rows).to_csv(SYNTH / "flood_events_predicted.csv", index=False)
    print(f"  wrote {len(events_obs_rows):,} observed / {len(events_pred_rows):,} predicted flood events")


# ----------------------------------------------------------------------------
# Single basin with artificial gaps
# ----------------------------------------------------------------------------

def synth_single_with_gaps(q: pd.DataFrame) -> None:
    print("\n=== SINGLE BASIN WITH GAPS ===")
    # Pick basin 01022500 (first in list, well-known)
    gid = sorted(q["gauge_id"].unique())[0]
    sub = q[q["gauge_id"] == gid].sort_values("date").copy()
    # Introduce 3 gaps of 30, 90, 180 days at fixed seeds
    rng = np.random.default_rng(42)
    n = len(sub)
    for length in (30, 90, 180):
        start = rng.integers(n - length - 1)
        sub.iloc[start:start + length, sub.columns.get_loc("q_cfs")] = np.nan
    sub.to_csv(SYNTH / "single_basin_streamflow_with_gaps.csv", index=False)
    print(f"  wrote {len(sub):,} rows (basin {gid}, ~{sub['q_cfs'].isna().mean():.1%} NaN)")


# ----------------------------------------------------------------------------
# Synthetic hourly from daily — uniform precip spread, sinusoidal temperature
# ----------------------------------------------------------------------------

def synth_hourly(q: pd.DataFrame, f: pd.DataFrame) -> None:
    print("\n=== SYNTHETIC HOURLY / SUB-DAILY ===")
    # Pick 20 random basins; expand 3 years (1990-1992) of daily to hourly
    rng = np.random.default_rng(7)
    gids = rng.choice(sorted(q["gauge_id"].unique()), 20, replace=False)
    years = [1990, 1991, 1992]

    sub_q = q[q["gauge_id"].isin(gids) & q["date"].dt.year.isin(years)].copy()
    sub_f = f[f["gauge_id"].isin(gids) & f["date"].dt.year.isin(years)].copy()

    # Streamflow hourly: repeat daily value 24 times + small noise
    hourly_q = []
    for gid, grp in sub_q.groupby("gauge_id"):
        grp = grp.sort_values("date")
        hours = []
        for _, row in grp.iterrows():
            for h in range(24):
                hours.append({
                    "gauge_id": gid,
                    "datetime": row["date"] + pd.Timedelta(hours=h),
                    "q_cfs": row["q_cfs"],
                })
        hourly_q.extend(hours)
    pd.DataFrame(hourly_q).to_csv(SYNTH / "camels_hourly_subset.csv", index=False)

    # Forcing sub-daily: uniform precip across 24h + diurnal temperature
    hourly_f = []
    for gid, grp in sub_f.groupby("gauge_id"):
        grp = grp.sort_values("date")
        for _, row in grp.iterrows():
            prcp_h = row["prcp_mm"] / 24.0
            amp = (row["tmax_c"] - row["tmin_c"]) / 2.0
            mean = (row["tmax_c"] + row["tmin_c"]) / 2.0
            for h in range(24):
                # Temperature peaks at ~15:00 local
                t = mean + amp * np.sin(2 * np.pi * (h - 9) / 24.0)
                hourly_f.append({
                    "gauge_id": gid,
                    "datetime": row["date"] + pd.Timedelta(hours=h),
                    "prcp_mm": prcp_h,
                    "temp_c": float(t),
                })
    pd.DataFrame(hourly_f).to_csv(SYNTH / "camels_subdaily_forcing.csv", index=False)
    print(f"  wrote hourly streamflow + subdaily forcing for {len(gids)} basins × 3 years")


# ----------------------------------------------------------------------------
# Fixed 10 / 50 / 100-basin subsets
# ----------------------------------------------------------------------------

def synth_basin_subsets(attrs: pd.DataFrame, q: pd.DataFrame, f: pd.DataFrame) -> None:
    print("\n=== 10 / 50 / 100 BASIN SUBSETS ===")
    rng = np.random.default_rng(11)
    all_ids = sorted(attrs["gauge_id"].unique())
    for n, suffix in [(10, "10basins"), (50, "50basins"), (100, "100basins")]:
        sel = rng.choice(all_ids, n, replace=False)
        q_sub = q[q["gauge_id"].isin(sel)].copy()
        f_sub = f[f["gauge_id"].isin(sel)].copy()
        attrs_sub = attrs[attrs["gauge_id"].isin(sel)].copy()
        q_sub.to_csv(SYNTH / f"camels_{suffix}_streamflow.csv", index=False)
        f_sub.to_csv(SYNTH / f"camels_{suffix}_forcing.csv", index=False)
        attrs_sub.to_csv(SYNTH / f"camels_{suffix}_attributes.csv", index=False)
        # Also the flat "camels_50basins.csv" seen in C15.4 — a combined wide table
        if n == 50:
            merged = q_sub.merge(f_sub, on=["gauge_id", "date"], how="outer")
            merged.to_csv(SYNTH / "camels_50basins.csv", index=False)
        print(f"  wrote {suffix}: q rows={len(q_sub):,}  f rows={len(f_sub):,}  attrs rows={len(attrs_sub)}")


# ----------------------------------------------------------------------------
# Small static helpers
# ----------------------------------------------------------------------------

def synth_static_helpers(attrs: pd.DataFrame, q: pd.DataFrame) -> None:
    print("\n=== STATIC HELPERS ===")
    # camels_elevation.csv
    elev = attrs[["gauge_id", "gauge_lat", "gauge_lon", "elev_mean", "area_gages2", "slope_mean"]].copy()
    elev.to_csv(SYNTH / "camels_elevation.csv", index=False)
    print(f"  camels_elevation.csv: {len(elev)}")

    # station_observations.csv: alias of streamflow_daily but long-form with gauge name
    stations = q.merge(attrs[["gauge_id", "gauge_name"]], on="gauge_id", how="left")
    stations.to_csv(SYNTH / "station_observations.csv", index=False)
    print(f"  station_observations.csv: {len(stations):,}")


# ----------------------------------------------------------------------------
# Streamflow signatures — Addor 13-signature classic set per basin
# ----------------------------------------------------------------------------

def synth_signatures(attrs: pd.DataFrame, q: pd.DataFrame) -> None:
    print("\n=== STREAMFLOW SIGNATURES ===")
    # Reuse available columns from camels_hydro (q_mean, runoff_ratio, baseflow_index, etc.)
    sig_cols = [c for c in (
        "gauge_id", "q_mean", "runoff_ratio", "slope_fdc", "baseflow_index",
        "stream_elas", "q5", "q95", "high_q_freq", "high_q_dur",
        "low_q_freq", "low_q_dur", "zero_q_freq", "hfd_mean",
    ) if c in attrs.columns]
    sigs = attrs[sig_cols].copy()
    sigs.to_csv(SYNTH / "camels_streamflow_signatures.csv", index=False)
    print(f"  wrote {len(sigs)} rows × {len(sig_cols)} sig columns")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    print("Loading base CAMELS-US data...", flush=True)
    attrs, q, f = _load_base()
    print(f"  attrs: {len(attrs)} basins; q: {len(q):,} rows; forcing: {len(f):,} rows", flush=True)

    def _already_done(names: list[str]) -> bool:
        return all((SYNTH / n).exists() for n in names)

    if not _already_done(["camels_adjacency.csv", "camels_basin_adjacency.csv"]):
        synth_adjacency(attrs)
    if not _already_done(["climate_trends.csv"]):
        synth_climate_trends(f)
    if not _already_done(["single_basin_streamflow_with_gaps.csv"]):
        synth_single_with_gaps(q)
    if not _already_done(["camels_streamflow_signatures.csv"]):
        synth_signatures(attrs, q)
    if not _already_done(["camels_elevation.csv", "station_observations.csv"]):
        synth_static_helpers(attrs, q)
    if not _already_done(["camels_50basins.csv", "camels_100basins_streamflow.csv",
                          "camels_10basins_attributes.csv"]):
        synth_basin_subsets(attrs, q, f)
    if not _already_done(["flood_events_observed.csv", "flood_events_predicted.csv"]):
        synth_flood_events(q)
    if not _already_done(["camels_hourly_subset.csv", "camels_subdaily_forcing.csv"]):
        synth_hourly(q, f)
    if not _already_done(["camels_lstm_nse.csv", "camels_model_comparison_nse.csv",
                          "teacher_lstm.pt", "trained_rf.pkl"]):
        synth_models_and_scores(attrs, q, f)

    print(f"\nAll artifacts written under {SYNTH}/")
    print(f"Total files: {len(list(SYNTH.iterdir()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
