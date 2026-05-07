"""
MethaneWet synthesis.

FLUXNET-CH4 v1.0 (Delwiche 2021) and BAWLD-CH4 (Kuhn 2021) require registration
to access raw half-hourly data. Since ST_Bench's tasks are about analysis /
modelling workflows (not about the specific flux values), we synthesize a
realistic site-level daily panel from first principles using:

  * A hand-curated site list of ~80 sites grouped by wetland class and
    latitude band (matches the taxonomy in the Scope Sources PDF).
  * Site-level daily environmental drivers (air_temp, soil_temp at 5cm / 10cm,
    water_table_depth, GPP, NDVI, precip, radiation) simulated from site
    latitude with seasonal + noise.
  * Per-site daily CH4 flux driven by a parametric physics-inspired model:
        flux = base(class) * Q10^((T_soil-10)/10) * exp(-|WTD|*k) * gpp_factor

This is a deliberate synthesis — tasks that would normally need real FLUXNET
fluxes will be evaluated on this consistent synthesized dataset for both the
single-agent baseline and any MAS systems, so the comparison is fair.

Produces raw_data/MethaneWet/:
  * site_metadata.csv                     per-site lat/lon/class/climate_zone
  * xmethanewet_flux_timeseries.csv       site-daily CH4 flux
  * xmethanewet_environmental_drivers.csv site-daily env drivers
  * xmethanewet_flux_halfhourly.csv       half-hourly subset (first 10 sites × 2 yr)
  * xmethanewet_halfhourly_flux.csv       alias of above
  * xmethanewet_halfhourly_met.csv        half-hourly met for same subset
  * xmethanewet_drivers_halfhourly.csv    alias
  * xmethanewet_daily_flux.csv            alias of timeseries
  * xmethanewet_daily_drivers.csv         alias of environmental_drivers
  * xmethanewet_drivers.csv               alias
  * xmethanewet_all_drivers.csv           drivers + static
  * xmethanewet_site_metadata.csv         alias of site_metadata
  * xmethanewet_metadata.csv              alias
  * xmethanewet_site_info.csv             alias
  * xmethanewet_existing_sites.csv        alias
  * xmethanewet_tower_data.csv            flux + drivers joined
  * xmethanewet_tower_fluxes.csv          flux_timeseries alias
  * xmethanewet_boreal.csv, _temperate.csv, _tropical.csv  by climate_zone
  * xmethanewet_ec_sites.csv              EC sites (exclude chamber)
  * xmethanewet_chamber_sites.csv         BAWLD-style chamber plots
  * xmethanewet_daily_20sites.csv         20-site subset daily flux
  * xmethanewet_temporal_split.csv        per-site temporal train/test years
  * fluxnet_qc_flags.csv                  synthetic QC flags
  * methane_flux_daily.csv                alias of flux_timeseries
  * tower_observations.csv                alias
  * environmental_drivers.csv             alias
  * gridded_covariates_05deg.nc           synthetic gridded 0.5° covariates
  * tem_global_ch4.nc                     synthetic TEM-MDM-like model output
  * wetcharts_output.csv                  synthetic WetCHARTs-like global output
  * ml_ch4_predictions.csv                synthetic ML predictions for the panel
  * xmethanewet_site_performance.csv      per-site R2 of the ML predictions
  * era5_reforecast_sitelocations.csv     synthetic ERA5 reforecast at site points
  * monthly_ch4_totals.csv                site-month aggregated flux
  * ch4_monthly_observed.csv, ch4_monthly_predicted.csv
  * process_model_outputs.nc              synthetic process-model output
  * covariate_grids.nc                    synthetic 0.25° covariate grids
  * hls_patches_per_site.npz              small synthetic satellite patches
  * gridded_ch4_products.nc               synthetic multi-product gridded CH4
  * teacher_model.pt, trained_ch4_model.pkl  baseline RF on the synth panel
  * candidate_locations.csv               candidate new-site locations with cost
  * cost_model.json                       deployment cost model
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "raw_data" / "MethaneWet"
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Site list — curated from Delwiche 2021 ESSD Table 2 + Scope Sources PDF
# Classes: bog, fen, marsh, swamp, wet_tundra, rice, brackish/saline, upland, drained
# Climate zones: arctic, boreal, temperate, tropical
# Lat/lon are either the published values or reasonable approximations within
# the reported region.
# ----------------------------------------------------------------------------

SITES = [
    # FLUXNET-CH4 v1.0 (Delwiche 2021)
    # Arctic / boreal wet tundra
    ("US-Ivo",  71.28, -156.61, "wet_tundra",  "arctic",  "ec"),
    ("US-Atq",  70.47, -157.41, "wet_tundra",  "arctic",  "ec"),
    ("US-Beo",  71.28, -156.60, "wet_tundra",  "arctic",  "ec"),
    ("US-NGB",  71.28, -156.60, "wet_tundra",  "arctic",  "ec"),
    ("RU-Cok",  70.83, 147.49,  "wet_tundra",  "arctic",  "ec"),
    ("SE-St1",  68.35, 19.05,   "bog",         "arctic",  "ec"),  # Stordalen mire
    ("US-BZS",  64.70, -148.32, "bog",         "boreal",  "ec"),
    ("US-ICs",  68.93, -150.27, "wet_tundra",  "arctic",  "ec"),
    ("US-Uaf",  64.87, -147.86, "upland",      "boreal",  "ec"),
    # Boreal peatlands
    ("FI-Sii",  61.83, 24.19,   "fen",         "boreal",  "ec"),
    ("FI-Lom",  67.99, 24.21,   "fen",         "boreal",  "ec"),
    ("SE-Deg",  64.18, 19.55,   "fen",         "boreal",  "ec"),
    ("SE-Sto",  68.35, 19.05,   "bog",         "boreal",  "ec"),
    ("CA-SCB",  54.49, -105.82, "bog",         "boreal",  "ec"),
    ("CA-SCC",  53.94, -104.66, "bog",         "boreal",  "ec"),
    ("RU-Che",  68.61, 161.34,  "fen",         "boreal",  "ec"),
    ("DE-SfN",  47.81, 11.33,   "fen",         "boreal",  "ec"),
    ("FI-Hyy",  61.85, 24.30,   "upland",      "boreal",  "ec"),
    # Temperate marshes
    ("US-Myb",  38.05, -121.77, "marsh",       "temperate","ec"),
    ("US-Tw1",  38.11, -121.65, "marsh",       "temperate","ec"),
    ("US-Tw4",  38.10, -121.64, "marsh",       "temperate","ec"),
    ("US-Tw5",  38.11, -121.65, "marsh",       "temperate","ec"),
    ("US-Srr",  38.20, -122.02, "brackish",    "temperate","ec"),
    ("US-ORv",  40.02, -83.02,  "marsh",       "temperate","ec"),
    ("US-WPT",  41.46, -82.99,  "marsh",       "temperate","ec"),
    ("NL-Hor",  52.02, 5.07,    "marsh",       "temperate","ec"),
    ("DE-Hte",  54.21, 12.18,   "marsh",       "temperate","ec"),
    ("UK-AMo",  55.79, -3.24,   "bog",         "temperate","ec"),
    # Swamps
    ("US-LA1",  29.50, -90.50,  "swamp",       "temperate","ec"),
    ("US-LA2",  29.10, -90.20,  "swamp",       "temperate","ec"),
    ("US-DPW",  33.46, -90.83,  "swamp",       "temperate","ec"),
    ("US-Sne",  38.04, -121.75, "marsh",       "temperate","ec"),
    ("US-EDN",  38.10, -121.90, "brackish",    "temperate","ec"),
    ("US-StJ",  29.10, -90.20,  "brackish",    "temperate","ec"),
    ("US-Tbr",  38.10, -122.00, "brackish",    "temperate","ec"),
    # Rice paddies
    ("US-Bi1",  38.10, -121.50, "rice",        "temperate","ec"),
    ("US-Bi2",  38.10, -121.50, "rice",        "temperate","ec"),
    ("US-Twt",  38.11, -121.65, "rice",        "temperate","ec"),
    ("IT-Cas",  45.07, 8.72,    "rice",        "temperate","ec"),
    ("PH-RiF",  14.14, 121.27,  "rice",        "tropical", "ec"),
    ("JP-Mse",  36.05, 140.03,  "rice",        "temperate","ec"),
    ("JP-BBY",  43.32, 141.81,  "rice",        "temperate","ec"),
    # Tropical
    ("BR-Npw",  -16.50,-56.50,  "marsh",       "tropical", "ec"),
    ("BW-Gum",  -18.95, 22.45,  "marsh",       "tropical", "ec"),
    ("BW-Nxr",  -19.56, 23.17,  "marsh",       "tropical", "ec"),
    ("MY-MLM",   2.10, 112.28,  "swamp",       "tropical", "ec"),
    # Drained wetlands (7 sites)
    ("DE-Akm",  53.87, 13.68,   "drained",     "temperate","ec"),
    ("US-ORv2", 40.02, -83.02,  "drained",     "temperate","ec"),
    ("US-Wkg",  31.74, -109.94, "drained",     "temperate","ec"),
    ("DE-Spw",  51.89, 14.03,   "drained",     "temperate","ec"),
    ("CZ-wet",  49.02, 14.77,   "drained",     "temperate","ec"),
    ("US-Los",  46.08, -89.98,  "drained",     "temperate","ec"),
    ("US-Cem",  40.50, -87.00,  "drained",     "temperate","ec"),
    # Upland candidates (15 sites)
    ("US-Ho1",  45.20, -68.74,  "upland",      "boreal",   "ec"),
    ("US-Me2",  44.45, -121.56, "upland",      "temperate","ec"),
    ("US-Vcp",  35.86, -106.60, "upland",      "temperate","ec"),
    ("US-Wrc",  45.82, -121.95, "upland",      "temperate","ec"),
    ("CA-Obs",  53.99, -105.12, "upland",      "boreal",   "ec"),
    ("CA-Oas",  53.63, -106.20, "upland",      "boreal",   "ec"),
    ("DE-Tha",  50.96, 13.57,   "upland",      "temperate","ec"),
    ("FI-Kaa",  67.35, 26.34,   "upland",      "boreal",   "ec"),
    ("IT-Ren",  46.59, 11.43,   "upland",      "temperate","ec"),
    ("SE-Nor",  60.09, 17.48,   "upland",      "boreal",   "ec"),
    ("AU-Rob",  -17.12, 145.63, "upland",      "tropical", "ec"),
    ("AU-Wac",  -37.43, 145.19, "upland",      "temperate","ec"),
    ("CH-Cha",  47.21, 8.41,    "upland",      "temperate","ec"),
    ("CN-Din",  23.17, 112.54,  "upland",      "tropical", "ec"),
    ("NL-Loo",  52.17, 5.74,    "upland",      "temperate","ec"),
    # BAWLD-CH4 chamber plots (sampled representative set)
    ("BW-B01",  62.30, -114.50, "bog",         "boreal",   "chamber"),
    ("BW-B02",  67.42, 149.50,  "permafrost_bog","boreal", "chamber"),
    ("BW-T01",  68.80, -148.90, "wet_tundra",  "arctic",   "chamber"),
    ("BW-T02",  71.20, -156.40, "wet_tundra",  "arctic",   "chamber"),
    ("BW-F01",  64.20, 19.60,   "fen",         "boreal",   "chamber"),
    ("BW-M01",  60.90, 10.50,   "marsh",       "boreal",   "chamber"),
    # Extra short-record / unique
    ("US-CRT",  41.63, -83.35,  "marsh",       "temperate","ec"),
    ("FR-Pue",  43.74, 3.60,    "upland",      "temperate","ec"),
    ("AU-Fog",  -12.55, 131.31, "marsh",       "tropical", "ec"),
    ("IT-Ro1",  42.41, 11.93,   "upland",      "temperate","ec"),
    ("IL-Yat",  31.35, 35.05,   "upland",      "temperate","ec"),
    ("DK-Gls",  56.07, 9.33,    "marsh",       "temperate","ec"),
]

SITE_COLS = ["SITE_ID", "lat", "lon", "SITE_CLASSIFICATION", "climate_zone", "source_type"]

# Class -> (base flux nmol/m²/s, GPP multiplier)
CLASS_BASE = {
    "fen":             (120.0, 1.2),
    "bog":             (50.0,  1.0),
    "marsh":           (80.0,  1.3),
    "swamp":           (60.0,  1.1),
    "wet_tundra":      (70.0,  0.9),
    "rice":            (200.0, 1.5),
    "brackish":        (15.0,  1.0),
    "upland":          (2.5,   0.5),
    "drained":         (8.0,   0.8),
    "permafrost_bog":  (40.0,  0.9),
}


# ----------------------------------------------------------------------------
# Synthesis
# ----------------------------------------------------------------------------

def make_site_meta() -> pd.DataFrame:
    df = pd.DataFrame(SITES, columns=SITE_COLS)
    # Add IGBP-like, country, etc.
    df["IGBP"] = df["SITE_CLASSIFICATION"].map({
        "fen": "WET", "bog": "WET", "marsh": "WET", "swamp": "WET",
        "wet_tundra": "WET", "rice": "CRO", "brackish": "WET",
        "upland": "ENF", "drained": "GRA", "permafrost_bog": "WET",
    })
    df["country"] = df["SITE_ID"].str[:2]
    # Annual years of record: synthesized
    rng = np.random.default_rng(11)
    df["ANN_YEARS"] = rng.integers(1, 9, size=len(df))
    return df


def simulate_site_daily(site: pd.Series, years: list[int], rng: np.random.Generator
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (env_drivers_df, flux_df) for one site, daily resolution."""
    dates = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    doy = dates.dayofyear.values
    lat = site["lat"]
    # Annual air temperature: latitude-dependent amplitude + baseline
    if lat > 0:
        amp = 18 if abs(lat) > 40 else 8
        phase = 172   # peak around day-of-year 172 (late June, NH)
        tmean = 10 + (25 - abs(lat) * 0.4)
    else:
        amp = 10
        phase = 355
        tmean = 20 + (abs(lat) * 0.1)
    T_air = tmean + amp * np.cos(2 * np.pi * (doy - phase) / 365.0) + rng.normal(0, 2, size=len(doy))
    T_soil_5cm = T_air * 0.8 + 1.0 + rng.normal(0, 1, size=len(doy))
    T_soil_10cm = T_air * 0.7 + 1.5 + rng.normal(0, 0.8, size=len(doy))

    # Water table depth (cm below surface); negative = above surface (flooded)
    cls = site["SITE_CLASSIFICATION"]
    if cls in ("marsh", "rice", "fen", "bog", "wet_tundra", "permafrost_bog"):
        wtd_mean = -5.0
        wtd_sd = 4.0
    elif cls in ("swamp", "brackish"):
        wtd_mean = 0.0
        wtd_sd = 6.0
    elif cls == "drained":
        wtd_mean = 35.0
        wtd_sd = 10.0
    else:  # upland
        wtd_mean = 80.0
        wtd_sd = 15.0
    WTD = wtd_mean + amp * 0.15 * np.cos(2 * np.pi * (doy - (phase + 30)) / 365.0) \
          + rng.normal(0, wtd_sd * 0.2, size=len(doy))

    # GPP (gC/m²/d) proportional to air temp > 0 and daylight fraction
    daylen = 12 + 6 * np.sin(np.radians(lat)) * np.sin(2 * np.pi * (doy - 80) / 365.0)
    gpp = np.maximum(0, (T_air - 2) * 0.4 * (daylen / 12.0)) + rng.normal(0, 0.5, size=len(doy))
    gpp = np.maximum(0, gpp)

    # Precip (mm/d) bursty
    base = 2.0 if cls in ("marsh", "swamp", "fen") else 1.0
    precip = rng.gamma(0.5, base * 3.0, size=len(doy))

    # NDVI 0-1, seasonally varying
    ndvi = 0.3 + 0.4 * np.maximum(0, np.cos(2 * np.pi * (doy - phase) / 365.0))
    ndvi = np.clip(ndvi + rng.normal(0, 0.03, size=len(doy)), 0.05, 0.95)

    # Shortwave radiation (W/m² daily mean)
    srad = 150 + 120 * np.cos(2 * np.pi * (doy - phase) / 365.0) + rng.normal(0, 30, size=len(doy))
    srad = np.maximum(5, srad)

    env = pd.DataFrame({
        "site_id": site["SITE_ID"],
        "date": dates,
        "air_temp_c": T_air,
        "soil_temp_5cm_c": T_soil_5cm,
        "soil_temp_10cm_c": T_soil_10cm,
        "water_table_depth_cm": WTD,
        "gpp_gc_m2_d": gpp,
        "precip_mm": precip,
        "ndvi": ndvi,
        "srad_wm2": srad,
    })

    # CH4 flux: nmol / m² / s
    base_flux, gpp_mult = CLASS_BASE.get(cls, (10.0, 1.0))
    q10 = 3.0 ** ((T_soil_10cm - 10.0) / 10.0)
    wtd_factor = np.exp(-np.abs(WTD) * 0.04)
    gpp_factor = 1.0 + gpp_mult * (gpp / (gpp.mean() + 1e-6) - 1.0) * 0.5
    flux = base_flux * q10 * wtd_factor * gpp_factor
    # Light tailed noise
    flux = flux * np.exp(rng.normal(0, 0.25, size=len(doy)))
    # Occasional negative values (upland sinks)
    if cls == "upland":
        flux = -np.abs(flux) * 0.3
    flux = np.clip(flux, -50, 2000)

    flx = pd.DataFrame({
        "site_id": site["SITE_ID"],
        "date": dates,
        "ch4_flux_nmol_m2_s": flux,
        "quality_flag": rng.choice(["A", "B", "C"], p=[0.7, 0.2, 0.1], size=len(doy)),
    })
    return env, flx


def main() -> int:
    meta = make_site_meta()
    meta.to_csv(OUT / "site_metadata.csv", index=False)
    for alias in ("xmethanewet_site_metadata.csv", "xmethanewet_metadata.csv",
                  "xmethanewet_site_info.csv", "xmethanewet_existing_sites.csv"):
        meta.to_csv(OUT / alias, index=False)
    print(f"wrote site_metadata ({len(meta)} sites)")

    # Decide per-site year range based on ANN_YEARS (cap at 2016-2022 range)
    rng = np.random.default_rng(42)
    base_year = 2012
    env_parts, flx_parts = [], []
    temporal_split_rows = []
    for _, site in meta.iterrows():
        n_yr = int(site["ANN_YEARS"])
        years = list(range(base_year, base_year + n_yr + 2))
        env, flx = simulate_site_daily(site, years, rng)
        env_parts.append(env)
        flx_parts.append(flx)
        # Temporal split: last 2 years = test, rest train
        test_years = years[-2:]
        train_years = years[:-2]
        temporal_split_rows.append({
            "site_id": site["SITE_ID"],
            "train_years": ",".join(map(str, train_years)),
            "test_years": ",".join(map(str, test_years)),
        })
    env_all = pd.concat(env_parts, ignore_index=True)
    flx_all = pd.concat(flx_parts, ignore_index=True)
    print(f"env rows: {len(env_all):,}, flux rows: {len(flx_all):,}")

    # Write primary files + many aliases used in tasks
    env_all.to_csv(OUT / "xmethanewet_environmental_drivers.csv", index=False)
    for a in ("xmethanewet_daily_drivers.csv", "xmethanewet_drivers.csv",
              "xmethanewet_all_drivers.csv", "environmental_drivers.csv"):
        env_all.to_csv(OUT / a, index=False)

    flx_all.to_csv(OUT / "xmethanewet_flux_timeseries.csv", index=False)
    for a in ("xmethanewet_daily_flux.csv", "xmethanewet_tower_fluxes.csv",
              "methane_flux_daily.csv", "tower_observations.csv"):
        flx_all.to_csv(OUT / a, index=False)

    # Tower data = flux + drivers joined
    tower = flx_all.merge(env_all, on=["site_id", "date"], how="left")
    tower.to_csv(OUT / "xmethanewet_tower_data.csv", index=False)

    # By climate zone
    cz_map = dict(zip(meta["SITE_ID"], meta["climate_zone"]))
    tower["climate_zone"] = tower["site_id"].map(cz_map)
    for z in ("boreal", "temperate", "tropical"):
        sub = tower[tower["climate_zone"] == z].drop(columns=["climate_zone"])
        sub.to_csv(OUT / f"xmethanewet_{z}.csv", index=False)

    # EC vs chamber
    ec_ids = set(meta[meta["source_type"] == "ec"]["SITE_ID"])
    ch_ids = set(meta[meta["source_type"] == "chamber"]["SITE_ID"])
    tower[tower["site_id"].isin(ec_ids)].to_csv(OUT / "xmethanewet_ec_sites.csv", index=False)
    tower[tower["site_id"].isin(ch_ids)].to_csv(OUT / "xmethanewet_chamber_sites.csv", index=False)

    # 20-site subset daily flux
    twenty = sorted(meta["SITE_ID"])[:20]
    flx_all[flx_all["site_id"].isin(twenty)].to_csv(OUT / "xmethanewet_daily_20sites.csv", index=False)

    # Temporal split
    pd.DataFrame(temporal_split_rows).to_csv(OUT / "xmethanewet_temporal_split.csv", index=False)

    # Half-hourly subset (first 10 sites × 2 years)
    hh_sites = sorted(meta["SITE_ID"])[:10]
    hh_parts_f, hh_parts_m = [], []
    for sid in hh_sites:
        sub_f = flx_all[(flx_all["site_id"] == sid) & (flx_all["date"].dt.year.isin([2015, 2016]))]
        sub_e = env_all[(env_all["site_id"] == sid) & (env_all["date"].dt.year.isin([2015, 2016]))]
        for _, row in sub_f.iterrows():
            for h in range(48):
                hh_parts_f.append({
                    "site_id": sid,
                    "datetime": row["date"] + pd.Timedelta(minutes=30 * h),
                    "ch4_flux_nmol_m2_s": row["ch4_flux_nmol_m2_s"] * (0.8 + 0.4 * np.random.rand()),
                })
        for _, row in sub_e.iterrows():
            for h in range(48):
                hh_parts_m.append({
                    "site_id": sid,
                    "datetime": row["date"] + pd.Timedelta(minutes=30 * h),
                    "air_temp_c": row["air_temp_c"] + (np.sin(2*np.pi*h/48) * 4),
                    "soil_temp_5cm_c": row["soil_temp_5cm_c"],
                    "water_table_depth_cm": row["water_table_depth_cm"],
                })
    pd.DataFrame(hh_parts_f).to_csv(OUT / "xmethanewet_flux_halfhourly.csv", index=False)
    pd.DataFrame(hh_parts_f).to_csv(OUT / "xmethanewet_halfhourly_flux.csv", index=False)
    pd.DataFrame(hh_parts_m).to_csv(OUT / "xmethanewet_halfhourly_met.csv", index=False)
    pd.DataFrame(hh_parts_m).to_csv(OUT / "xmethanewet_drivers_halfhourly.csv", index=False)

    # Fluxnet QC flags synthetic
    qc = flx_all[["site_id", "date", "quality_flag"]].copy()
    qc["qc_fraction"] = (qc["quality_flag"] == "A").astype(float)
    qc.to_csv(OUT / "fluxnet_qc_flags.csv", index=False)

    # Monthly aggregates
    mn = flx_all.assign(month=flx_all["date"].dt.to_period("M").astype(str)).groupby(["site_id", "month"]).agg(
        ch4_flux_mean=("ch4_flux_nmol_m2_s", "mean"),
        ch4_flux_sum=("ch4_flux_nmol_m2_s", "sum"),
    ).reset_index()
    mn.to_csv(OUT / "monthly_ch4_totals.csv", index=False)
    mn.rename(columns={"ch4_flux_mean": "ch4_flux_obs"}).to_csv(OUT / "ch4_monthly_observed.csv", index=False)
    mn["ch4_flux_pred"] = mn["ch4_flux_mean"] * (1 + np.random.default_rng(0).normal(0, 0.1, len(mn)))
    mn[["site_id", "month", "ch4_flux_pred"]].to_csv(OUT / "ch4_monthly_predicted.csv", index=False)

    # ML predictions panel + per-site R²
    pred = flx_all.copy()
    noise = np.random.default_rng(5).normal(0, 0.2, size=len(pred))
    pred["ch4_flux_pred"] = pred["ch4_flux_nmol_m2_s"] * (1 + noise)
    pred.to_csv(OUT / "ml_ch4_predictions.csv", index=False)

    # Per-site performance
    perf_rows = []
    for sid, g in pred.groupby("site_id"):
        obs = g["ch4_flux_nmol_m2_s"].values
        prd = g["ch4_flux_pred"].values
        if np.var(obs) < 1e-9:
            r2 = np.nan
        else:
            r2 = 1 - np.sum((obs - prd)**2) / np.sum((obs - obs.mean())**2)
        rmse = np.sqrt(np.mean((obs - prd)**2))
        perf_rows.append({"site_id": sid, "R2": float(r2), "RMSE": float(rmse)})
    pd.DataFrame(perf_rows).to_csv(OUT / "xmethanewet_site_performance.csv", index=False)

    # Synthetic ERA5 reforecast at site locations: short (2015-2016) daily
    era5 = env_all.rename(columns={
        "air_temp_c": "t2m", "precip_mm": "tp", "srad_wm2": "ssrd",
    })[["site_id", "date", "t2m", "tp", "ssrd"]]
    era5[era5["date"].dt.year.isin([2015, 2016])].to_csv(
        OUT / "era5_reforecast_sitelocations.csv", index=False)

    # Candidate new-site locations for task optimization problems
    cand_lat = np.random.default_rng(0).uniform(-20, 75, 200)
    cand_lon = np.random.default_rng(1).uniform(-170, 170, 200)
    cand = pd.DataFrame({
        "candidate_id": [f"C{i:03d}" for i in range(200)],
        "lat": cand_lat, "lon": cand_lon,
        "wetland_class": np.random.default_rng(2).choice(list(CLASS_BASE), 200),
        "est_install_cost_usd": np.random.default_rng(3).uniform(50_000, 250_000, 200),
        "est_annual_ops_usd": np.random.default_rng(4).uniform(10_000, 60_000, 200),
    })
    cand.to_csv(OUT / "candidate_locations.csv", index=False)
    # Cost model JSON
    (OUT / "cost_model.json").write_text(json.dumps({
        "install_base_usd": 80000, "install_per_km_remoteness": 2000,
        "ops_base_usd_yr": 25000, "data_value_per_site_yr_usd": 150000,
        "discount_rate": 0.05, "horizon_years": 10
    }, indent=2))

    # Synthetic gridded products (small netCDFs, 0.5° & 0.25°)
    import xarray as xr
    lat = np.arange(-60, 82.5, 2.5)   # 4° grid to keep file small
    lon = np.arange(-180, 182.5, 2.5)
    months = pd.date_range("2005-01", "2020-12", freq="MS")
    rng2 = np.random.default_rng(7)
    data = rng2.uniform(0, 100, size=(len(months), len(lat), len(lon))).astype("float32")
    ds = xr.Dataset({"ch4_flux_nmol_m2_s": (("time", "lat", "lon"), data)},
                    coords={"time": months, "lat": lat, "lon": lon})
    ds.to_netcdf(OUT / "tem_global_ch4.nc")
    ds.to_netcdf(OUT / "gridded_ch4_products.nc")
    ds.to_netcdf(OUT / "process_model_outputs.nc")
    ds2 = ds.rename({"ch4_flux_nmol_m2_s": "wetland_fraction"})
    ds2.to_netcdf(OUT / "wetland_fraction_wad2m.nc")
    # Covariate grids (env drivers in xarray)
    cov = xr.Dataset({
        "soil_temp": (("time", "lat", "lon"), rng2.uniform(-20, 30, data.shape).astype("float32")),
        "water_table_depth": (("time", "lat", "lon"), rng2.uniform(-30, 80, data.shape).astype("float32")),
        "gpp": (("time", "lat", "lon"), rng2.uniform(0, 15, data.shape).astype("float32")),
    }, coords={"time": months, "lat": lat, "lon": lon})
    cov.to_netcdf(OUT / "covariate_grids.nc")
    cov.to_netcdf(OUT / "gridded_covariates_05deg.nc")

    # WetCHARTs-style CSV (global monthly per grid cell): just a stub flat CSV
    wc = ds.to_dataframe().reset_index().rename(columns={"ch4_flux_nmol_m2_s": "wetcharts_ch4"})
    wc.sample(50000, random_state=0).to_csv(OUT / "wetcharts_output.csv", index=False)

    # HLS-like synthetic satellite patches (32x32x6 per site × 20 dates)
    patch_dict = {}
    for sid in meta["SITE_ID"][:20]:
        patch_dict[sid] = rng2.integers(0, 10000, size=(20, 32, 32, 6)).astype("uint16")
    np.savez_compressed(OUT / "hls_patches_per_site.npz", **patch_dict)

    # Train a small RF as "teacher"
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    merged = flx_all.merge(env_all, on=["site_id", "date"], how="inner")
    feats = ["air_temp_c", "soil_temp_5cm_c", "soil_temp_10cm_c",
             "water_table_depth_cm", "gpp_gc_m2_d", "precip_mm", "ndvi", "srad_wm2"]
    X = merged[feats].astype(float).fillna(0)
    y = merged["ch4_flux_nmol_m2_s"].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    rf = RandomForestRegressor(n_estimators=60, max_depth=10, random_state=0, n_jobs=-1)
    rf.fit(Xtr, ytr)
    with open(OUT / "teacher_model.pt", "wb") as fh:
        pickle.dump({"model": rf, "feature_cols": feats, "target": "ch4_flux_nmol_m2_s"}, fh)
    with open(OUT / "trained_ch4_model.pkl", "wb") as fh:
        pickle.dump({"model": rf, "feature_cols": feats, "target": "ch4_flux_nmol_m2_s"}, fh)
    print(f"RF R²: {rf.score(Xte, yte):.3f}")

    print(f"\nAll MethaneWet synthesis files under {OUT}")
    print(f"Total files: {len(list(OUT.iterdir()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
