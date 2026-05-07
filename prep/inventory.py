"""
Canonical inventory of raw data sources, scope resolution rules, and synthesis
targets for ST_Bench. Consumed by the download, scope-extraction, and synthesis
scripts in this directory.

Anchored on Scope Sources (1).pdf (Apr 2026). Where the scope library
(data_tasks/mas_bench_scope_library.json) is authoritative on scope membership
criteria, this file maps those criteria to concrete raw-data columns / ID lists.
"""

# -----------------------------------------------------------------------------
# RAW DATASETS — the 3–4 sources everything else is derived from
# -----------------------------------------------------------------------------

RAW_DATASETS = {
    "camels_us": {
        "domain": "CAMELS",
        "description": "CAMELS-US 671 HCDN-2009 near-natural gauges (Newman 2015, Addor 2017). Superset of the 531 benchmark.",
        "urls": {
            # Streamflow + Daymet/Maurer/NLDAS forcing bundle
            "basin_timeseries_v1p2": "https://gdex.ucar.edu/dataset/camels/file/basin_timeseries_v1p2_metForcing_obsFlow.zip",
            # Attribute files (camels_clim.txt, camels_hydro.txt, etc.)
            "camels_attributes_v2p0": "https://gdex.ucar.edu/dataset/camels/file/camels_attributes_v2.0.zip",
            # Basin list + shapefiles
            "camels_topo_meta": "https://gdex.ucar.edu/dataset/camels/file/camels_topo.txt",
        },
        "attribute_files": [
            "camels_clim.txt",   # aridity, frac_snow, p_seasonality, etc.
            "camels_hydro.txt",  # baseflow_index, runoff_ratio, zero_q_freq
            "camels_topo.txt",   # area_gages2, elev_mean, slope_mean, huc_02
            "camels_soil.txt",
            "camels_vege.txt",   # frac_forest
            "camels_geol.txt",
        ],
        "size_gb_approx": 12.0,
    },
    "caravan_v1": {
        "domain": "CAMELS",
        "description": "Caravan v1 global 6,830 basins with ERA5-Land forcing + HydroATLAS attrs (Kratzert 2023).",
        "urls": {
            "caravan_zenodo": "https://zenodo.org/records/7540792",  # direct files listed in the record
        },
        "size_gb_approx": 5.0,
        "optional": True,
    },
    "cropbench_raw": {
        "domain": "CropBench",
        "description": "County-level Corn Belt yield + satellite bundle. Anchored on Khaki 13-state (763 counties).",
        "urls": {
            # USDA NASS Quick Stats county-level corn/soy yield (requires free API key)
            "nass_quickstats_api": "https://quickstats.nass.usda.gov/api",
            # HuggingFace packaged dataset matching Khaki 13-state + MODIS features
            "hf_usa_corn_belt": "https://huggingface.co/datasets/notadib/usa-corn-belt-crop-yield",
            # USDA Cropland Data Layer (annual national tif)
            "usda_cdl": "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/",
            # County boundaries (TIGER)
            "tiger_counties": "https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip",
        },
        "size_gb_approx": 8.0,
    },
    "fluxnet_ch4_v1": {
        "domain": "MethaneWet",
        "description": "FLUXNET-CH4 v1.0 eddy-covariance flux tower data (Delwiche 2021, 81 sites, 79 CC-BY).",
        "urls": {
            # AmeriFlux / FLUXNET portal (requires free registration)
            "ameriflux_ch4": "https://ameriflux.lbl.gov/data/download-data/",
            # Delwiche 2021 ESSD supplement (Table 2 site list)
            "delwiche_supp": "https://essd.copernicus.org/articles/13/3607/2021/essd-13-3607-2021-supplement.zip",
            # McNicol 2023 UpCH4 supp03.xlsx with training site list
            "mcnicol_supp03": "https://agupubs.onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1029%2F2023AV000956&file=2023AV000956-sup-0003-Supplementary+Material.xlsx",
            # BAWLD-CH4 chamber compilation (Kuhn 2021)
            "bawld_ch4": "https://arcticdata.io/catalog/view/doi:10.18739/A2DN3ZX1W",
        },
        "size_gb_approx": 3.0,
    },
    "era5_subset": {
        "domain": "General",
        "description": "ERA5 reanalysis subset for General-domain tasks. Use 1° regridded 5-var WeatherBench-style slice rather than 0.25° global to keep storage tractable.",
        "urls": {
            # WeatherBench 2 pre-regridded ERA5 (Google Cloud, anonymous)
            "weatherbench2_1deg": "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-1deg.zarr",
            # Copernicus CDS (if original 0.25° is needed; requires API key)
            "cds_api": "https://cds.climate.copernicus.eu/api",
        },
        "variables": ["2m_temperature", "total_precipitation", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure"],
        "size_gb_approx": 40.0,  # 1° 6h 5-var ~1990-2023
    },
}


# -----------------------------------------------------------------------------
# SCOPE RESOLUTION — how to turn each scope_id into a concrete ID list
# Encoded as: filter spec on the raw attribute/metadata tables
# -----------------------------------------------------------------------------
# Encoding:
#   {"base": "<raw list to start from>", "filter": "<pandas query string>", "columns": [...], "limit": <int|None>}
# Only CAMELS filled here first; CropBench/MethaneWet follow the same pattern.

CAMELS_SCOPES = {
    # Base lists
    "camels_us_531":      {"base": "basin_list_531.txt"},
    "camels_us_671":      {"base": "camels_name.txt"},
    "camels_us_516_chem": {"base": "camels_chem_intersect_516.txt"},

    # Addor 2017 classes via camels_clim.txt
    "snow_dominated":        {"base": "531", "filter": "frac_snow > 0.3",           "table": "camels_clim.txt"},
    "snow_dominated_strict": {"base": "531", "filter": "frac_snow > 0.5",           "table": "camels_clim.txt"},
    "arid":                  {"base": "531", "filter": "pet_mean / p_mean > 1",     "table": "camels_clim.txt"},
    "very_arid":             {"base": "531", "filter": "pet_mean / p_mean > 2",     "table": "camels_clim.txt"},
    "humid":                 {"base": "531", "filter": "pet_mean / p_mean < 0.75",  "table": "camels_clim.txt"},
    "humid_no_snow":         {"base": "531", "filter": "(pet_mean / p_mean < 1) and (frac_snow < 0.15)", "table": "camels_clim.txt"},
    "seasonal_summer_p":     {"base": "531", "filter": "p_seasonality > 0",         "table": "camels_clim.txt"},
    "seasonal_winter_p":     {"base": "531", "filter": "p_seasonality < 0",         "table": "camels_clim.txt"},

    # camels_hydro.txt
    "high_baseflow":     {"base": "531", "filter": "baseflow_index > 0.5", "table": "camels_hydro.txt"},
    "low_baseflow":      {"base": "531", "filter": "baseflow_index < 0.3", "table": "camels_hydro.txt"},
    "intermittent":      {"base": "531", "filter": "zero_q_freq > 0",      "table": "camels_hydro.txt"},
    "high_runoff_ratio": {"base": "531", "filter": "runoff_ratio > 0.5",   "table": "camels_hydro.txt"},
    "low_runoff_ratio":  {"base": "531", "filter": "runoff_ratio < 0.3",   "table": "camels_hydro.txt"},

    # camels_topo.txt
    "small_basins":   {"base": "531", "filter": "area_gages2 < 250",                       "table": "camels_topo.txt"},
    "medium_basins":  {"base": "531", "filter": "(area_gages2 >= 250) and (area_gages2 < 1000)", "table": "camels_topo.txt"},
    "large_basins":   {"base": "531", "filter": "(area_gages2 >= 1000) and (area_gages2 < 2000)", "table": "camels_topo.txt"},
    "high_elevation": {"base": "531", "filter": "elev_mean > 1500",                         "table": "camels_topo.txt"},
    "low_elevation":  {"base": "531", "filter": "elev_mean < 500",                          "table": "camels_topo.txt"},
    "steep_slopes":   {"base": "531", "filter": "slope_mean > 0.1",                         "table": "camels_topo.txt"},

    # camels_vege.txt
    "forested": {"base": "531", "filter": "frac_forest > 0.5", "table": "camels_vege.txt"},

    # HUC-2 regions (camels_topo.txt huc_02)
    "huc2_pacific_nw":     {"base": "531", "filter": "huc_02 == 17",                   "table": "camels_topo.txt"},
    "huc2_california":     {"base": "531", "filter": "huc_02 == 18",                   "table": "camels_topo.txt"},
    "huc2_upper_colorado": {"base": "531", "filter": "huc_02 == 14",                   "table": "camels_topo.txt"},
    "huc2_midwest":        {"base": "531", "filter": "huc_02 in [5,6,7,8,10,11]",      "table": "camels_topo.txt"},
    "huc2_northeast":      {"base": "531", "filter": "huc_02 in [1,2,3,4]",            "table": "camels_topo.txt"},
    "huc2_southeast":      {"base": "531", "filter": "huc_02 in [3,6,8]",              "table": "camels_topo.txt"},

    # Temporal splits — implemented as date ranges, not ID filters
    "kratzert_temporal": {"base": "531", "temporal": {"train": "1999-10-01/2008-09-30", "test":  "1989-10-01/1999-09-30"}},
    "newman_temporal":   {"base": "531", "temporal": {"train": "1980-10-01/1995-09-30", "test":  "1995-10-01/2014-09-30"}},

    # CV splits — stored as fold files
    "pub_12fold": {"base": "531", "folds_file": "ealstm_pub_12fold.json"},
    "huc2_loo":   {"base": "531", "folds_by": "huc_02", "table": "camels_topo.txt"},

    # Sibling datasets (separate raw bundles)
    "camels_gb":       {"base": "camels_gb_all"},
    "camels_cl":       {"base": "camels_cl_all"},
    "camels_br":       {"base": "camels_br_all"},
    "camels_aus":      {"base": "camels_aus_v1_all"},
    "camels_ch":       {"base": "camels_ch_all"},
    "camels_de":       {"base": "camels_de_all"},
    "caravan_v1":      {"base": "caravan_v1_all"},
    "caravan_us_only": {"base": "caravan_v1_all", "filter": "source == 'CAMELS'"},
    "hysets_subset":   {"base": "hysets_caravan_filter"},
    "lamah_ce":        {"base": "lamah_ce_all"},
}


# -----------------------------------------------------------------------------
# SYNTHESIS TARGETS — artifacts tasks reference that don't exist in any
# public source and must be produced by prep scripts.
# -----------------------------------------------------------------------------

SYNTHESIS_TARGETS = {
    # CAMELS domain — pretrained hydrology models and their cross-basin scores
    "teacher_lstm.pt":                   {"domain": "CAMELS",    "kind": "pretrained_model", "recipe": "train single global EA-LSTM on 531 basins, Kratzert 2019a config"},
    "trained_lstm.pt":                   {"domain": "CAMELS",    "kind": "pretrained_model", "recipe": "same checkpoint as teacher_lstm.pt; reuse"},
    "trained_rf.pkl":                    {"domain": "CAMELS",    "kind": "pretrained_model", "recipe": "RandomForest predicting daily q from forcing+attributes, 531 basins"},
    "camels_lstm_nse.csv":               {"domain": "CAMELS",    "kind": "per_basin_metric", "recipe": "per-basin test-period NSE of teacher_lstm.pt"},
    "camels_global_lstm_nse.csv":        {"domain": "CAMELS",    "kind": "per_basin_metric", "recipe": "per-basin test-period NSE of the single global EA-LSTM"},
    "camels_model_comparison_nse.csv":   {"domain": "CAMELS",    "kind": "per_basin_metric", "recipe": "per-basin NSE for {LSTM, SAC-SMA, VIC, HBV} — pull published SAC-SMA/VIC from Newman 2015; LSTM from our teacher"},
    "camels_model_comparison_kge.csv":   {"domain": "CAMELS",    "kind": "per_basin_metric", "recipe": "same as above but KGE"},

    # CropBench domain — pretrained yield models
    "teacher_cnnrnn.pt":                 {"domain": "CropBench", "kind": "pretrained_model", "recipe": "CNN-RNN from Khaki 2020 reproduction on 13-state Corn Belt"},
    "trained_tft.pt":                    {"domain": "CropBench", "kind": "pretrained_model", "recipe": "Temporal Fusion Transformer on CropNet features"},
    "cropbench_rich_model.pt":           {"domain": "CropBench", "kind": "pretrained_model", "recipe": "high-capacity transformer combining MODIS+weather+soil"},
    "prithvi_weights.pt":                {"domain": "CropBench", "kind": "downloaded_model", "recipe": "download from HuggingFace ibm-nasa-geospatial/Prithvi-100M"},
    "prithvi_crop_classification_variant.pt": {"domain": "CropBench", "kind": "downloaded_model", "recipe": "download from HuggingFace ibm-nasa-geospatial/Prithvi-100M-multi-temporal-crop-classification"},
    "cropbench_5model_predictions.csv":  {"domain": "CropBench", "kind": "prediction_matrix", "recipe": "county-year predictions from 5 model variants (RF, XGB, LSTM, CNN-RNN, TFT)"},
    "cropbench_county_rmse.csv":         {"domain": "CropBench", "kind": "per_county_metric", "recipe": "per-county RMSE of teacher_cnnrnn.pt"},
    "cropbench_sparse_labels.csv":       {"domain": "CropBench", "kind": "sampling_subset",   "recipe": "random 10% of county-year labels (seeded)"},
    "cropbench_sparse_region.csv":       {"domain": "CropBench", "kind": "sampling_subset",   "recipe": "rainfed_dominant region labels only"},
    "cropbench_sparse_satellite.csv":    {"domain": "CropBench", "kind": "sampling_subset",   "recipe": "satellite features for cropbench_sparse_labels rows only"},

    # MethaneWet domain
    "teacher_model.pt":                  {"domain": "MethaneWet", "kind": "pretrained_model", "recipe": "RF on fluxnet_ch4 UpCH4-43 sites predicting daily CH4"},
    "trained_ch4_model.pkl":             {"domain": "MethaneWet", "kind": "pretrained_model", "recipe": "same checkpoint as teacher_model.pt; reuse"},
    "ml_ch4_predictions.csv":            {"domain": "MethaneWet", "kind": "prediction_matrix", "recipe": "per-site-day CH4 predictions from teacher_model.pt"},
    "xmethanewet_site_performance.csv":  {"domain": "MethaneWet", "kind": "per_site_metric",  "recipe": "per-site R2/RMSE of teacher_model.pt"},

    # General domain — weather/precip forecasts
    "teacher_forecasts.nc":              {"domain": "General", "kind": "pretrained_forecast", "recipe": "WeatherBench climatology+persistence+GraphCast deterministic forecast slice"},
    "predictions_precip.nc":             {"domain": "General", "kind": "pretrained_forecast", "recipe": "per-gridcell precip forecasts from teacher_forecasts.nc"},
    "predictions_streamflow.csv":        {"domain": "General", "kind": "prediction_matrix",   "recipe": "single-model streamflow predictions (reuse camels teacher_lstm)"},
    "predictions_yields.csv":            {"domain": "General", "kind": "prediction_matrix",   "recipe": "single-model yield predictions (reuse cropbench teacher_cnnrnn)"},
    "per_sample_preds_lstm.csv":         {"domain": "General", "kind": "prediction_matrix",   "recipe": "per-sample LSTM predictions on held-out set"},
    "per_sample_preds_rf.csv":           {"domain": "General", "kind": "prediction_matrix",   "recipe": "per-sample RF predictions on held-out set"},
    "per_sample_preds_transformer.csv":  {"domain": "General", "kind": "prediction_matrix",   "recipe": "per-sample Transformer predictions on held-out set"},
    "camels_preds.csv":                  {"domain": "General", "kind": "prediction_matrix",   "recipe": "reuse CAMELS teacher_lstm per-basin test preds"},
    "cropbench_preds.csv":               {"domain": "General", "kind": "prediction_matrix",   "recipe": "reuse CropBench teacher_cnnrnn per-county test preds"},
    "methane_preds.csv":                 {"domain": "General", "kind": "prediction_matrix",   "recipe": "reuse MethaneWet teacher_model per-site test preds"},
}


# -----------------------------------------------------------------------------
# RECOMMENDED PREP ORDER
# -----------------------------------------------------------------------------
PREP_ORDER = [
    # 1. CAMELS: smallest blast radius, 25 tasks / 529 queries, all scope rules
    #    are pandas filters on Addor attribute files. Proves the pipeline.
    "camels_us",
    # 2. MethaneWet: FLUXNET-CH4 manageable scale (~GB), 40 tasks / 840 queries.
    "fluxnet_ch4_v1",
    # 3. CropBench: NASS + MODIS, 24 tasks / 507 queries. Larger satellite data.
    "cropbench_raw",
    # 4. ERA5 subset + General: 11 tasks / 191 queries; largest data, do last.
    "era5_subset",
    # 5. caravan_v1: only referenced by a handful of CAMELS scopes; optional.
    "caravan_v1",
]
