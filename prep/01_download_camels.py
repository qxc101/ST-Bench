"""
Download the CAMELS-US raw data needed for ST_Bench from Zenodo.

Source: https://zenodo.org/records/15529996 (UCAR-NCAR CAMELS migrated 2024-05)
DOI:    https://dx.doi.org/10.5065/D6MW2F4D

Essential (~3.45 GB):
  - basin_timeseries_v1p2_metForcing_obsFlow.zip  (3.4 GB)  forcing + obs flow
  - camels_clim.txt, camels_hydro.txt, camels_topo.txt,
    camels_soil.txt, camels_vege.txt, camels_geol.txt,
    camels_name.txt, readme.txt                    (loose attribute files)
  - basin_set_full_res.zip                        (45 MB)  basin shapefiles

Skipped (not referenced by any task in data_tasks/):
  - basin_timeseries_v1p2_modelOutput_{daymet,maurer,nldas}.zip  (~11 GB total)

Sibling CAMELS datasets (GB/CL/CH/BR/AUS/DE, Caravan, HYSETS, LamaH-CE) cover
only ~55 queries out of 529 and are listed in SIBLING_SOURCES for a later
second-pass download — they require per-portal manual steps.

Run:
    conda activate stbench
    python prep/01_download_camels.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw_data" / "CAMELS"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ZENODO_RECORD = "15529996"
ZENODO_BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"


def _zurl(name: str) -> str:
    return f"{ZENODO_BASE}/{name}/content"


# --- Files to download --------------------------------------------------------

LOOSE_FILES = [
    ("camels_clim.txt",     102_739),
    ("camels_hydro.txt",    125_471),
    ("camels_topo.txt",      41_842),
    ("camels_soil.txt",     116_294),
    ("camels_vege.txt",     113_398),
    ("camels_geol.txt",      73_614),
    ("camels_name.txt",      31_616),
    ("readme.txt",            4_000),
    ("camels_attributes_v2.0.pdf",  93_696),
    ("camels_attributes_v2.0.xlsx", 22_528),
]

ARCHIVES = {
    # archive name -> {size, expected_top_level_dir, extract_to}
    "basin_timeseries_v1p2_metForcing_obsFlow.zip": {
        "size_hint_bytes": 3_406_630_000,
        "expected_dir": "basin_dataset_public_v1p2",
    },
    "basin_set_full_res.zip": {
        "size_hint_bytes": 45_180_000,
        "expected_dir": "basin_set_full_res",
    },
}


# --- Sibling sources (not downloaded by default) ------------------------------

SIBLING_SOURCES = {
    "camels_gb":       ("CAMELS-GB 671 basins (Coxon 2020)",    "https://data-package.ceh.ac.uk/data/8344e4f3-d2ea-44f5-8afa-86d2987543a9"),
    "camels_cl":       ("CAMELS-CL 516 Chilean gauges",         "https://camels.cr2.cl/"),
    "camels_br":       ("CAMELS-BR 593 Brazilian catchments",   "https://zenodo.org/record/3709338"),
    "camels_aus":      ("CAMELS-AUS v1 222 Australian basins",  "https://doi.pangaea.de/10.1594/PANGAEA.921850"),
    "camels_ch":       ("CAMELS-CH 331 Alpine catchments",      "https://zenodo.org/record/7784632"),
    "camels_de":       ("CAMELS-DE 1,582 German basins",        "https://zenodo.org/record/13837553"),
    "caravan_v1":      ("Caravan v1 6,830 basins global",       "https://zenodo.org/record/7540792"),
    "hysets_subset":   ("HYSETS North America 14,425 basins",   "https://osf.io/rpc3w/"),
    "lamah_ce":        ("LamaH-CE 859 Central European",        "https://zenodo.org/record/5153305"),
}


# --- Download helpers ---------------------------------------------------------

USER_AGENT = "ST_Bench-prep/0.1"


def download(url: str, dest: Path, size_hint: int | None = None) -> bool:
    """Resumable HTTP GET with tqdm progress. Returns True on success."""
    headers = {"User-Agent": USER_AGENT}
    existing = dest.stat().st_size if dest.exists() else 0
    if existing:
        if size_hint and existing >= size_hint * 0.99:
            return True  # treat as complete
        headers["Range"] = f"bytes={existing}-"

    try:
        r = requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
    except requests.RequestException as e:
        print(f"    request failed: {e}")
        return False

    if r.status_code == 416:
        return True  # already complete
    if r.status_code not in (200, 206):
        print(f"    HTTP {r.status_code} for {url}")
        return False

    remaining = int(r.headers.get("Content-Length", 0))
    total = existing + remaining if r.status_code == 206 else remaining

    mode = "ab" if r.status_code == 206 else "wb"
    with open(dest, mode) as f, tqdm(
        total=total or None, initial=existing,
        unit="B", unit_scale=True, desc=dest.name, leave=False,
    ) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    return True


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for m in tqdm(members, desc=f"unzip {zip_path.name}", leave=False):
            zf.extract(m, out_dir)


# --- Main ---------------------------------------------------------------------

def main() -> int:
    rc = 0

    print("=== Loose attribute / metadata files ===")
    for name, size_hint in LOOSE_FILES:
        dest = RAW_DIR / name
        if dest.exists() and dest.stat().st_size >= size_hint * 0.9:
            print(f"  ok  {name} (already present)")
            continue
        ok = download(_zurl(name), dest, size_hint)
        print(f"  {'ok' if ok else 'FAIL'}  {name}")
        if not ok:
            rc = 1

    print("\n=== Archives ===")
    for name, spec in ARCHIVES.items():
        zip_path = RAW_DIR / name
        extracted_marker = RAW_DIR / spec["expected_dir"]

        if extracted_marker.exists():
            print(f"  already extracted: {spec['expected_dir']}  (skipping download)")
            continue

        if not zip_path.exists() or zip_path.stat().st_size < spec["size_hint_bytes"] * 0.9:
            print(f"  downloading {name}  (~{spec['size_hint_bytes']/1e9:.2f} GB)")
            if not download(_zurl(name), zip_path, spec["size_hint_bytes"]):
                print(f"  FAIL: could not download {name}")
                rc = 1
                continue

        print(f"  extracting {name}")
        try:
            extract_zip(zip_path, RAW_DIR)
        except zipfile.BadZipFile as e:
            print(f"  bad zip: {e} — remove {zip_path} and retry")
            rc = 1
            continue

        if not extracted_marker.exists():
            print(f"  WARN: expected dir {spec['expected_dir']} not found after extract")
            rc = 1
        else:
            print(f"  ok: {spec['expected_dir']}")

    print("\n=== Sibling datasets (manual fetch; skipped for now) ===")
    for sid, (desc, landing) in SIBLING_SOURCES.items():
        print(f"  {sid:16s} {desc}")
        print(f"  {'':16s}   -> {landing}")

    print(f"\nraw_data root: {RAW_DIR}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
