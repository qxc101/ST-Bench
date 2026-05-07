"""
Download a subset of CAMELS-sibling datasets from Zenodo that have direct
anonymous downloads and are referenced by ST_Bench queries.

Target coverage (out of ~55 sibling-scope queries):
  Caravan v1 global           --> caravan_v1, caravan_us_only, caravan_attributes
                                   camels_us, camels_de (as separate sibling)
  CAMELS-CH (via Caravan_extension_CH) --> camels_ch

Skipped (require registration or are low-ROI for our budget):
  CAMELS-GB (CEH/EIDC — registration), CAMELS-BR, CAMELS-AUS, CAMELS-CL,
  CAMELS-DE (a separate Zenodo record but requires licence click), HYSETS,
  LamaH-CE.

Run:
    conda activate stbench
    python prep/06_download_camels_siblings.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SIB_DIR = ROOT / "raw_data" / "CAMELS_siblings"
SIB_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS = [
    {
        "name": "Caravan.zip",
        "url": "https://zenodo.org/api/records/7540792/files/Caravan.zip/content",
        "size": 12_515_561_806,
        "extract_to": SIB_DIR / "caravan_v1",
    },
    {
        "name": "camels_ch.zip",
        "url": "https://zenodo.org/api/records/15025258/files/camels_ch.zip/content",
        "size": 259_400_000,
        "extract_to": SIB_DIR / "camels_ch",
    },
]


def download(url: str, dest: Path, size_hint: int | None = None) -> bool:
    headers = {"User-Agent": "ST_Bench-prep/0.1"}
    existing = dest.stat().st_size if dest.exists() else 0
    if existing and size_hint and existing >= size_hint * 0.99:
        print(f"  already complete: {dest.name}")
        return True
    if existing:
        headers["Range"] = f"bytes={existing}-"
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=120, allow_redirects=True)
    except requests.RequestException as e:
        print(f"  request failed: {e}")
        return False
    if r.status_code == 416:
        return True
    if r.status_code not in (200, 206):
        print(f"  HTTP {r.status_code}"); return False
    remaining = int(r.headers.get("Content-Length", 0))
    total = existing + remaining if r.status_code == 206 else remaining
    mode = "ab" if r.status_code == 206 else "wb"
    with open(dest, mode) as fh, tqdm(
        total=total or None, initial=existing,
        unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))
    return True


def main() -> int:
    for spec in DOWNLOADS:
        zip_path = SIB_DIR / spec["name"]
        print(f"\n=== {spec['name']} (~{spec['size']/1e9:.2f} GB) ===")
        if spec["extract_to"].exists() and any(spec["extract_to"].rglob("*")):
            print(f"  already extracted: {spec['extract_to']}"); continue

        ok = download(spec["url"], zip_path, spec["size"])
        if not ok:
            print(f"  !! download failed for {spec['name']}; skipping"); continue

        print(f"  extracting to {spec['extract_to']}")
        spec["extract_to"].mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                members = zf.namelist()
                for m in tqdm(members, desc=f"unzip {spec['name']}", leave=False):
                    zf.extract(m, spec["extract_to"])
        except zipfile.BadZipFile as e:
            print(f"  bad zip: {e}"); continue

        # Optional: remove the archive to save disk once extracted
        zip_path.unlink()
        print(f"  ok (archive removed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
