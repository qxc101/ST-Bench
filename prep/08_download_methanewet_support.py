"""
Download public MethaneWet supporting datasets: WAD2M wetland fraction grid and
Peltola 2019 random-forest upscaling product. AmeriFlux FLUXNET-CH4 sites
require registration, so we derive MethaneWet working data from a combination
of these public gridded products + a synthesized per-site panel produced by
prep/09_methanewet_synthesis.py.
"""
from __future__ import annotations
import sys, zipfile
from pathlib import Path
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "raw_data" / "MethaneWet"
OUT.mkdir(parents=True, exist_ok=True)

DOWNLOADS = [
    # WAD2M wetland fraction 0.25° monthly, 2000-2018
    ("WAD2M_wetlands_2000-2018_025deg.nc.zip",
     "https://zenodo.org/api/records/3998454/files/WAD2M_wetlands_2000-2018_025deg.nc.zip/content",
      261_000_000),
    # Peltola 2019 RF upscaling model
    ("CH4_RF.zip",
     "https://zenodo.org/api/records/3247295/files/CH4_RF.zip/content",
     15_300_000),
]


def fetch(name: str, url: str, size: int) -> bool:
    dest = OUT / name
    if dest.exists() and dest.stat().st_size >= size * 0.99:
        print(f"  ok (exists) {name}"); return True
    r = requests.get(url, stream=True, timeout=120, headers={"User-Agent": "ST_Bench/0.1"})
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} for {url}"); return False
    total = int(r.headers.get("Content-Length", 0))
    with open(dest, "wb") as fh, tqdm(total=total or None, unit="B", unit_scale=True,
                                       desc=name, leave=False) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk: fh.write(chunk); bar.update(len(chunk))
    print(f"  ok {name} ({dest.stat().st_size/1e6:.1f} MB)")
    if name.endswith(".zip"):
        try:
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(OUT)
            print(f"  extracted {name}")
        except zipfile.BadZipFile as e:
            print(f"  !! bad zip: {e}")
            return False
    return True


if __name__ == "__main__":
    for name, url, size in DOWNLOADS:
        fetch(name, url, size)
