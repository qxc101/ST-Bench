"""
Download the HuggingFace-hosted Khaki 13-state Corn Belt crop-yield package.
This is the canonical CropBench scope anchor per the Scope Sources PDF.

Target: https://huggingface.co/datasets/notadib/usa-corn-belt-crop-yield
Size: ~1.5 GB across a few CSV / NPZ files.
"""
from __future__ import annotations
import sys
from pathlib import Path
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "raw_data" / "CropBench"
OUT.mkdir(parents=True, exist_ok=True)

FILES = [
    ("khaki_multi_crop_yield.csv",                             None),
    ("soybean_data_soilgrid250_modified_states_9_processed.csv", None),
    ("combined_dataset_weekly.npz",                            None),
    ("Soybeans_Loc_ID.csv",                                    None),
    ("test.csv",                                               None),
    ("weekly_weather_param_scalers.json",                      None),
    ("README.md",                                              None),
]

URL_BASE = "https://huggingface.co/datasets/notadib/usa-corn-belt-crop-yield/resolve/main"


def fetch(name: str) -> bool:
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ok (exists) {name}"); return True
    url = f"{URL_BASE}/{name}"
    r = requests.get(url, stream=True, timeout=120, headers={"User-Agent": "ST_Bench/0.1"})
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} {name}"); return False
    total = int(r.headers.get("Content-Length", 0))
    with open(dest, "wb") as fh, tqdm(total=total or None, unit="B", unit_scale=True,
                                       desc=name, leave=False) as bar:
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk: fh.write(chunk); bar.update(len(chunk))
    print(f"  ok {name} ({dest.stat().st_size/1e6:.1f} MB)")
    return True


if __name__ == "__main__":
    for name, _ in FILES:
        fetch(name)
