"""
Stage per-query working directories under data/<task_id>/<query_id>/ containing:

    prompt.txt           the query's assembled prompt, augmented with a
                         "files available" section describing what is staged
    query.json           the full query record from mas_bench_queries_v4.jsonl
                         (includes metrics, steps, scope, split)
    <task-named files>   symlinks to the scope's parquet outputs, using the
                         task's input_data.files stems (with .parquet extension
                         since we store parquet, not CSV)
    results/             empty dir for the agent to write metrics.json + evidence

For CAMELS, each task file is mapped to one of:
    attributes.parquet, streamflow_daily.parquet, forcing_daymet.parquet
  based on substring match ("attribute", "streamflow"/"flow"/"q_", "forcing").

Synthesis-dependent files (teacher_lstm.pt, camels_lstm_nse.csv, trained_rf.pkl,
camels_model_comparison_*.csv, predictions_*.csv, flood_events_*.csv, nid.csv,
gagesii_*, era5_*, nlcd_*, climate_trends.csv, station_observations.csv,
single_basin_streamflow_with_gaps.csv, caravan_*.csv, camels_gb/cl/br/aus/ch/de_*)
are written as `MISSING__<name>.txt` placeholder notes until their synthesis
scripts (prep/05_*) fill them in. The stage log reports which queries are
therefore incomplete.

Run:
    conda activate stbench
    python prep/04_stage_queries.py [--domain CAMELS] [--task C1.1] [--limit 10]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "data_tasks" / "mas_bench_queries_v4.jsonl"
SCOPES_ROOT = ROOT / "scopes"
DATA_ROOT = ROOT / "data"


# Domains in the benchmark
DOMAIN_RAW_ROOTS = {
    "CAMELS":     ROOT / "raw_data" / "CAMELS",
    "CropBench":  ROOT / "raw_data" / "CropBench",
    "MethaneWet": ROOT / "raw_data" / "MethaneWet",
    "General":    ROOT / "raw_data" / "General",
}


def _pick_camels_source(task_file: str, scope_dir: Path, synth_dir: Path) -> Path | None:
    """Pick the best-fit source file for a CAMELS task-file. Tries, in order:
      1. Scope parquet (attributes / streamflow / forcing / panel)
      2. Synthesized artifact by exact filename under raw_data/CAMELS/synth/
    """
    name = task_file.lower()

    # Scope parquet dispatch
    if scope_dir.exists():
        if "attribute" in name:
            p = scope_dir / "attributes.parquet"
            if p.exists(): return p
        if "forcing" in name:
            p = scope_dir / "forcing_daymet.parquet"
            if p.exists(): return p
        if "streamflow" in name or "flow" in name:
            p = scope_dir / "streamflow_daily.parquet"
            if p.exists(): return p

    # Synthesized exact filename
    p = synth_dir / task_file
    if p.exists(): return p
    return None


def _pick_cropbench_source(task_file: str, scope_dir: Path, synth_dir: Path,
                           raw_dir: Path) -> Path | None:
    name = task_file.lower()
    # Scope parquets first
    if scope_dir.exists():
        if "yield" in name and "climate" not in name:
            p = scope_dir / "yields.parquet"
            if p.exists(): return p
        if "weather" in name or "climate" in name:
            p = scope_dir / "weather.parquet"
            if p.exists(): return p
        if "soil" in name:
            p = scope_dir / "soil.parquet"
            if p.exists(): return p
        if "ndvi" in name or "modis" in name or "satellite" in name or "sentinel" in name:
            p = scope_dir / "ndvi_biweekly.parquet"
            if p.exists(): return p
        if "coord" in name or "boundar" in name or "parcel" in name:
            p = scope_dir / "county_meta.parquet"
            if p.exists(): return p
        if "label" in name or task_file in ("cropbench_yields.csv", "county_yields.csv"):
            p = scope_dir / "yields.parquet"
            if p.exists(): return p

    # Synthesized exact filename
    p = synth_dir / task_file
    if p.exists(): return p

    # Loose in raw root (HF files)
    for base_alt in (task_file, task_file.replace(".csv", ""), task_file.replace(".csv", ".parquet")):
        p = raw_dir / base_alt
        if p.exists(): return p
    return None


def _pick_methanewet_source(task_file: str, synth_dir: Path, raw_dir: Path) -> Path | None:
    # MethaneWet is mostly synthesized directly under raw_data/MethaneWet/
    for candidate in (raw_dir / task_file, synth_dir / task_file):
        if candidate.exists():
            return candidate
    return None


def _pick_general_source(task_file: str, scope_dir: Path, synth_dir: Path,
                         raw_dir: Path) -> Path | None:
    # General: scope slice .nc first, then synth file, then raw
    if scope_dir.exists():
        slice_nc = scope_dir / "era5_slice.nc"
        if slice_nc.exists() and ("era5" in task_file.lower() or task_file.endswith(".nc")):
            return slice_nc
    for candidate in (synth_dir / task_file, raw_dir / task_file):
        if candidate.exists():
            return candidate
    return None


def _needs_synthesis(task_file: str, domain: str) -> bool:
    """Return True if this file can't be sourced from what we have for this domain."""
    # Everything is potentially resolvable now; we check existence via _pick_* helpers.
    return False


def _file_target_name(task_file: str, src: Path) -> str:
    """Keep the task's original filename stem, using the source file's extension."""
    stem = Path(task_file).stem
    ext = src.suffix
    return f"{stem}{ext}"


def stage_query(q: dict, overwrite: bool = False) -> dict:
    """Stage one query's working dir. Returns a status record."""
    tid = q["task_id"]
    qid = q["query_id"]
    domain = q["task_domain"]
    scope_id = q["scope"]["scope_id"]
    qdir = DATA_ROOT / tid / qid

    # Find the scope directory — check the task's own domain first, then fall
    # back to any other domain (cross-domain scope references are common in
    # General-domain tasks that reference camels_us_531 / khaki_13state etc.)
    scope_dir = SCOPES_ROOT / domain / scope_id
    if not scope_dir.exists():
        for alt in ("CAMELS", "CropBench", "MethaneWet", "General"):
            cand = SCOPES_ROOT / alt / scope_id
            if cand.exists():
                scope_dir = cand
                break
    if not scope_dir.exists():
        return {"query_id": qid, "task_id": tid, "domain": domain,
                "scope_id": scope_id, "split": q["split"],
                "status": "scope_missing", "staged": [], "missing": []}

    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "results").mkdir(exist_ok=True)

    # Write query.json (full metadata) + prompt.txt (prompt field, augmented)
    (qdir / "query.json").write_text(json.dumps(q, indent=2) + "\n")

    # Find the task's file list from query.reference_files if present, else
    # re-read from the parent task file. Simpler: pull from input_data.files
    # via the task metadata by scanning mas_bench_tasks_v4.jsonl once.
    # For now, the query doesn't embed input_data.files; read it from the task.
    task_files = _TASK_INPUT_FILES.get(tid, [])

    # Raw + synth dirs for this domain
    raw_dir = DOMAIN_RAW_ROOTS.get(domain, ROOT / "raw_data" / domain)
    synth_dir = raw_dir / "synth"

    staged: list[str] = []
    missing: list[str] = []
    for tf in task_files:
        if domain == "CAMELS":
            src = _pick_camels_source(tf, scope_dir, synth_dir)
        elif domain == "CropBench":
            src = _pick_cropbench_source(tf, scope_dir, synth_dir, raw_dir)
        elif domain == "MethaneWet":
            src = _pick_methanewet_source(tf, synth_dir, raw_dir)
        elif domain == "General":
            src = _pick_general_source(tf, scope_dir, synth_dir, raw_dir)
        else:
            src = None

        if src is None:
            target_name = f"MISSING__{tf}"
            (qdir / target_name).write_text(
                f"File not yet available: {tf}\n"
                f"No scope/raw/synth source resolved (domain={domain}, scope={scope_id}).\n"
            )
            missing.append(tf)
            continue

        target = qdir / _file_target_name(tf, src)
        if target.exists() or target.is_symlink():
            if overwrite:
                target.unlink()
            else:
                staged.append(target.name)
                continue
        target.symlink_to(os.path.relpath(src.resolve(), qdir))
        staged.append(target.name)

    # Write augmented prompt
    file_listing = "\n".join(f"  - {f}" for f in sorted(staged)) or "  (no data files staged yet)"
    missing_listing = (
        "\n\nThese files are referenced by the task but not available yet:\n"
        + "\n".join(f"  - {m}" for m in missing)
        if missing else ""
    )
    # Enumerate the exact metric names the grader will look for.
    required_metrics = q.get("metrics", [])
    metrics_listing = "\n".join(
        f"  - {m['name']}  (target: {m.get('target','')})"
        for m in required_metrics
    ) or "  (task defines no metrics — write any sensible numeric summary)"
    augmented = (
        q["prompt"]
        + "\n\nFILES IN YOUR WORKING DIRECTORY (parquet format, not CSV):\n"
        + file_listing
        + missing_listing
        + "\n\nREQUIRED OUTPUT — write results/metrics.json as a flat JSON object "
          "mapping these EXACT metric names (strings) to numeric values. "
          "Do NOT invent other keys; the grader matches by name.\n"
        + metrics_listing
        + "\n\nIf a metric is unreachable in your budget, write your best estimate "
          "(or 0.0) rather than skipping it.\n"
    )
    (qdir / "prompt.txt").write_text(augmented)

    return {
        "query_id": qid,
        "task_id": tid,
        "domain": domain,
        "scope_id": scope_id,
        "split": q["split"],
        "status": "ok" if staged and not missing else ("partial" if staged else "empty"),
        "staged": staged,
        "missing": missing,
    }


# ----------------------------------------------------------------------------
# Task input_data.files index (read once)
# ----------------------------------------------------------------------------

_TASK_INPUT_FILES: dict[str, list[str]] = {}


def _load_task_files() -> None:
    tasks_path = ROOT / "data_tasks" / "mas_bench_tasks_v4.jsonl"
    for line in open(tasks_path):
        t = json.loads(line)
        _TASK_INPUT_FILES[t["task_id"]] = t["input_data"]["files"]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", help="Restrict to one domain (CAMELS/CropBench/MethaneWet/General)")
    ap.add_argument("--task", help="Restrict to one task_id (e.g. C1.1)")
    ap.add_argument("--limit", type=int, help="Limit to N queries")
    ap.add_argument("--overwrite", action="store_true", help="Replace existing symlinks")
    args = ap.parse_args()

    _load_task_files()

    queries = [json.loads(l) for l in open(QUERIES)]
    if args.domain:
        queries = [q for q in queries if q["task_domain"] == args.domain]
    if args.task:
        queries = [q for q in queries if q["task_id"] == args.task]
    if args.limit:
        queries = queries[: args.limit]

    print(f"Staging {len(queries)} queries...")
    results = [stage_query(q, overwrite=args.overwrite) for q in queries]

    from collections import Counter
    statuses = Counter(r["status"] for r in results)
    print(f"\nStatus summary: {dict(statuses)}")

    # Per-task breakdown (brief)
    by_task: dict[str, Counter] = {}
    for r in results:
        by_task.setdefault(r["task_id"], Counter())[r["status"]] += 1
    for tid in sorted(by_task):
        c = by_task[tid]
        print(f"  {tid}: ok={c.get('ok',0)} partial={c.get('partial',0)} empty={c.get('empty',0)} scope_missing={c.get('scope_missing',0)}")

    # Log
    log_path = DATA_ROOT / "_stage_log.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nWrote staging log to {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
