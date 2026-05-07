"""
Grade ST_Bench runs.

Two metrics per task:
  1. Success rate — fraction of queries where results/metrics.json exists and
     parses as a JSON object.
  2. Per-metric scores — for each metric defined on the task, the per-query
     value the agent reported, plus a pass/fail flag when the metric's target
     string is an inequality we can parse (> X, < X, >= X, <= X, > X%, ...).

Usage:
    conda activate stbench
    python grader/grade.py                      # grade everything in data/
    python grader/grade.py --task C1.1          # grade one task
    python grader/grade.py --split test         # grade one split across all tasks
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
RUNS_ROOT = ROOT / "runs"
SPLITS_ROOT = ROOT / "splits"
TASKS_JSONL = ROOT / "data_tasks" / "mas_bench_tasks_v4.jsonl"
REPORT_DIR = ROOT / "results"


def resolve_run_dir(run_arg: str | None) -> Path | None:
    """Return the per-run queries dir to grade from.
    If run_arg is None, pick the newest runs/<id>/. If it's a bare id, expand.
    If it's an absolute/relative path, use as-is.
    Returns None to mean "grade legacy data/ layout".
    """
    if run_arg == "legacy":
        return None
    if run_arg:
        p = Path(run_arg)
        if not p.exists():
            p = RUNS_ROOT / run_arg
        return p
    # Default: newest run folder if any exist
    candidates = sorted([p for p in RUNS_ROOT.glob("*") if p.is_dir()],
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


# --- Metric target parser -----------------------------------------------------

_TARGET_RE = re.compile(r"(>=|<=|>|<)\s*(-?\d+\.?\d*)\s*(%?)")


def parse_target(target: str) -> tuple[str, float] | None:
    """Parse '> 0.3' or '> 80%' into ('>', 0.3) / ('>', 0.8). Return None if not an inequality."""
    if not isinstance(target, str):
        return None
    m = _TARGET_RE.search(target)
    if not m:
        return None
    op, num, pct = m.group(1), float(m.group(2)), m.group(3)
    if pct == "%":
        num /= 100.0
    return op, num


def passes(value: float, op: str, threshold: float) -> bool:
    return {
        ">":  value > threshold,
        "<":  value < threshold,
        ">=": value >= threshold,
        "<=": value <= threshold,
    }[op]


# --- Name-matching helpers ----------------------------------------------------

def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _match_metric(reported: dict, metric_name: str) -> float | None:
    """Find a value in `reported` whose key canonicalizes to metric_name's canon form."""
    target = _canon(metric_name)
    # Exact canonical match first
    for k, v in reported.items():
        if _canon(k) == target and isinstance(v, (int, float)):
            return float(v)
    # Substring fallback (e.g. reported "silhouette" matches metric "Silhouette Score")
    for k, v in reported.items():
        if (target in _canon(k) or _canon(k) in target) and isinstance(v, (int, float)):
            return float(v)
    return None


# --- Per-task grading ---------------------------------------------------------

def load_tasks() -> dict[str, dict]:
    return {json.loads(l)["task_id"]: json.loads(l) for l in open(TASKS_JSONL)}


def load_splits() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in SPLITS_ROOT.glob("*.json")}


def grade_task(task: dict, split_info: dict, split_filter: str | None = None,
               run_dir: Path | None = None) -> dict:
    tid = task["task_id"]
    metrics_def = task.get("metrics", [])

    # Gather query IDs for this task, optionally filtered by split
    if split_filter:
        qids = split_info.get(split_filter, [])
    else:
        qids = split_info["train"] + split_info["val"] + split_info["test"]

    # Per-query records
    records: list[dict] = []
    for qid in qids:
        if run_dir is not None:
            qdir = run_dir / "queries" / qid
        else:
            qdir = DATA_ROOT / tid / qid
        metrics_path = qdir / "results" / "metrics.json"
        rec: dict = {"query_id": qid, "split": None}
        # Split lookup
        for sp in ("train", "val", "test"):
            if qid in split_info.get(sp, []):
                rec["split"] = sp
                break

        if not metrics_path.exists():
            rec["status"] = "no_results"
            records.append(rec)
            continue
        try:
            reported = json.loads(metrics_path.read_text())
            if not isinstance(reported, dict):
                rec["status"] = "malformed"
                records.append(rec)
                continue
        except Exception as e:
            rec["status"] = f"parse_error: {e}"
            records.append(rec)
            continue

        rec["status"] = "ok"
        rec["metrics"] = {}
        rec["pass"] = {}
        for m in metrics_def:
            mname = m["name"]
            value = _match_metric(reported, mname)
            rec["metrics"][mname] = value
            tgt = parse_target(m.get("target", ""))
            if tgt is None or value is None:
                rec["pass"][mname] = None  # not checkable
            else:
                op, thr = tgt
                rec["pass"][mname] = passes(value, op, thr)
        records.append(rec)

    # Aggregate
    n_total = len(records)
    n_success = sum(1 for r in records if r["status"] == "ok")
    per_metric_agg: dict[str, dict] = {}
    for m in metrics_def:
        vals = [r["metrics"][m["name"]] for r in records
                if r["status"] == "ok" and r["metrics"].get(m["name"]) is not None]
        passes_list = [r["pass"][m["name"]] for r in records
                       if r["status"] == "ok" and r["pass"].get(m["name"]) is not None]
        per_metric_agg[m["name"]] = {
            "target":        m.get("target", ""),
            "n_reported":    len(vals),
            "mean":          mean(vals) if vals else None,
            "median":        median(vals) if vals else None,
            "pass_rate":     (sum(passes_list) / len(passes_list)) if passes_list else None,
            "n_passes_checked": len(passes_list),
        }

    return {
        "task_id": tid,
        "split": split_filter or "all",
        "n_queries": n_total,
        "success_rate": n_success / n_total if n_total else None,
        "n_success": n_success,
        "per_metric": per_metric_agg,
        "records": records,
    }


# --- Main ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="Restrict to one task")
    ap.add_argument("--split", choices=["train", "val", "test"], help="Filter by split")
    ap.add_argument("--out", help="Output report path (JSON). Default: <run>/grading_report.json")
    ap.add_argument("--run", default=None,
                    help="Run folder (under runs/) to grade. Default: newest. "
                         "Pass 'legacy' to grade the old data/ layout.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    tasks = load_tasks()
    splits = load_splits()

    run_dir = resolve_run_dir(args.run)
    if run_dir is not None:
        if not (run_dir / "queries").exists():
            print(f"ERROR: {run_dir} has no queries/ subdir")
            return 2
        print(f"Grading run: {run_dir}")
        REPORT_DIR_EFFECTIVE = run_dir
    else:
        print("Grading legacy data/ layout (no run folder)")
        REPORT_DIR_EFFECTIVE = REPORT_DIR
        REPORT_DIR_EFFECTIVE.mkdir(exist_ok=True)

    target_tasks = [args.task] if args.task else sorted(tasks)

    summaries = []
    for tid in target_tasks:
        if tid not in tasks or tid not in splits:
            print(f"  skip {tid} (missing task or split metadata)")
            continue
        g = grade_task(tasks[tid], splits[tid], split_filter=args.split, run_dir=run_dir)
        summaries.append(g)
        if args.verbose or args.task:
            print(f"\n=== {tid} ({g['n_queries']} queries, split={g['split']}) ===")
            print(f"  success_rate: {g['success_rate']:.1%} ({g['n_success']}/{g['n_queries']})"
                  if g['n_queries'] else "  (no queries)")
            for mname, agg in g["per_metric"].items():
                print(f"    {mname}: target={agg['target']!r}  reported={agg['n_reported']}"
                      + (f"  mean={agg['mean']:.4g}"         if agg['mean']       is not None else "")
                      + (f"  pass_rate={agg['pass_rate']:.1%}" if agg['pass_rate']  is not None else ""))

    # Overall (no records, just aggregates)
    overall = [{k: v for k, v in s.items() if k != "records"} for s in summaries]
    n_q_total = sum(s["n_queries"] for s in summaries)
    n_succ_total = sum(s["n_success"] for s in summaries)
    print(f"\n=== OVERALL ===")
    print(f"  tasks graded: {len(summaries)}")
    if n_q_total:
        print(f"  total queries: {n_q_total}, success rate {n_succ_total/n_q_total:.1%}"
              f" ({n_succ_total}/{n_q_total})")

    out_path = Path(args.out) if args.out else (REPORT_DIR_EFFECTIVE / (
        f"grading_report{('_' + args.split) if args.split else ''}.json"))
    out_path.write_text(json.dumps({"per_task": summaries, "overall": overall}, indent=2, default=str) + "\n")
    print(f"\nWrote report -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
