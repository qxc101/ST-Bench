"""
Render a compact per-domain + per-task summary of a grading_report.json
produced by grader/grade.py.

Usage:
    python grader/summarize.py                       # summarize newest run
    python grader/summarize.py --run <run_id>
    python grader/summarize.py --report path.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# Task-id prefix -> domain (from task_id naming: C*, F*, P*, etc.)
TASK_FIRST_LETTER_DOMAIN = {
    "C":  "Clustering / Catchment",  # but ambiguous; not used
}


def load_tasks_meta():
    tasks_p = ROOT / "data_tasks" / "mas_bench_tasks_v4.jsonl"
    return {json.loads(l)["task_id"]: json.loads(l) for l in open(tasks_p)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="Run id under runs/ (default newest)")
    ap.add_argument("--report", default=None, help="Explicit path to grading_report.json")
    args = ap.parse_args()

    if args.report:
        report_path = Path(args.report)
    else:
        # newest run
        if args.run:
            run_dir = RUNS / args.run
        else:
            runs = sorted([p for p in RUNS.iterdir() if p.is_dir()],
                          key=lambda p: p.stat().st_mtime, reverse=True)
            if not runs:
                print("No runs/ folder content."); return 1
            run_dir = runs[0]
        report_path = run_dir / "grading_report.json"
    if not report_path.exists():
        print(f"ERROR: {report_path} not found — run grader/grade.py first"); return 1

    data = json.loads(report_path.read_text())
    tasks_meta = load_tasks_meta()

    # Group per domain
    by_domain = {}
    for task_rep in data["per_task"]:
        tid = task_rep["task_id"]
        dom = tasks_meta.get(tid, {}).get("domain", "?")
        by_domain.setdefault(dom, []).append(task_rep)

    print(f"=== Summary for {report_path} ===\n")

    total_q = total_succ = 0
    for dom in sorted(by_domain):
        tasks = by_domain[dom]
        n_q = sum(t["n_queries"] for t in tasks)
        n_succ = sum(t.get("n_success", 0) for t in tasks)
        total_q += n_q
        total_succ += n_succ
        print(f"--- {dom}: {len(tasks)} tasks · {n_q} queries · success {n_succ}/{n_q} ({100*n_succ/n_q:.1f}%) ---")
        # Per-task brief
        for t in sorted(tasks, key=lambda x: x["task_id"]):
            n = t["n_queries"]; s = t.get("n_success", 0)
            tid = t["task_id"]
            # Pass-rate across metrics
            metrics = t.get("per_metric", {})
            pr_parts = []
            for mname, agg in metrics.items():
                pr = agg.get("pass_rate")
                mean = agg.get("mean")
                if pr is not None:
                    pr_parts.append(f"{mname[:16]}={100*pr:.0f}%")
                elif mean is not None:
                    pr_parts.append(f"{mname[:16]}={mean:.3g}")
            pr_str = "  ".join(pr_parts[:3])
            print(f"  {tid:7s} {s:3d}/{n:<3d} ({100*s/n:5.1f}%)  {pr_str}")
        print()

    print(f"=== OVERALL: {total_succ}/{total_q} ({100*total_succ/max(1,total_q):.1f}%) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
