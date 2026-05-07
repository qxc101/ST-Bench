"""
Driver that iterates over staged query folders and runs the single-agent loop.

Usage:
    conda activate stbench
    export INTERNAL_API_KEY=...
    python runner/run_queries.py --task C1.1 [--split test] [--limit 2]
    python runner/run_queries.py --all

Layout assumed:
    data/<task_id>/<query_id>/prompt.txt            (staged by prep/04_stage_queries.py)
    data/<task_id>/<query_id>/results/metrics.json  (written by the agent)
    data/<task_id>/<query_id>/run_summary.json      (written by the runner)
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from runner.env import load_env_file  # noqa: E402
load_env_file()
from runner.agent import run_query  # noqa: E402

DATA_DIR = ROOT / "data"
SPLITS_DIR = ROOT / "splits"
RUNS_DIR = ROOT / "runs"


# Files inside a staged query folder that should be visible to the agent
# (symlinked into the per-run working dir). Anything else created during
# a run (metrics.json, exec scripts, state pickle, agent_log, etc.) lives
# only inside the run folder.
_STAGED_PASSTHROUGH = {"prompt.txt", "query.json"}


def _new_run_id(system: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rand = secrets.token_hex(3)
    # Sanitize system name for filesystem (replace / : spaces)
    safe = system.replace("/", "-").replace(":", "-").replace(" ", "-")
    return f"{safe}__{ts}__{rand}"


def _prepare_workdir(run_dir: Path, staged_qdir: Path, qid: str) -> Path:
    """Create run_dir/queries/<qid>/ and symlink data + prompt.txt + query.json from
    the staged query folder. Returns the per-run working dir.
    """
    workdir = run_dir / "queries" / qid
    (workdir / "results").mkdir(parents=True, exist_ok=True)

    for p in staged_qdir.iterdir():
        if p.is_dir():
            continue  # never mirror subdirectories (e.g. old results from prior runs)
        name = p.name
        # Skip anything that looks like a prior-run artifact
        if name.startswith("MISSING__") or name in ("agent_log.jsonl", "run_summary.json",
                                                     "_agent_exec.py", "_agent_state.pkl"):
            continue
        link = workdir / name
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(os.path.relpath(p.resolve(), workdir))
    return workdir


def pick_queries(task: str | None, split: str | None, limit: int | None) -> list[tuple[str, Path]]:
    """Return (task_id, query_dir) pairs filtered by --task and --split."""
    if task:
        task_ids = [task]
    else:
        task_ids = [p.stem for p in sorted(SPLITS_DIR.glob("*.json"))]

    out: list[tuple[str, Path]] = []
    for tid in task_ids:
        split_file = SPLITS_DIR / f"{tid}.json"
        if not split_file.exists():
            continue
        meta = json.loads(split_file.read_text())

        if split:
            qids = meta.get(split, [])
        else:
            qids = meta["train"] + meta["val"] + meta["test"]

        for qid in qids:
            qdir = DATA_DIR / tid / qid
            if qdir.is_dir():
                out.append((tid, qdir))
    if limit is not None:
        out = out[:limit]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="Restrict to a specific task_id (e.g. C1.1). Default: all tasks.")
    ap.add_argument("--split", choices=["train", "val", "test"], help="Restrict to one split.")
    ap.add_argument("--limit", type=int, help="Limit to N queries (for smoke tests).")
    ap.add_argument("--model", default="gpt-5", help="Model name (default: gpt-5).")
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--exec-timeout", type=int, default=300,
                    help="Per-code-block timeout in seconds.")
    ap.add_argument("--wall-time", type=int, default=1800,
                    help="Per-query wall-clock budget in seconds (default 30 min).")
    ap.add_argument("--max-completion-tokens", type=int, default=32768,
                    help="Chat max_completion_tokens (GPT-5 handles 32K; GPT-4o caps at 16384).")
    ap.add_argument("--parallel", type=int, default=1,
                    help="Run up to N queries concurrently in threads.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip queries whose results/metrics.json already exists.")
    ap.add_argument("--all", action="store_true", help="Run every staged query.")
    ap.add_argument("--query-id", action="append",
                    help="Run only these specific query_ids (repeatable).")
    ap.add_argument("--system", default=None,
                    help="System/agent name for the run folder (default: --model).")
    ap.add_argument("--run-id", default=None,
                    help="Explicit run id (default: <system>__<utc-timestamp>__<rand>).")
    ap.add_argument("--only-ok", action="store_true",
                    help="Restrict to queries whose stage status is 'ok' (all task files staged). "
                         "Reads data/_stage_log.jsonl from the most recent stage run.")
    ap.add_argument("--include-partial", action="store_true",
                    help="With --only-ok, also include queries whose status is 'partial' "
                         "(some files missing but most staged).")
    args = ap.parse_args()

    if not (args.task or args.all or args.limit or args.only_ok or args.query_id):
        ap.error("must pass --task, --all, --limit, --only-ok, or --query-id")

    if not os.environ.get("INTERNAL_API_KEY"):
        print("ERROR: INTERNAL_API_KEY env var is not set", file=sys.stderr)
        return 2

    queries = pick_queries(args.task, args.split, args.limit)
    if args.query_id:
        wanted = set(args.query_id)
        queries = [(t, q) for t, q in queries if q.name in wanted]
    if args.only_ok:
        log_path = DATA_DIR / "_stage_log.jsonl"
        if not log_path.exists():
            print(f"ERROR: --only-ok requires {log_path} (run prep/04_stage_queries.py first)")
            return 2
        keep_statuses = {"ok"}
        if args.include_partial:
            keep_statuses.add("partial")
        ok_ids: set[str] = set()
        for line in open(log_path):
            r = json.loads(line)
            if r.get("status") in keep_statuses:
                ok_ids.add(r["query_id"])
        pre = len(queries)
        queries = [(t, q) for t, q in queries if q.name in ok_ids]
        print(f"(--only-ok{' +partial' if args.include_partial else ''}) {len(queries)}/{pre} queries kept from stage log")
    if not queries:
        print("No queries matched.")
        return 1

    # Set up the run folder — all per-query outputs live under here.
    system = args.system or args.model
    run_id = args.run_id or _new_run_id(system)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "queries").mkdir(exist_ok=True)

    metadata = {
        "run_id": run_id,
        "system": system,
        "model": args.model,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "args": {
            "task": args.task, "split": args.split, "limit": args.limit,
            "max_turns": args.max_turns, "exec_timeout": args.exec_timeout,
            "wall_time": args.wall_time, "parallel": args.parallel,
            "query_ids": args.query_id,
        },
        "n_queries_planned": len(queries),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Run folder: {run_dir}")

    # Filter out resumable (already-complete) queries up front — in this run
    if args.resume:
        pre = len(queries)
        queries = [(t, q) for t, q in queries
                   if not (run_dir / "queries" / q.name / "results" / "metrics.json").exists()]
        print(f"(resume) skipping {pre - len(queries)} already-complete queries in this run")

    def run_one(item: tuple[str, Path]) -> tuple[str, Path, dict | Exception]:
        tid, staged_qdir = item
        qid = staged_qdir.name
        workdir = _prepare_workdir(run_dir, staged_qdir, qid)
        try:
            s = run_query(
                workdir, model=args.model,
                max_turns=args.max_turns,
                exec_timeout=args.exec_timeout,
                wall_time_s=args.wall_time,
                max_completion_tokens=args.max_completion_tokens,
            )
            # Augment the run_summary with task_id + staged source for easy grading
            summary_path = workdir / "run_summary.json"
            if summary_path.exists():
                d = json.loads(summary_path.read_text())
                d["task_id"] = tid
                d["staged_query_dir"] = str(staged_qdir)
                summary_path.write_text(json.dumps(d, indent=2, default=str) + "\n")
            return tid, staged_qdir, s
        except Exception as e:
            return tid, staged_qdir, e

    print(f"Running {len(queries)} queries (parallel={args.parallel}, model={args.model})...")
    n_ok = n_fail = 0

    if args.parallel <= 1:
        iter_results = (run_one(x) for x in queries)
    else:
        ex = ThreadPoolExecutor(max_workers=args.parallel)
        futures = [ex.submit(run_one, x) for x in queries]
        iter_results = (f.result() for f in as_completed(futures))

    done = 0
    for tid, qdir, res in iter_results:
        done += 1
        if isinstance(res, Exception):
            n_fail += 1
            print(f"[{done}/{len(queries)}] ERROR {tid}/{qdir.name}: {res}")
            continue
        if res["success"]:
            n_ok += 1
            print(f"[{done}/{len(queries)}] ok  {tid}/{qdir.name}  turns={res['turns_used']} elapsed={res['elapsed_s']}s")
        else:
            n_fail += 1
            print(f"[{done}/{len(queries)}] FAIL {tid}/{qdir.name}  reason={res['reason_done']} turns={res['turns_used']}")

    finished_metadata = {
        **metadata,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "n_ok": n_ok,
        "n_fail": n_fail,
    }
    (run_dir / "metadata.json").write_text(json.dumps(finished_metadata, indent=2) + "\n")
    print(f"\nDone: ok={n_ok} fail={n_fail}   run={run_dir}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
