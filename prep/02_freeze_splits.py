"""
Freeze the per-task train/val/test split from query metadata into
splits/<task_id>.json. These files are the authoritative source for
MAS frameworks that optimize on train and evaluate on test.

The split field is already present on each query in
data_tasks/mas_bench_queries_v4.jsonl; this script just persists the
mapping {query_id: split} per task into a stable on-disk form.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "data_tasks" / "mas_bench_queries_v4.jsonl"
OUT_DIR = ROOT / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    by_task: dict[str, list[dict]] = defaultdict(list)
    with open(QUERIES) as f:
        for line in f:
            q = json.loads(line)
            by_task[q["task_id"]].append({
                "query_id": q["query_id"],
                "split": q["split"],
                "scope_id": q["scope"]["scope_id"],
                "scope_name": q["scope"]["name"],
            })

    totals = Counter()
    for tid, queries in sorted(by_task.items()):
        sp = Counter(q["split"] for q in queries)
        totals.update(sp)
        out = {
            "task_id": tid,
            "n_queries": len(queries),
            "counts": {"train": sp["train"], "val": sp["val"], "test": sp["test"]},
            "train":  sorted(q["query_id"] for q in queries if q["split"] == "train"),
            "val":    sorted(q["query_id"] for q in queries if q["split"] == "val"),
            "test":   sorted(q["query_id"] for q in queries if q["split"] == "test"),
        }
        (OUT_DIR / f"{tid}.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"Wrote {len(by_task)} split files to {OUT_DIR}")
    print(f"Totals across tasks: train={totals['train']} val={totals['val']} test={totals['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
