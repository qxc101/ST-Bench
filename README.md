# ST-Bench

ST-Bench is a spatial-temporal scientific data-analysis benchmark for evaluating single-agent and generated multi-agent systems on Earth-science workflows.

## Data
The query-only release is  hosted at https://huggingface.co/datasets/qic69/ST-Bench.

## Setup


Set any required model API keys in your shell or `.env`.

## Use

```bash
# stage query working directories under data/<task_id>/<query_id>/
python prep/04_stage_queries.py

# stage a subset
python prep/04_stage_queries.py --domain CAMELS
python prep/04_stage_queries.py --task C1.1 --limit 10

# grade newest run
python grader/grade.py

# grade a specific run
python grader/grade.py --run <run_id> --split test
python grader/grade.py --run <run_id> --task C1.1
```

Runs are written to `runs/<run_id>/`. The grader expects each query result at `results/metrics.json`.
