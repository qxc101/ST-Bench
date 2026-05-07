"""
Single-agent GPT-5 code loop for ST_Bench.

Design goals (user: "keep this simple so that the result actually represents
single-agent capability"):
  - No tool-use protocol beyond extracting ```python ... ``` blocks from the
    response and exec-ing them.
  - No self-reflection / retry prompts. Just feed back stdout/stderr.
  - Fixed turn budget; fixed per-exec timeout; fixed output truncation.
  - Agent writes results/metrics.json (and optionally results/*) in its
    working directory. Runner stops when that file appears or budget expires.

Public entry point: run_query(query_dir: Path, model: str = 'gpt-5',
max_turns: int = 10, exec_timeout: int = 300) -> dict
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

from runner.env import load_env_file

load_env_file()

BASE = "https://api.genai-api.nec-cloud.com/genai-oai-api/v1"

SYSTEM_PROMPT = """You are a single-agent data scientist working on a benchmark query.
You have access to a Python code-execution tool and a working directory that contains
the query's prompt, data files, and a `results/` subfolder where you must write outputs.

Protocol:
  1. Your responses may contain Python code blocks using triple-backtick python fences.
  2. The harness will execute each code block in order in a persistent Python process
     rooted at the working directory, and reply with the combined stdout+stderr
     (truncated to 8 KB). Variables persist across turns.
  3. When finished, write the required metrics to `results/metrics.json` as a single
     JSON object mapping metric name -> numeric value (no nested structures).
     Optionally write supporting files (predictions.csv, plots, notes) under `results/`.
  4. After the metrics file is written, respond with a one-line message `DONE` and
     stop producing code. The harness checks for results/metrics.json and exits.

Constraints:
  - Turn budget is limited; prioritize reaching valid metrics.json over perfection.
  - Per-code-block timeout is 5 minutes; split long work into smaller blocks.
  - Use packages already available in the environment (pandas, numpy, scikit-learn,
    pyarrow). If a needed package is missing, pip-install it inside a code block.
  - Data files in the working directory are your inputs; don't fetch external data.
"""

CODE_RE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)
MAX_OUTPUT_BYTES = 8192
EXEC_SCRIPT_NAME = "_agent_exec.py"   # written/overwritten each turn
STATE_FILE = "_agent_state.pkl"       # pickled interpreter globals for persistence


def _chat(messages: list[dict], model: str, api_key: str,
          max_completion_tokens: int = 32768,
          temperature: float | None = None) -> str:
    """POST to the chat completion endpoint with exponential backoff on
    429 (rate limit), 5xx (server error), and network timeouts. Raises
    RuntimeError only after exhausting retries or on unrecoverable 4xx.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    backoffs = [5, 15, 45, 90, 180]  # seconds; ~5.5 min total
    last_err: str = "no request attempted"
    for attempt, delay in enumerate([0] + backoffs):
        if delay:
            time.sleep(delay)
        try:
            r = requests.post(
                f"{BASE}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = f"network_error: {e.__class__.__name__}: {str(e)[:200]}"
            continue
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        # 429 or 5xx -> retry; 4xx other -> give up immediately
        if r.status_code == 429 or 500 <= r.status_code < 600:
            last_err = f"http_{r.status_code}: {r.text[:200]}"
            continue
        raise RuntimeError(f"chat API {r.status_code}: {r.text[:400]}")
    raise RuntimeError(f"chat API retries exhausted: {last_err}")


def _extract_code(text: str) -> list[str]:
    return [m.strip() for m in CODE_RE.findall(text) if m.strip()]


def _exec_in_workdir(code: str, workdir: Path, timeout: int) -> tuple[int, str]:
    """Execute `code` in workdir with stbench python, persisting globals across turns via pickle.
    Returns (returncode, combined_output_truncated_to_MAX_OUTPUT_BYTES).
    """
    # Wrap user code with load/save of a persistent globals dict
    wrapper = f"""
import os, sys, pickle, pathlib, traceback, io, contextlib
_state_path = pathlib.Path({str(workdir / STATE_FILE)!r})
_g = {{}}
if _state_path.exists():
    try:
        _g = pickle.loads(_state_path.read_bytes())
    except Exception:
        _g = {{}}
_g['__name__'] = '__agent__'
_user_src = {code!r}
try:
    exec(_user_src, _g)
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
# Save only picklable entries
_keep = {{}}
for k, v in _g.items():
    if k.startswith('__') and k != '__name__':
        continue
    try:
        pickle.dumps(v)
        _keep[k] = v
    except Exception:
        pass
_state_path.write_bytes(pickle.dumps(_keep))
"""
    exec_file = workdir / EXEC_SCRIPT_NAME
    exec_file.write_text(wrapper)
    py = sys.executable or "/home/qcheng/miniconda3/envs/stbench/bin/python"
    try:
        proc = subprocess.run(
            [py, str(exec_file)],
            cwd=str(workdir),
            capture_output=True, text=True, timeout=timeout,
        )
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        if len(out.encode("utf-8", errors="ignore")) > MAX_OUTPUT_BYTES:
            out = out[:MAX_OUTPUT_BYTES] + f"\n...[truncated, total {len(out)} chars]"
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        return 124, f"[timeout after {timeout}s]"


def run_query(
    query_dir: Path,
    model: str = "gpt-5",
    max_turns: int = 10,
    exec_timeout: int = 300,
    wall_time_s: int = 1800,
    max_completion_tokens: int = 32768,
    api_key: str | None = None,
) -> dict:
    """Run the single-agent loop for one query. Returns a summary dict and leaves
    artifacts in query_dir/results/.
    """
    query_dir = Path(query_dir).resolve()
    results_dir = query_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Load prompt
    prompt_path = query_dir / "prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"missing prompt.txt at {prompt_path}")
    user_prompt = prompt_path.read_text()

    api_key = api_key or os.environ.get("INTERNAL_API_KEY", "")
    if not api_key:
        raise RuntimeError("INTERNAL_API_KEY not set")

    # Fresh state for each run
    state_path = query_dir / STATE_FILE
    if state_path.exists():
        state_path.unlink()

    # Opening message tells the agent where it is and what to produce
    file_listing = "\n".join(
        f"  {p.relative_to(query_dir)}" for p in sorted(query_dir.rglob("*"))
        if p.is_file() and p.name not in (EXEC_SCRIPT_NAME, STATE_FILE)
    )
    opening_user = (
        f"Working directory: {query_dir}\n"
        f"Files available:\n{file_listing}\n\n"
        f"Query prompt:\n{user_prompt}\n\n"
        f"Write your metrics to results/metrics.json when done and reply DONE."
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": opening_user},
    ]

    log_path = query_dir / "agent_log.jsonl"
    log_f = open(log_path, "w")

    def log(event: dict) -> None:
        log_f.write(json.dumps(event, default=str) + "\n")
        log_f.flush()

    start = time.time()
    metrics_path = results_dir / "metrics.json"
    reason_done = None

    for turn in range(1, max_turns + 1):
        if time.time() - start > wall_time_s:
            reason_done = f"wall_time_exceeded_{wall_time_s}s"
            log({"turn": turn, "event": "wall_time_exceeded"})
            break
        try:
            assistant = _chat(messages, model=model, api_key=api_key,
                              max_completion_tokens=max_completion_tokens)
        except Exception as e:
            reason_done = f"chat_error: {e}"
            log({"turn": turn, "event": "chat_error", "error": str(e)})
            break
        messages.append({"role": "assistant", "content": assistant})
        log({"turn": turn, "event": "assistant", "content": assistant})

        # Execute code blocks FIRST (if any) — the agent may send code + DONE
        # in the same turn and we must run the code before checking completion.
        code_blocks = _extract_code(assistant)
        done_signal = any("DONE" == s.strip() for s in assistant.splitlines()[-5:])

        if code_blocks:
            all_output_parts: list[str] = []
            for i, code in enumerate(code_blocks):
                rc, out = _exec_in_workdir(code, query_dir, exec_timeout)
                all_output_parts.append(f"--- block {i+1} (exit={rc}) ---\n{out}")
                log({"turn": turn, "event": "exec", "block": i+1, "rc": rc, "output": out})
                if metrics_path.exists():
                    break
            combined = "\n".join(all_output_parts)
            messages.append({"role": "user", "content": combined})

        if metrics_path.exists():
            reason_done = "metrics_written"
            break

        if done_signal:
            # Agent said DONE but no metrics.json — prompt to write one
            messages.append({"role": "user", "content":
                "You said DONE but results/metrics.json does not exist (or is empty). "
                "Write a JSON object with the REQUIRED metric names (see query.json "
                "'metrics' field) mapped to numeric values to results/metrics.json, "
                "then reply DONE again."})
            continue

        if not code_blocks:
            messages.append({"role": "user", "content":
                "No code block detected. Either provide a ```python fence to execute "
                "or write results/metrics.json and reply DONE."})
            continue

    elapsed = time.time() - start
    success = metrics_path.exists()
    metrics_obj: dict | None = None
    if success:
        try:
            metrics_obj = json.loads(metrics_path.read_text())
        except Exception as e:
            success = False
            reason_done = f"metrics_parse_error: {e}"

    summary = {
        "query_dir": str(query_dir),
        "model": model,
        "success": success,
        "reason_done": reason_done or "budget_exceeded",
        "turns_used": turn if 'turn' in locals() else 0,
        "elapsed_s": round(elapsed, 1),
        "metrics": metrics_obj,
    }
    (query_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    log({"event": "summary", **summary})
    log_f.close()

    # Clean persistent state so re-runs start fresh
    if state_path.exists():
        state_path.unlink()
    exec_file = query_dir / EXEC_SCRIPT_NAME
    if exec_file.exists():
        exec_file.unlink()

    return summary
