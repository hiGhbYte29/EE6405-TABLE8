"""Quiz benchmark, step 2: send the jobs to one or more models.

    python scripts/run_bench.py --jobs reports/bench/jobs.jsonl \
        --out reports/bench/responses_deepseek.jsonl --models deepseek-flash deepseek-v4-pro

Standard library only, on purpose: the jobs file already contains the rendered
prompts, so this script can be copied to any machine that can reach an
OpenAI-compatible endpoint and run there with a bare ``python3``.

The endpoint and key come from ``EE6405_LLM_BASE_URL`` / ``EE6405_LLM_API_KEY``
(read from ``--env-file`` if they are not already set), so no key is ever
passed on the command line. The run is resumable: a (job, model) pair that
already has an answer in ``--out`` is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def chat(base_url: str, key: str, model: str, messages: list, extra: dict, timeout: float) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0, **extra}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_one(job: dict, model: str, args, base_url: str, key: str) -> dict:
    last = ""
    for attempt in range(args.retries):
        t0 = time.time()
        try:
            data = chat(base_url, key, model, job["messages"], args.extra, args.timeout)
            message = data["choices"][0]["message"]
            usage = data.get("usage") or {}
            return {
                "job_id": job["job_id"], "model": model, "text": message.get("content") or "",
                "latency": round(time.time() - t0, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
            if e.code in (400, 401, 403, 404):
                break  # retrying will not help
        except Exception as e:  # timeouts, resets, malformed bodies
            last = f"{type(e).__name__}: {e}"
        time.sleep(2 * (attempt + 1))
    return {"job_id": job["job_id"], "model": model, "error": last}


def main() -> None:
    p = argparse.ArgumentParser(description="Answer benchmark jobs with one or more models.")
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--configs", nargs="*", help="only these configurations (default: all)")
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--limit", type=int, help="only the first N jobs per model (smoke test)")
    p.add_argument("--extra", type=json.loads, default={},
                   help='extra request fields as JSON, e.g. \'{"thinking":{"type":"disabled"}}\'')
    args = p.parse_args()

    load_env(args.env_file)
    base_url = os.environ.get("EE6405_LLM_BASE_URL", "")
    key = os.environ.get("EE6405_LLM_API_KEY", "")
    if not base_url or not key:
        raise SystemExit("EE6405_LLM_BASE_URL / EE6405_LLM_API_KEY are not set")

    jobs = [json.loads(line) for line in args.jobs.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.configs:
        jobs = [j for j in jobs if j["config"] in args.configs]
    if args.limit:
        jobs = jobs[: args.limit]

    done = set()
    if args.out.is_file():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if "error" not in row:
                done.add((row["job_id"], row["model"]))

    todo = [(j, m) for m in args.models for j in jobs if (j["job_id"], m) not in done]
    print(f"endpoint {base_url} | {len(jobs)} jobs x {len(args.models)} models | "
          f"{len(done)} already answered, {len(todo)} to run", flush=True)

    lock = threading.Lock()
    counts = {"ok": 0, "error": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        def work(pair):
            row = run_one(pair[0], pair[1], args, base_url, key)
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                counts["error" if "error" in row else "ok"] += 1
                n = counts["ok"] + counts["error"]
                if n % 50 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} ok={counts['ok']} error={counts['error']}", flush=True)
                if "error" in row and counts["error"] <= 5:
                    print(f"  ! {row['model']} {row['job_id']}: {row['error']}", flush=True)
        list(pool.map(work, todo))
    print(f"done: ok={counts['ok']} error={counts['error']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
