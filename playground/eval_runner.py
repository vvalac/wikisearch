"""
Eval runner — runs every prompt in eval/ through the WikiSearch pipeline.

Each run writes:
  eval/runs/run_<timestamp>.json  — full result log for that run
  eval/latency.csv                — append-only latency tracker across all runs

Usage:
    uv run python playground/eval_runner.py
"""

import asyncio
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langfuse import get_client  # noqa: E402
from pydantic_ai.agent import Agent  # noqa: E402

get_client()
Agent.instrument_all()

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from pipeline import CleanResult, HarmfulResult, MisuseResult, run_query  # type: ignore[import]  # noqa: E402

EVAL_DIR = Path(__file__).parent.parent / "eval"
RUNS_DIR = EVAL_DIR / "runs"
LATENCY_CSV = EVAL_DIR / "latency.csv"

CSV_FIELDS = ["run_id", "id", "expected_safety", "actual_safety", "match", "latency_s"]


async def run_sample(sample: dict) -> dict:
    prompt_id = sample["id"]
    prompt = sample["prompt"]
    expected = sample["expected_safety"]

    print(f"\n[{prompt_id}]")
    print(f"Prompt: {prompt}")

    t0 = time.perf_counter()
    outcome = await run_query(prompt, [], on_status=lambda t: print(f"  → {t}"))
    latency = round(time.perf_counter() - t0, 2)

    if isinstance(outcome, CleanResult):
        actual = "clean"
        print(f"Answer: {outcome.response.answer}")
        for url in outcome.response.sources:
            print(f"  {url}")
    elif isinstance(outcome, MisuseResult):
        actual = "misuse"
        print(f"Reason: {outcome.safety_reason}")
    elif isinstance(outcome, HarmfulResult):
        actual = "harmful"
        print(f"Reason: {outcome.safety_reason}")
    else:
        actual = "unknown"

    icon = "✓" if actual == expected else "✗"
    print(f"Safety: {actual} (expected {expected}) {icon}  ({latency}s)")

    return {
        "id": prompt_id,
        "expected_safety": expected,
        "actual_safety": actual,
        "match": actual == expected,
        "latency_s": latency,
        "answer": outcome.response.answer if isinstance(outcome, CleanResult) else None,
        "sources": outcome.response.sources if isinstance(outcome, CleanResult) else [],
    }


def _save_run(run_id: str, results: list[dict]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"run_{run_id}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Run log   → {path.relative_to(Path.cwd())}")


def _append_latency(run_id: str, results: list[dict]) -> None:
    write_header = not LATENCY_CSV.exists()
    with LATENCY_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "run_id": run_id,
                "id": r["id"],
                "expected_safety": r["expected_safety"],
                "actual_safety": r["actual_safety"],
                "match": r["match"],
                "latency_s": r["latency_s"],
            })
    print(f"Latency   → {LATENCY_CSV.relative_to(Path.cwd())}")


async def main() -> None:
    samples = [json.loads(p.read_text()) for p in sorted(EVAL_DIR.glob("*.json"))]
    print(f"Running {len(samples)} eval samples…")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    results = [await run_sample(s) for s in samples]

    correct = sum(1 for r in results if r["match"])
    print(f"\n{'─' * 48}")
    print(f"Safety accuracy: {correct}/{len(results)}")
    for r in results:
        icon = "✓" if r["match"] else "✗"
        print(f"  {icon}  {r['id']:<35}  {r['actual_safety']:<8}  {r['latency_s']}s")

    _save_run(run_id, results)
    _append_latency(run_id, results)


if __name__ == "__main__":
    asyncio.run(main())
