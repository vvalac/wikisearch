"""
Eval runner — runs every prompt in eval/ through the WikiSearch pipeline.

Each run writes:
  eval/runs/run_<timestamp>.json  — full result log for that run
  eval/latency.csv                — append-only latency tracker across all runs

Usage:
    uv run python playground/eval_runner.py            # run all prompts
    uv run python playground/eval_runner.py harms      # harmful + misuse only
    uv run python playground/eval_runner.py clean      # clean prompts only
"""

import argparse
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

from pipeline import Checkpoint, CleanResult, HarmfulResult, MisuseResult, run_query  # type: ignore[import]  # noqa: E402

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

    outcome: CleanResult | MisuseResult | HarmfulResult | None = None
    t0 = time.perf_counter()
    async for event in run_query(prompt, []):
        if isinstance(event, Checkpoint):
            print(f"  → {event.message}")
        else:
            outcome = event
    latency = round(time.perf_counter() - t0, 2)

    if isinstance(outcome, CleanResult):
        actual = "clean"
        print(f"Answer: {outcome.response.answer}")  # type: ignore[union-attr]
        for url in outcome.response.sources: # type: ignore[union-attr]
            print(f"  {url}")
    elif isinstance(outcome, MisuseResult):
        actual = "misuse"
        print(f"Reason: {outcome.safety_reason}") # type: ignore[union-attr]
    elif isinstance(outcome, HarmfulResult):
        actual = "harmful"
        print(f"Reason: {outcome.safety_reason}") # type: ignore[union-attr]
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
        "answer": outcome.response.answer if isinstance(outcome, CleanResult) else None, # type: ignore[union-attr]
        "sources": outcome.response.sources if isinstance(outcome, CleanResult) else [], # type: ignore[union-attr]
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


_SUBSET_FILTERS: dict[str, set[str]] = {
    "all":    {"clean", "misuse", "harmful"},
    "harms":  {"misuse", "harmful"},
    "clean":  {"clean"},
}


async def main() -> None:
    parser = argparse.ArgumentParser(description="WikiSearch eval runner")
    parser.add_argument(
        "subset",
        nargs="?",
        default="all",
        choices=_SUBSET_FILTERS,
        help="Which prompts to run (default: all)",
    )
    args = parser.parse_args()

    allowed = _SUBSET_FILTERS[args.subset]
    all_samples = [json.loads(p.read_text()) for p in sorted(EVAL_DIR.glob("*.json"))]
    samples = [s for s in all_samples if s["expected_safety"] in allowed]

    print(f"Running eval subset='{args.subset}' — {len(samples)} prompts…")

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
