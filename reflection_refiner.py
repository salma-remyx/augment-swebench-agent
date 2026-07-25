#!/usr/bin/env python3
"""Reflection Refiner CLI Tool.

Companion to ``majority_vote_ensembler.py``. Where the ensembler SELECTS one of
the N candidate diffs for a problem, this tool REFLECTS across the whole
candidate history (diffs + their verified eval outcomes) and emits one refined
candidate via final-round generation. It reads the same JSONL the ensembler
reads (produced by ``run_agent_on_swebench_problem.py``), so the two can be
A/B-tested on the same candidate set.

The reflection + final-round-generation mechanism is adapted from *PhoenixRepair*
(arXiv:2607.18859); see ``utils/solution_refiner.py``.

To see example input, see `example_ensembler_data.jsonl`.
"""

import argparse
import concurrent.futures
import json
import os
import sys
from typing import Any, List

try:
    from tqdm import tqdm
except ImportError:  # tqdm is an undeclared transitive dep (via datasets); keep
    # the CLI usable when it is absent by passing the iterable through untouched.

    def tqdm(iterable: Any = None, **kwargs: Any) -> Any:
        return iterable


from utils.solution_refiner import partition_by_eval, refine


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Reflection Refiner CLI Tool")
    parser.add_argument(
        "input_jsonl_path",
        type=str,
        help="Path to a JSONL file containing problems, candidate diffs, and "
        "eval_outcomes (same format as majority_vote_ensembler.py)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads for parallel processing (default: 8)",
    )
    return parser.parse_args()


def load_problems(json_path: str) -> List[dict[str, Any]]:
    """Load problems from a JSONL file."""
    try:
        data = []
        with open(json_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)


def process_problem(
    problem: dict[str, Any], problem_index: int, total_problems: int
) -> dict[str, Any]:
    """Reflect across one problem's candidates and emit a refined candidate."""
    print(
        f"Processing problem {problem_index + 1}/{total_problems}: "
        f"{problem.get('id', f'Problem {problem_index + 1}')}"
    )

    instruction = problem.get("instruction", "")
    diffs = problem.get("diffs", [])
    eval_outcomes = problem.get("eval_outcomes", {})

    if not diffs:
        print(f"  Warning: No diffs found for problem {problem_index + 1}, skipping")
        return {
            "id": problem.get("id", f"Problem {problem_index + 1}"),
            "instruction": instruction,
            "error": "No diffs provided",
            "refined_diff": None,
            "source": "no_candidate",
        }

    result = refine(instruction, diffs, eval_outcomes)
    passing, _failing = partition_by_eval(diffs, eval_outcomes)

    print(f"  Source: {result['source']} (passing candidates: {len(passing)})")
    return {
        "id": problem.get("id", f"Problem {problem_index + 1}"),
        "instruction": instruction,
        "insights": result["insights"],
        "refined_diff": result["refined_diff"],
        "source": result["source"],
        # The refined diff is freshly generated and not re-evaluated here; report
        # whether any INPUT candidate was verified passing (the strongest
        # anchor), not whether the new patch passes.
        "best_anchor_eval_success": len(passing) > 0,
    }


def refine_problems(
    problems: List[dict[str, Any]], num_workers: int = 8
) -> List[dict[str, Any]]:
    """Refine problems using a thread pool for parallel processing."""
    effective_workers = min(num_workers, len(problems))
    print(
        f"Processing {len(problems)} problems using {effective_workers} worker threads"
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=effective_workers
    ) as executor:
        problem_data = [
            (problem, i, len(problems)) for i, problem in enumerate(problems)
        ]
        results = list(
            tqdm(
                executor.map(lambda x: process_problem(*x), problem_data),
                total=len(problems),
                desc="Processing problems",
            )
        )
    return results


def main():
    """Main function."""
    args = parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set")
        sys.exit(1)

    problems = load_problems(args.input_jsonl_path)
    output_path = args.output_path or "refiner_results.json"

    results = refine_problems(problems, num_workers=args.workers)

    generated = sum(1 for r in results if r.get("source") == "generated")
    fallback = sum(1 for r in results if r.get("source") == "best_pass_fallback")
    print(
        f"Refined candidates: {generated} generated, {fallback} "
        "verified-passing fallback"
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
