#!/usr/bin/env python3
"""Adaptive best-of-N ensembler (agreement-based sampling budget).

Reads the same JSONL of problems + candidate diffs + ``eval_outcomes`` that
``majority_vote_ensembler.py`` consumes and, for each problem, reports the
adaptive sampling budget prescribed by agreement-based best-of-N
(arXiv:2509.21091): the smallest number of rollouts at which the candidate
answers agreed, and how many of the generated rollouts that saves. Also writes
the agreement-selected diff per problem, in the ensembler's output shape so the
two strategies stay comparable.

No LLM calls -- selection is parameter-free (majority vote on normalized patch
content). For the o1 single-shot vote or the LLM-as-a-Verifier pairwise
tournament, use ``majority_vote_ensembler.py``.

To see example input, see ``example_ensembler_data.jsonl``. Example::

    python adaptive_best_of_n_ensembler.py example_ensembler_data.jsonl \\
        --output_path adaptive_results.json
"""

import argparse
import json
import sys
from typing import Any

from utils.adaptive_best_of_n import (
    DEFAULT_AGREEMENT_THRESHOLD,
    DEFAULT_MIN_SAMPLES,
    adaptive_n,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Adaptive best-of-N ensembler (agreement-based budget)"
    )
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
        "--agreement-threshold",
        type=float,
        default=DEFAULT_AGREEMENT_THRESHOLD,
        help="Leader must hold strictly more than this share of the votes seen "
        "so far for the adaptive stopping rule to fire (default: %(default)s, "
        "a strict majority)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Smallest prefix the stopping rule may commit on; agreement is "
        "undefined for a single sample (default: %(default)s)",
    )
    return parser.parse_args()


def load_problems(json_path: str) -> list[dict[str, Any]]:
    """Load problems from a JSONL file."""
    data: list[dict[str, Any]] = []
    try:
        with open(json_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    except Exception as e:  # noqa: BLE001 -- CLI: report and exit on bad input
        print(f"Error loading JSON file: {e}")
        sys.exit(1)


def _eval_success(eval_outcomes: Any, solution_index: int) -> bool:
    """Safely look up whether the selected candidate passed its eval."""
    if not isinstance(eval_outcomes, list):
        return False
    if not 0 <= solution_index < len(eval_outcomes):
        return False
    return bool(eval_outcomes[solution_index].get("is_success", False))


def process_problem(
    problem: dict[str, Any],
    problem_index: int,
    total_problems: int,
    agreement_threshold: float,
    min_samples: int,
) -> dict[str, Any]:
    """Run adaptive best-of-N analysis on a single problem."""
    pid = problem.get("id", f"Problem {problem_index + 1}")
    instruction = problem.get("instruction", "")
    diffs = problem.get("diffs", [])
    eval_outcomes = problem.get("eval_outcomes", {})

    print(f"Processing problem {problem_index + 1}/{total_problems}: {pid}")

    if not diffs:
        print(f"  Warning: No diffs found for {pid}, skipping")
        return {
            "id": pid,
            "instruction": instruction,
            "error": "No diffs provided",
            "selected_diff_index": None,
            "selected_diff": None,
        }

    decision = adaptive_n(
        diffs,
        eval_outcomes,
        agreement_threshold=agreement_threshold,
        min_samples=min_samples,
    )
    assert decision is not None  # diffs is non-empty above

    idx = decision.selected_index
    print(
        f"  {decision.adaptive_n}/{decision.n} rollouts needed "
        f"({decision.rollouts_saved} saved); agreement_reached="
        f"{decision.agreement_reached}"
    )

    return {
        "id": pid,
        "instruction": instruction,
        "selected_diff_index": idx,
        "selected_diff": diffs[idx],
        "is_eval_success": _eval_success(eval_outcomes, idx),
        "n": decision.n,
        "adaptive_n": decision.adaptive_n,
        "rollouts_saved": decision.rollouts_saved,
        "leader_share": decision.leader_share,
        "n_successful": decision.n_successful,
        "agreement_reached": decision.agreement_reached,
        "matches_full_n": decision.matches_full_n,
    }


def main() -> None:
    """Main function."""
    args = parse_args()
    problems = load_problems(args.input_jsonl_path)
    output_path = args.output_path or "adaptive_results.json"

    results = [
        process_problem(
            problem,
            i,
            len(problems),
            args.agreement_threshold,
            args.min_samples,
        )
        for i, problem in enumerate(problems)
    ]

    total_rollouts = sum(r.get("n", 0) for r in results)
    total_saved = sum(r.get("rollouts_saved", 0) for r in results)
    reached = sum(1 for r in results if r.get("agreement_reached"))
    if total_rollouts:
        print(
            f"Adaptive budget: {total_rollouts - total_saved}/{total_rollouts} "
            f"rollouts needed across {len(results)} problems "
            f"({total_saved} saved, {100 * total_saved / total_rollouts:.1f}%); "
            f"agreement reached on {reached}/{len(results)} problems."
        )

    success_rate = (
        sum(r["is_eval_success"] for r in results) / len(results) if results else 0.0
    )
    print(f"Selected-solution success rate: {success_rate:.2f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
