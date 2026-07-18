#!/usr/bin/env python3
"""Majority Vote Ensembler CLI Tool

This tool takes a path to a JSONL file containing problems, each with a list of
candidate diffs. It then uses the ensembler prompt to generate prompts and submits them
to the specified LLM (Claude or OpenAI) for results.

To see example input, see `example_ensembler_data.jsonl`. To see example output,
run `python majority_vote_ensembler.py example_ensembler_data.jsonl --output_path example_ensembler_results.json`.
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
from typing import Dict, List, Any, Optional, Callable
from tqdm import tqdm

from prompts.ensembler_prompt import build_ensembler_prompt
from utils.llm_client import get_client, TextPrompt
from utils.solution_verifier import select as verifier_select

# Optional selector with the (instruction, candidates) -> index contract
# of utils.solution_verifier; replaces the single-shot o1 pick.
Selector = Callable[[str, List[str]], Optional[int]]

MAX_TOKENS = 16384
TEMPERATURE = 0.0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Majority Vote Ensembler CLI Tool")
    parser.add_argument(
        "input_jsonl_path",
        type=str,
        help="Path to a JSONL file containing problems and candidate diffs",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads for parallel processing (default: 4)",
    )
    parser.add_argument(
        "--verifier",
        action="store_true",
        help="Rank candidates with the LLM-as-a-Verifier scorer (Claude Sonnet 4) "
        "instead of the single-shot o1 majority vote",
    )
    parser.add_argument(
        "--verifier-samples",
        type=int,
        default=8,
        help="Repeated-evaluation count (n_evaluations) for --verifier: score "
        "each directed pair this many times (with sampling) and average to "
        "reduce scoring variance (default: 8, matching the paper)",
    )
    return parser.parse_args()


def load_problems(json_path: str) -> List[Dict[str, Any]]:
    """Load problems from a JSON file."""
    try:
        data = []
        with open(json_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)


def extract_solution_index(response_text: str) -> Optional[int]:
    """Extract the solution index from the model's response."""
    pattern = r"<solution_index>(\d+)</solution_index>"
    match = re.search(pattern, response_text)
    if match:
        return int(match.group(1)) - 1
    return None


def _eval_success(eval_outcomes: Any, solution_index: Optional[int]) -> bool:
    """Safely look up whether the selected candidate passed its eval."""
    if solution_index is None or not isinstance(eval_outcomes, list):
        return False
    if not 0 <= solution_index < len(eval_outcomes):
        return False
    return bool(eval_outcomes[solution_index].get("is_success", False))


def process_problem(
    problem: Dict[str, Any],
    problem_index: int,
    total_problems: int,
    selector: Optional[Selector] = None,
) -> Dict[str, Any]:
    """Process a single problem using the LLM.

    Args:
        problem: The problem to process
        problem_index: The index of the problem (for logging)
        total_problems: The total number of problems (for logging)
        selector: Optional verifier selector (see utils.solution_verifier).

    Returns:
        A dictionary containing the result for the problem
    """
    print(
        f"Processing problem {problem_index + 1}/{total_problems}: {problem.get('id', f'Problem {problem_index + 1}')}"
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
            "selected_diff_index": None,
            "selected_diff": None,
        }

    # Verifier path: pick via a pairwise pivot tournament instead of a discrete o1 pick.
    if selector is not None:
        solution_index = selector(instruction, diffs)
        if solution_index is not None and not 0 <= solution_index < len(diffs):
            print(
                f"  Warning: verifier returned invalid index {solution_index} "
                f"for problem {problem_index + 1}"
            )
            solution_index = None
        selected_diff = diffs[solution_index] if solution_index is not None else None
        return {
            "id": problem.get("id", f"Problem {problem_index + 1}"),
            "instruction": instruction,
            "response": "",
            "selected_diff_index": solution_index,
            "selected_diff": selected_diff,
            "is_eval_success": _eval_success(eval_outcomes, solution_index),
        }

    # Majority-vote path: create the o1 client lazily so the verifier path
    # (which uses an Anthropic client via the selector) does not require an
    # OpenAI API key.
    client = get_client("openai-direct", model_name="o1-2024-12-17", cot_model=True)

    # Build the ensembler prompt
    prompt = build_ensembler_prompt(instruction, diffs)

    # Prepare the message for the LLM
    messages = [[TextPrompt(text=prompt)]]

    # Submit to the LLM
    try:
        response, metadata = client.generate(
            messages=messages,  # type: ignore
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        # Extract the response text
        response_text = (
            response[0].text if hasattr(response[0], "text") else str(response[0])  # pyright: ignore[reportAttributeAccessIssue]
        )

        # Extract the solution index
        solution_index = extract_solution_index(response_text)

        if solution_index is not None and 0 <= solution_index < len(diffs):
            selected_diff = diffs[solution_index]  # Convert to 0-indexed
        else:
            selected_diff = None
            print(
                f"  Warning: Invalid solution index {solution_index} for problem {problem_index + 1}"
            )

        result = {
            "id": problem.get("id", f"Problem {problem_index + 1}"),
            "instruction": instruction,
            "response": response_text,
            "selected_diff_index": solution_index,
            "selected_diff": selected_diff,
            "is_eval_success": eval_outcomes[solution_index]["is_success"],
        }

        print(f"  Selected solution index: {solution_index}")
        return result

    except Exception as e:
        print(f"  Error processing problem {problem_index + 1}: {e}")
        return {
            "id": problem.get("id", f"Problem {problem_index + 1}"),
            "instruction": instruction,
            "error": str(e),
            "selected_diff_index": None,
            "selected_diff": None,
            "is_eval_success": False,
        }


def ensemble_problems(
    problems: List[Dict[str, Any]],
    num_workers: int = 8,
    selector: Optional[Selector] = None,
) -> List[Dict[str, Any]]:
    """Ensemble problems using a thread pool for parallel processing.

    Args:
        problems: List of problems to process
        num_workers: Number of worker threads to use
        selector: Optional verifier selector forwarded to ``process_problem``

    Returns:
        List of results for each problem
    """
    # Adjust number of workers based on the number of problems
    effective_workers = min(num_workers, len(problems))

    print(
        f"Processing {len(problems)} problems using {effective_workers} worker threads"
    )

    # Create a thread pool and process problems in parallel
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=effective_workers
    ) as executor:
        # Create a list of (problem, index, total) tuples to pass to the worker function
        problem_data = [
            (problem, i, len(problems)) for i, problem in enumerate(problems)
        ]

        # Map the worker function over the problems with tqdm progress bar
        # This preserves the order of the results
        results = list(
            tqdm(
                executor.map(
                    lambda x: process_problem(*x, selector=selector), problem_data
                ),
                total=len(problems),
                desc="Processing problems",
            )
        )

    return results


def main():
    """Main function."""
    args = parse_args()

    # --verifier selects candidates via LLM-as-a-Verifier scoring (Claude
    # Sonnet 4); otherwise fall back to the single-shot o1 majority vote.
    if args.verifier:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY environment variable is not set")
            sys.exit(1)
    elif not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    selector: Optional[Selector] = None
    if args.verifier:

        def _verifier_selector(
            instruction: str, candidates: List[str]
        ) -> Optional[int]:
            return verifier_select(
                instruction, candidates, n_evaluations=args.verifier_samples
            )

        selector = _verifier_selector

    # Load problems from JSON file
    problems = load_problems(args.input_jsonl_path)

    # Determine output path
    output_path = args.output_path or "ensembler_results.json"

    # Ensemble problems using thread pool
    results = ensemble_problems(problems, num_workers=args.workers, selector=selector)

    # get success rate
    success_rate = sum([result["is_eval_success"] for result in results]) / len(results)
    print(f"Success rate: {success_rate:.2f}")

    # Save results to output file in JSONL format
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
