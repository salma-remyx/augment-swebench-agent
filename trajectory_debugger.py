#!/usr/bin/env python3
"""Trajectory debugger CLI -- post-hoc critical-error attribution.

Reads the ``agent_logs.txt`` artifact a failed SWE-bench rollout already
produces (see ``run_agent_on_swebench_problem.py``) plus, optionally, its
``is_success`` verdict, and prints the earliest decisive error responsible
for the final failure -- the actionable feedback "TrajDebug" (arXiv
2608.06346) shows improves downstream agent success.

Mirrors the analysis-CLI shape of ``majority_vote_ensembler.py``: a
standalone script over logged rollout artifacts rather than a change to the
agent's runtime loop.

Examples
--------
    python trajectory_debugger.py path/to/rollout/agent_logs.txt
    python trajectory_debugger.py path/to/rollout/ --eval-json augment-agent.ID.json
    python trajectory_debugger.py logs.txt --no-is-success --output report.json
"""

import argparse
import json
import sys
from pathlib import Path

from utils.trajectory_error_attribution import (
    analyze_trajectory,
    format_report,
    report_to_dict,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attribute the critical failure in a failed agent trajectory."
    )
    parser.add_argument(
        "logs_path",
        type=str,
        help="Path to agent_logs.txt (or a rollout dir containing it).",
    )
    parser.add_argument(
        "--eval-json",
        type=str,
        default=None,
        help="Optional eval-result JSON providing the is_success verdict "
        "(reads an 'is_success' bool or a 'resolved_ids' list).",
    )
    parser.add_argument(
        "--is-success",
        "--no-is-success",
        dest="is_success",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly set the rollout verdict. Default: unknown (treated as "
        "failed for attribution).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the JSON attribution report.",
    )
    return parser.parse_args(argv)


def _resolve_logs_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "agent_logs.txt"
    if not candidate.is_file():
        raise FileNotFoundError(f"agent_logs.txt not found at {candidate}")
    return candidate


def _load_is_success(eval_json: str) -> bool | None:
    data = json.loads(Path(eval_json).read_text())
    if isinstance(data, dict):
        if "is_success" in data:
            return bool(data["is_success"])
        if "resolved_ids" in data:
            return bool(data["resolved_ids"])
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        logs_file = _resolve_logs_path(args.logs_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    is_success = args.is_success
    if is_success is None and args.eval_json:
        is_success = _load_is_success(args.eval_json)

    log_text = logs_file.read_text()
    report = analyze_trajectory(log_text, is_success=is_success)

    print(format_report(report))
    if args.output:
        Path(args.output).write_text(json.dumps(report_to_dict(report), indent=2))
        print(f"\nReport written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
