"""Evidence gate for the agent's DONE lifecycle transition.

Adapted from "Proof-or-Stop: Don't Trust the Agent, Trust the Evidence -- Loop
Engineering for Verifiable Evidence-Gated Lifecycle Control"
(arXiv:2607.14890). The paper's central claim: an agent's ``DONE`` is an
unverified *claim*, and the lifecycle may act on it only when fresh,
tracked-source-state-bound, mechanically verifiable evidence satisfies the
gate. This repo's weakest link is exactly that -- the system prompt merely
*suggests* running tests and ``CompleteTool`` honors stop with zero
verification, a classic premature-DONE SWE-bench failure mode.

This module keeps the paper's mechanism intact -- treat the transition as a
claim, require mechanical proof, refuse the transition otherwise -- while
substituting the paper's auxiliary machinery (control-policy ablation, receipt
bundles / tamper classes, the separate benchmark suite) with target-native
evidence that the agent *already produces*:

  * In-loop evidence -- the agent runs the test suite with its ``bash`` tool;
    the gate scans those tool results for a fresh passing test run, and refuses
    DONE when the evidence is absent, failing, or *stale* (source was edited
    after the last passing run). "Stale" operationalizes the paper's
    "tracked-source-state-bound" requirement: the proof must reflect the
    *current* source.
  * Report evidence -- ``evaluate_swebench_report`` reads the
    ``FAIL_TO_PASS``/``PASS_TO_PASS`` ``report.json`` that ``run_evaluation``
    emits, so the same gate notion can be applied to a finished rollout.

Preserved from the paper:
  * DONE-as-claim -- the gate never trusts the model's assertion that it is
    done; it trusts only mechanically verifiable evidence.
  * Freshness / source-state binding -- evidence from before the most recent
    source mutation is rejected (re-run the tests).
  * Stop over false-DONE -- when the gate refuses, the lifecycle does NOT
    transition; control returns to the agent with a directive to produce proof.

Substituted (Mode 2):
  * The paper's learned control policy / ablation harness is not ported; the
    gate is a parameter-free mechanical check over evidence already in the
    dialog.
  * The paper's receipt-bundle tamper rejection is out of scope for a single
    agent loop; this is the gate, not the receipt layer.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from utils.common import DialogMessages
from utils.llm_client import ToolCall, ToolFormattedResult

BASH_TOOL_NAME = "bash"
STR_REPLACE_TOOL_NAME = "str_replace_editor"

# Mechanically-verifiable signatures pulled from pytest / unittest summaries.
# These are deliberately structural (counts, banners) rather than free-text so
# the agent cannot satisfy the gate by merely asserting "tests passed".
_FAIL_SIGNALS = (
    re.compile(r"\b\d+\s+failed\b"),
    re.compile(r"\bFAILED\b"),
    re.compile(r"\b\d+\s+errors?\b"),
    re.compile(r"={3,}\s*ERRORS?\s*={3,}"),
)
_PASS_SIGNALS = (
    re.compile(r"\b\d+\s+passed\b"),
    re.compile(r"\bRan \d+ tests?[\s\S]{0,160}?\nOK\b"),
    re.compile(r"\ball tests? passed\b", re.IGNORECASE),
)

# A bash command that mutates tracked source, so test evidence from before it
# no longer reflects the current code. Conservative by design: the paper's bias
# is toward Stop (re-run) over false-DONE, so over-flagging re-runs is cheap.
_BASH_WRITE_RE = re.compile(
    r">>"
    r"|&>"
    r"|(?<![&>])>(?!&)"
    r"|(?<![A-Za-z0-9])(?:sed\s+-i|patch\b|git\s+apply|tee\b|cp\b|mv\b|rm\b|rsync\b)"
)


@dataclass
class GateVerdict:
    """Outcome of an evidence-gate check.

    Attributes:
        admissible: True iff mechanically-verifiable evidence satisfies the
            gate and the lifecycle transition may proceed.
        reason: Human-readable explanation of the decision (shown to the agent
            when the gate refuses).
        evidence: The specific evidence items (test snippets / test names) the
            decision rested on.
    """

    admissible: bool
    reason: str
    evidence: list[str] = field(default_factory=list)


# A gate is a function from the dialog (and nothing else) to a verdict, so the
# ``CompleteTool`` can call it without changing its contract.
EvidenceGate = Callable[[Optional[DialogMessages]], GateVerdict]


def _classify_test_output(text: str) -> Optional[str]:
    """Classify a single tool output as a test run.

    Returns ``"pass"`` for a clean passing run, ``"fail"`` for a run with any
    failure/error, or ``None`` if the output does not look like a test run at
    all. A run with both pass and failure signals counts as ``"fail"``.
    """
    has_fail = any(sig.search(text) for sig in _FAIL_SIGNALS)
    has_pass = any(sig.search(text) for sig in _PASS_SIGNALS)
    if not (has_fail or has_pass):
        return None
    return "fail" if has_fail else "pass"


def _is_source_mutation(tool_name: str, tool_input: Any) -> bool:
    """Whether a tool call mutates tracked source (invalidating prior evidence)."""
    if tool_name == STR_REPLACE_TOOL_NAME:
        return True
    if tool_name == BASH_TOOL_NAME and isinstance(tool_input, dict):
        return bool(_BASH_WRITE_RE.search(str(tool_input.get("command", ""))))
    return False


def _collect_tool_events(
    dialog: DialogMessages,
) -> list[tuple[str, Any, str]]:
    """Chronological ``(tool_name, tool_input, tool_output)`` for every tool call.

    Assistant turns carry ``ToolCall`` blocks; the following user turn carries
    the matching ``ToolFormattedResult`` blocks. They are paired in order.
    """
    events: list[tuple[str, Any, str]] = []
    pending_calls: list[ToolCall] = []
    for turn in dialog.get_message_lists():
        results = [m for m in turn if isinstance(m, ToolFormattedResult)]
        if results and pending_calls:
            for call, result in zip(pending_calls, results):
                events.append((call.tool_name, call.tool_input, result.tool_output))
            pending_calls = []
        calls = [m for m in turn if isinstance(m, ToolCall)]
        if calls:
            pending_calls = calls
    return events


def evaluate_dialog(dialog: Optional[DialogMessages]) -> GateVerdict:
    """Gate the DONE transition on fresh passing test-run evidence in the dialog.

    Admissible iff there is a passing test run AND no source mutation and no
    later failing test run after it -- i.e. the most recent test state is a
    clean pass over the current source.
    """
    if dialog is None:
        return GateVerdict(False, "no dialog to inspect for evidence", [])

    events = _collect_tool_events(dialog)

    last_pass_idx: Optional[int] = None
    for i, (tool_name, _tool_input, tool_output) in enumerate(events):
        if tool_name != BASH_TOOL_NAME:
            continue
        classification = _classify_test_output(tool_output)
        if classification == "pass":
            last_pass_idx = i
        elif classification == "fail":
            # A later failing run supersedes earlier passing evidence.
            last_pass_idx = None

    if last_pass_idx is None:
        return GateVerdict(
            False,
            "no mechanically-verifiable passing test run was found. Run the "
            "project's test suite (or verification command) with the bash tool "
            "and let it finish before completing.",
            [],
        )

    for tool_name, tool_input, _tool_output in events[last_pass_idx + 1 :]:
        if _is_source_mutation(tool_name, tool_input):
            return GateVerdict(
                False,
                "tracked source was edited after the last passing test run, so "
                "that evidence no longer reflects the current code. Re-run the "
                "tests and let them finish before completing.",
                [],
            )

    _name, _tool_input, tool_output = events[last_pass_idx]
    snippet = tool_output.strip().replace("\n", " ")[:160]
    return GateVerdict(True, "fresh passing test-run evidence present", [snippet])


def _status_passed(status: Any) -> bool:
    """Whether a single SWE-bench test status counts as passing."""
    if isinstance(status, bool):
        return status
    return str(status).strip().upper() in {"SUCCESS", "PASSED", "PASS", "TRUE", "1"}


_INSTANCE_MARKERS = ("fail_to_pass", "pass_to_pass", "tests_status")


def _select_instance(report: Any) -> Optional[dict[str, Any]]:
    """Pull the per-instance test-status object out of a SWE-bench report.

    Tolerates a bare instance object, a top-level ``{instance_id: {...}}`` map,
    and missing sections. An instance object is recognized by either the
    normalized lowercase sections or the native ``tests_status`` block.
    """
    if not isinstance(report, dict):
        return None
    if any(marker in report for marker in _INSTANCE_MARKERS):
        return report
    for value in report.values():
        if isinstance(value, dict) and any(
            marker in value for marker in _INSTANCE_MARKERS
        ):
            return value
    return None


def _status_sections(
    instance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize either report flavor to ``(fail_to_pass, pass_to_pass)`` maps.

    Accepts the normalized lowercase shape (``{"fail_to_pass": {name: status}}``)
    and the native SWE-bench ``tests_status`` shape emitted by
    ``run_evaluation``, where each uppercase section holds ``success`` /
    ``failure`` *lists* of test names rather than a name-to-status map.
    """
    if "fail_to_pass" in instance or "pass_to_pass" in instance:
        return (
            instance.get("fail_to_pass") or {},
            instance.get("pass_to_pass") or {},
        )
    tests_status = instance.get("tests_status")
    if not isinstance(tests_status, dict):
        return {}, {}
    fail_to_pass: dict[str, Any] = {}
    pass_to_pass: dict[str, Any] = {}
    for section_key, statuses in (
        ("FAIL_TO_PASS", fail_to_pass),
        ("PASS_TO_PASS", pass_to_pass),
    ):
        section = tests_status.get(section_key)
        if not isinstance(section, dict):
            continue
        for name in section.get("success") or []:
            statuses[str(name)] = "SUCCESS"
        for name in section.get("failure") or []:
            statuses[str(name)] = "FAILED"
    return fail_to_pass, pass_to_pass


def evaluate_swebench_report(report: Any) -> GateVerdict:
    """Gate on a SWE-bench ``report.json`` (the post-run evidence stream).

    Admissible iff every ``FAIL_TO_PASS`` and ``PASS_TO_PASS`` test is reported
    as passing. This is the same gate notion applied to ``run_evaluation``'s
    structured output rather than to in-dialog test runs. Both the normalized
    lowercase shape and the native ``tests_status`` report shape are accepted.
    """
    instance = _select_instance(report)
    if instance is None:
        return GateVerdict(False, "report contains no test status to verify", [])

    fail_to_pass, pass_to_pass = _status_sections(instance)
    if not fail_to_pass and not pass_to_pass:
        return GateVerdict(False, "report contains no tests to verify", [])

    failed = [
        name
        for section in (fail_to_pass, pass_to_pass)
        for name, status in section.items()
        if not _status_passed(status)
    ]
    if failed:
        return GateVerdict(
            False,
            f"report shows {len(failed)} test(s) not passing "
            f"(across FAIL_TO_PASS / PASS_TO_PASS)",
            failed,
        )
    return GateVerdict(
        True,
        "all FAIL_TO_PASS and PASS_TO_PASS tests passed",
        list(fail_to_pass) + list(pass_to_pass),
    )


def format_stop_message(verdict: GateVerdict) -> str:
    """The directive returned to the agent when the gate refuses its DONE claim."""
    return (
        "COMPLETION BLOCKED (Proof-or-Stop): your 'complete' call is a claim, "
        "not evidence. Before completing, produce mechanically-verifiable "
        "evidence that the task is actually done -- run the relevant test suite "
        "(or the project's verification command) with your bash tool and let it "
        "finish, then call 'complete' again.\n\n"
        f"Reason this claim was not admitted: {verdict.reason}"
    )
