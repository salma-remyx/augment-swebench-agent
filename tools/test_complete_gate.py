"""Tests for the Proof-or-Stop evidence gate wired into the complete path.

These tests import the (non-new) call-site module ``tools.complete_tool`` plus
``utils.common`` / ``utils.llm_client`` to build a real ``DialogMessages``
populated with scripted tool calls, then exercise the wiring edit made to
``CompleteTool.run_impl`` -- the DONE claim is admitted only when the gate sees
fresh, mechanically-verifiable test-pass evidence. No real tools run and no API
calls are made.
"""

import logging
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from rich.console import Console

from tools.agent import Agent
from tools.complete_tool import CompleteTool
from utils.common import DialogMessages, ToolCallParameters, ToolImplOutput
from utils.evidence_gate import (
    evaluate_dialog,
    evaluate_swebench_report,
    format_stop_message,
)
from utils.llm_client import AssistantContentBlock, ToolCall
from utils.workspace_manager import WorkspaceManager

PYTEST_PASS = "============================= test session starts =============================\ncollected 5 items\n\ntests/test_x.py .....                                              [100%]\n\n============================== 5 passed in 1.23s ===============================\n"
PYTEST_FAIL = "collected 5 items\n\ntests/test_x.py ..F..                                              [100%]\n\n=================================== FAILURES ===================================\n\n=============================== 1 failed, 4 passed in 1.45s ===============================\n"
UNITTEST_PASS = "....\n----------------------------------------------------------------------\nRan 4 tests in 0.5s\n\nOK\n"
NON_TEST_OUTPUT = "src/utils.py  README.md  setup.py\n"


def _make_dialog(events: list[tuple[str, dict, str]]) -> DialogMessages:
    """Build a dialog from chronological ``(tool_name, tool_input, output)``."""
    dialog = DialogMessages(
        logger_for_agent_logs=logging.getLogger("test_complete_gate"),
        use_prompt_budgeting=False,
    )
    dialog.add_user_prompt("Fix the bug and verify your fix.")
    for i, (tool_name, tool_input, output) in enumerate(events):
        call = ToolCall(
            tool_call_id=f"c{i}", tool_name=tool_name, tool_input=tool_input
        )
        response: list[AssistantContentBlock] = [call]
        dialog.add_model_response(response)
        params = ToolCallParameters(
            tool_call_id=f"c{i}", tool_name=tool_name, tool_input=tool_input
        )
        dialog.add_tool_call_results([params], [output])
    return dialog


def _run_complete(tool: CompleteTool, dialog: DialogMessages) -> ToolImplOutput:
    """Invoke the gate hook exactly as the agent loop does."""
    return tool.run_impl({"answer": "I am done."}, dialog)


def test_no_gate_is_backward_compatible():
    """Without an injected gate, CompleteTool honors DONE as before."""
    tool = CompleteTool()
    dialog = _make_dialog([("bash", {"command": "ls"}, NON_TEST_OUTPUT)])
    result = _run_complete(tool, dialog)
    assert tool.should_stop is True
    assert tool.answer == "I am done."
    assert result.tool_output == "Task completed"


def test_gate_blocks_done_without_evidence():
    """No test run in the dialog -> DONE refused, agent loop must continue."""
    tool = CompleteTool(evidence_gate=evaluate_dialog)
    dialog = _make_dialog([("bash", {"command": "ls"}, NON_TEST_OUTPUT)])
    result = _run_complete(tool, dialog)
    assert tool.should_stop is False
    assert tool.answer == ""
    assert result.tool_output.startswith("COMPLETION BLOCKED (Proof-or-Stop)")


def test_gate_blocks_done_on_none_dialog():
    """A missing dialog is not admissible evidence."""
    tool = CompleteTool(evidence_gate=evaluate_dialog)
    result = tool.run_impl({"answer": "done"}, None)
    assert tool.should_stop is False
    assert "not admitted" in result.tool_output


def test_gate_admits_done_with_passing_pytest():
    tool = CompleteTool(evidence_gate=evaluate_dialog)
    dialog = _make_dialog([("bash", {"command": "pytest -q"}, PYTEST_PASS)])
    _run_complete(tool, dialog)
    assert tool.should_stop is True
    assert tool.answer == "I am done."


def test_gate_admits_done_with_passing_unittest():
    tool = CompleteTool(evidence_gate=evaluate_dialog)
    dialog = _make_dialog([("bash", {"command": "python -m unittest"}, UNITTEST_PASS)])
    _run_complete(tool, dialog)
    assert tool.should_stop is True


def test_gate_blocks_done_on_failing_tests():
    tool = CompleteTool(evidence_gate=evaluate_dialog)
    dialog = _make_dialog([("bash", {"command": "pytest -q"}, PYTEST_FAIL)])
    result = _run_complete(tool, dialog)
    assert tool.should_stop is False
    assert tool.answer == ""
    assert "not admitted" in result.tool_output


def test_gate_blocks_done_on_stale_evidence():
    """A source edit after the last passing run makes the evidence stale."""
    tool = CompleteTool(evidence_gate=evaluate_dialog)
    dialog = _make_dialog(
        [
            ("bash", {"command": "pytest -q"}, PYTEST_PASS),
            (
                "str_replace_editor",
                {"path": "src/fix.py", "old_str": "a", "new_str": "b"},
                "The file src/fix.py has been edited.",
            ),
        ]
    )
    result = _run_complete(tool, dialog)
    assert tool.should_stop is False
    assert "edited after" in result.tool_output


def test_gate_blocks_done_on_bash_write_after_pass():
    """A file-writing bash command after a passing run is also a mutation."""
    tool = CompleteTool(evidence_gate=evaluate_dialog)
    dialog = _make_dialog(
        [
            ("bash", {"command": "pytest -q"}, PYTEST_PASS),
            ("bash", {"command": "echo 'x = 1' > src/conf.py"}, "done"),
        ]
    )
    result = _run_complete(tool, dialog)
    assert tool.should_stop is False
    assert "edited after" in result.tool_output


def test_gate_resets_when_a_later_run_fails():
    """A failing run after a passing one supersedes the passing evidence."""
    tool = CompleteTool(evidence_gate=evaluate_dialog)
    dialog = _make_dialog(
        [
            ("bash", {"command": "pytest -q"}, PYTEST_PASS),
            ("bash", {"command": "pytest -q"}, PYTEST_FAIL),
        ]
    )
    _run_complete(tool, dialog)
    assert tool.should_stop is False


def test_gate_admits_after_fixing_followed_by_rerun():
    """Fail then later pass -> the latest state is a clean pass -> admissible."""
    tool = CompleteTool(evidence_gate=evaluate_dialog)
    dialog = _make_dialog(
        [
            ("bash", {"command": "pytest -q"}, PYTEST_FAIL),
            ("bash", {"command": "pytest -q"}, PYTEST_PASS),
        ]
    )
    _run_complete(tool, dialog)
    assert tool.should_stop is True


def test_format_stop_message_directs_to_run_tests():
    verdict = evaluate_dialog(None)
    message = format_stop_message(verdict)
    assert "COMPLETION BLOCKED" in message
    assert "no dialog" in message


def test_report_admits_when_all_tests_pass():
    report = {
        "instance-1": {
            "fail_to_pass": {"tests/test_new.py::test_it": "SUCCESS"},
            "pass_to_pass": {"tests/test_old.py::test_it": "SUCCESS"},
        }
    }
    verdict = evaluate_swebench_report(report)
    assert verdict.admissible is True
    assert len(verdict.evidence) == 2


def test_report_rejects_when_fail_to_pass_fails():
    report = {
        "instance-1": {
            "fail_to_pass": {
                "tests/test_new.py::test_a": "SUCCESS",
                "tests/test_new.py::test_b": "FAILED",
            },
            "pass_to_pass": {"tests/test_old.py::test_it": "SUCCESS"},
        }
    }
    verdict = evaluate_swebench_report(report)
    assert verdict.admissible is False
    assert "tests/test_new.py::test_b" in verdict.evidence


def test_report_rejects_when_pass_to_pass_regresses():
    """Breaking existing functionality (PASS_TO_PASS) must also block."""
    report = {
        "instance-1": {
            "fail_to_pass": {"tests/test_new.py::test_it": "SUCCESS"},
            "pass_to_pass": {"tests/test_old.py::test_it": "FAILED"},
        }
    }
    verdict = evaluate_swebench_report(report)
    assert verdict.admissible is False


# The native SWE-bench report.json shape that run_evaluation actually emits:
# per-instance "tests_status" with success/failure *lists* of test names.
NATIVE_REPORT_PASS = {
    "instance-1": {
        "patch_is_None": False,
        "tests_status": {
            "FAIL_TO_PASS": {"success": ["tests/test_new.py::test_it"], "failure": []},
            "PASS_TO_PASS": {"success": ["tests/test_old.py::test_it"], "failure": []},
        },
    }
}

NATIVE_REPORT_REGRESSION = {
    "instance-1": {
        "patch_is_None": False,
        "tests_status": {
            "FAIL_TO_PASS": {"success": ["tests/test_new.py::test_it"], "failure": []},
            "PASS_TO_PASS": {"success": [], "failure": ["tests/test_old.py::test_it"]},
        },
    }
}


def test_report_admits_native_tests_status_shape():
    """The gate accepts the native report.json shape, not just the normalized one."""
    verdict = evaluate_swebench_report(NATIVE_REPORT_PASS)
    assert verdict.admissible is True
    assert "tests/test_new.py::test_it" in verdict.evidence
    assert "tests/test_old.py::test_it" in verdict.evidence


def test_report_rejects_native_pass_to_pass_regression():
    verdict = evaluate_swebench_report(NATIVE_REPORT_REGRESSION)
    assert verdict.admissible is False
    assert "tests/test_old.py::test_it" in verdict.evidence


def test_report_rejects_native_shape_with_no_tests():
    report = {"instance-1": {"tests_status": {}}}
    verdict = evaluate_swebench_report(report)
    assert verdict.admissible is False
    assert "no tests" in verdict.reason


def test_agent_wires_gate_when_enforced(tmp_path):
    """Agent wires the evidence gate into CompleteTool only when requested."""
    workspace = WorkspaceManager(root=Path(tmp_path))
    client = MagicMock()
    logger = logging.getLogger("test_agent_wiring")

    with patch("tools.agent.create_bash_tool") as mock_bash:
        mock_bash.return_value = MagicMock()
        enforced = Agent(
            client=client,
            workspace_manager=workspace,
            console=Console(),
            logger_for_agent_logs=logger,
            enforce_completion_gate=True,
        )
        plain = Agent(
            client=client,
            workspace_manager=workspace,
            console=Console(),
            logger_for_agent_logs=logger,
            enforce_completion_gate=False,
        )

    assert enforced.complete_tool.evidence_gate is evaluate_dialog
    assert plain.complete_tool.evidence_gate is None


def _import_runner_module():
    """Import the SWE-bench runner, stubbing heavy optional deps if absent.

    ``run_agent_on_swebench_problem`` imports ``datasets`` and ``docker`` at
    module level; neither is in requirements.txt, and the tests below never
    reach the code paths that use them, so stubs keep this suite runnable in
    the lean test environment while exercising the real wiring.
    """
    for optional_dep in ("datasets", "docker", "huggingface_hub"):
        try:
            __import__(optional_dep)
        except ImportError:
            sys.modules[optional_dep] = MagicMock()
    import run_agent_on_swebench_problem

    return run_agent_on_swebench_problem


def _make_eval_workspace(workspace: Path, problem_id: str, report: dict) -> None:
    """Lay down the eval artifacts run_eval_on_single_problem consumes."""
    (workspace / f"augment-agent.{problem_id}.json").write_text(
        json.dumps({"resolved_ids": [problem_id]})
    )
    report_dir = (
        workspace
        / "logs"
        / "run_evaluation"
        / problem_id
        / "augment-agent"
        / problem_id
    )
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(json.dumps(report))


def test_runner_attaches_gate_verdict_to_eval_outcomes(tmp_path):
    """run_eval_on_single_problem gates the 'resolved' claim on the instance
    report and records the structured verdict in eval_outcomes."""
    runner = _import_runner_module()
    problem_id = "instance-1"
    _make_eval_workspace(tmp_path, problem_id, NATIVE_REPORT_PASS)
    with patch.object(runner, "run_evaluation"):
        outcomes = runner.run_eval_on_single_problem(problem_id, tmp_path, Console())
    assert outcomes["is_success"] is True
    assert outcomes["evidence_gate"]["admissible"] is True
    assert "tests/test_new.py::test_it" in outcomes["evidence_gate"]["evidence"]


def test_runner_flags_visible_pass_hidden_fail(tmp_path):
    """resolved_ids says resolved, but the granular report shows a PASS_TO_PASS
    regression -- the gate refuses the claim (the paper's amplification case)."""
    runner = _import_runner_module()
    problem_id = "instance-1"
    _make_eval_workspace(tmp_path, problem_id, NATIVE_REPORT_REGRESSION)
    with patch.object(runner, "run_evaluation"):
        outcomes = runner.run_eval_on_single_problem(problem_id, tmp_path, Console())
    assert outcomes["is_success"] is True  # harness contract unchanged
    assert outcomes["evidence_gate"]["admissible"] is False
    assert "tests/test_old.py::test_it" in outcomes["evidence_gate"]["evidence"]


def test_runner_omits_verdict_when_no_report(tmp_path):
    """No instance report on disk -> no gate verdict, is_success still works."""
    runner = _import_runner_module()
    problem_id = "instance-1"
    (tmp_path / f"augment-agent.{problem_id}.json").write_text(
        json.dumps({"resolved_ids": [problem_id]})
    )
    with patch.object(runner, "run_evaluation"):
        outcomes = runner.run_eval_on_single_problem(problem_id, tmp_path, Console())
    assert outcomes["is_success"] is True
    assert "evidence_gate" not in outcomes
