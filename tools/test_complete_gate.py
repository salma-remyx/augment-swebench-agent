"""Tests for the Proof-or-Stop evidence gate wired into the complete path.

These tests import the (non-new) call-site module ``tools.complete_tool`` plus
``utils.common`` / ``utils.llm_client`` to build a real ``DialogMessages``
populated with scripted tool calls, then exercise the wiring edit made to
``CompleteTool.run_impl`` -- the DONE claim is admitted only when the gate sees
fresh, mechanically-verifiable test-pass evidence. No real tools run and no API
calls are made.
"""

import logging
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
