"""Tests for trajectory error-lifecycle attribution and its debugger wiring.

These tests build a synthetic ``agent_logs.txt`` in the exact format
``tools/agent.py`` emits, then exercise ``utils.trajectory_error_attribution``
and the ``trajectory_debugger`` CLI end-to-end. They import the non-new
``utils.indent_utils`` -- the indentation model the repo's own str_replace
tool depends on -- to construct realistic tool-output error context, so the
test is grounded in the repo's existing abstractions rather than the new
module alone.
"""

import json

import trajectory_debugger
from utils import trajectory_error_attribution as tea
from utils.indent_utils import detect_indent_type

_DASHES = "-" * 45
TURN = f"{_DASHES} NEW TURN {_DASHES}"
USER = f"{_DASHES} USER INPUT {_DASHES}"


def _bash_turn(output: str, command: str = "cmd") -> str:
    return (
        f"Calling tool bash with input:\n - command: {command}\n"
        f"Tool output: \n{output}\n\n"
    )


def _log(*turn_bodies: str, instruction: str = "Fix the bug.") -> str:
    body = f"{USER}\n{instruction}\n"
    for turn_body in turn_bodies:
        body += f"{TURN}\n{turn_body}"
    return body


class TestParseAndAttribute:
    def test_parses_instruction_and_turns(self):
        instruction, turns = tea.parse_agent_log(_log(_bash_turn("hello")))
        assert instruction == "Fix the bug."
        assert len(turns) == 1
        assert turns[0].tool_name == "bash"
        assert turns[0].tool_output == "hello"

    def test_critical_is_earliest_unresolved_with_terminal_impact(self):
        log = _log(
            _bash_turn(
                "FAILED tests/test_widget.py::test_open - AssertionError", "pytest"
            ),
            _bash_turn("unrelated clean output", "echo hi"),
            _bash_turn(
                "FAILED tests/test_widget.py::test_open - AssertionError", "pytest"
            ),
        )
        report = tea.analyze_trajectory(log, is_success=False)
        assert report.critical is not None
        assert report.critical.turn_index == 0
        assert report.critical.kind == "test_failure"
        assert report.critical.terminal_impact is True
        assert "test_widget.py::test_open" in report.critical.evidence

    def test_resolved_error_is_not_critical(self):
        # The failing node later appears in a clean (passing) turn that
        # references the same artifact -> the error is traced as resolved.
        log = _log(
            _bash_turn("FAILED tests/test_widget.py::test_open", "pytest"),
            _bash_turn("tests/test_widget.py::test_open PASSED", "pytest"),
        )
        report = tea.analyze_trajectory(log, is_success=False)
        assert report.errors[0].resolved is True
        assert report.critical is None

    def test_success_verdict_suppresses_critical(self):
        log = _log(_bash_turn("FAILED tests/test_widget.py::test_open", "pytest"))
        report = tea.analyze_trajectory(log, is_success=True)
        assert report.critical is None

    def test_resolved_spurious_error_is_skipped_for_real_failure(self):
        # A transient tool error that gets fixed must not be attributed as the
        # critical failure; the later unresolved test failure should be.
        log = _log(
            _bash_turn("No such file or directory: 'nope.py'", "cat nope.py"),
            _bash_turn("created nope.py", "touch nope.py"),
            _bash_turn("FAILED tests/test_widget.py::test_open", "pytest"),
        )
        report = tea.analyze_trajectory(log, is_success=False)
        assert report.critical is not None
        assert report.critical.turn_index == 2


class TestIndentationErrorDetection:
    # Grounded in utils.indent_utils (the model the str_replace tool itself
    # uses): a mixed-indent edit produces the IndentationError signal the
    # attribution must surface as verbatim evidence.
    def test_indent_utils_classifies_mixed_indent(self):
        mixed = "def f():\n    x = 1\n\ty = 2\n"  # 4 spaces then a tab
        assert detect_indent_type(mixed).is_mixed

    def test_indentation_error_is_flagged_as_evidence(self):
        log = _log(
            _bash_turn(
                '  File "/testbed/widget.py", line 4\n'
                "    \treturn value\n"
                "IndentationError: unindent does not match any outer level\n",
                "python -c 'import widget'",
            )
        )
        report = tea.analyze_trajectory(log, is_success=False)
        assert report.critical is not None
        assert report.critical.kind == "exception"
        assert "IndentationError" in report.critical.evidence


class TestDebuggerCli:
    def test_cli_attributes_failure_and_writes_json(self, tmp_path):
        log_file = tmp_path / "agent_logs.txt"
        log_file.write_text(
            _log(_bash_turn("FAILED tests/test_widget.py::test_open", "pytest"))
        )
        out = tmp_path / "report.json"
        code = trajectory_debugger.main(
            [str(log_file), "--no-is-success", "--output", str(out)]
        )
        assert code == 0
        data = json.loads(out.read_text())
        assert data["critical"]["turn_index"] == 0
        assert "test_widget" in data["critical"]["evidence"]

    def test_cli_resolves_rollout_dir_and_traceback(self, tmp_path):
        roll = tmp_path / "rollout"
        roll.mkdir()
        (roll / "agent_logs.txt").write_text(
            _log(
                _bash_turn(
                    "Traceback (most recent call last):\nValueError: boom", "run"
                )
            )
        )
        code = trajectory_debugger.main([str(roll), "--no-is-success"])
        assert code == 0

    def test_cli_missing_logs_returns_error(self, tmp_path):
        code = trajectory_debugger.main([str(tmp_path / "nope.txt")])
        assert code == 1
