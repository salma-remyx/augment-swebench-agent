"""Tests for the state-bound revision contract and its wiring into the agent loop.

The contract unit tests exercise ``utils.revision_contract`` directly. The
integration test imports the (non-new) call-site module ``tools.agent`` and
drives the real ``Agent.run_impl`` loop with an injected fake client, asserting
that a stale-evidence note is threaded into the dialog -- i.e. the wiring edit
in ``run_impl`` actually fires. The bash tool's shell I/O is mocked so no real
subprocess is spawned; the str_replace primitive runs for real.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from tools.agent import Agent
from utils.common import ToolCallParameters
from utils.llm_client import LLMClient, TextResult, ToolCall
from utils.revision_contract import RevisionContract
from utils.workspace_manager import WorkspaceManager


def _bash_test_call(cid: str = "t1", command: str = "python -m pytest -q"):
    return ToolCallParameters(
        tool_call_id=cid, tool_name="bash", tool_input={"command": command}
    )


def _str_replace_call(cid: str = "t2"):
    return ToolCallParameters(
        tool_call_id=cid,
        tool_name="str_replace_editor",
        tool_input={
            "command": "str_replace",
            "path": "module.py",
            "old_str": "x",
            "new_str": "y",
        },
    )


class TestRevisionContract:
    def test_code_state_hash_is_deterministic_and_sensitive(self, tmp_path):
        contract = RevisionContract(workspace_root=tmp_path)
        (tmp_path / "a.py").write_text("x = 1\n")
        before = contract.code_state_hash()
        # Unchanged content + path -> same digest.
        (tmp_path / "a.py").write_text("x = 1\n")
        assert contract.code_state_hash() == before
        # Content change -> digest changes.
        (tmp_path / "a.py").write_text("x = 2\n")
        assert contract.code_state_hash() != before

    def test_code_state_hash_ignores_untracked_artifacts(self, tmp_path):
        contract = RevisionContract(workspace_root=tmp_path)
        (tmp_path / "tracked.py").write_text("a = 1\n")
        base = contract.code_state_hash()
        # Generated / vendored / non-source files must not move the state.
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "tracked.cpython-311.pyc").write_text("junk")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text("ref")
        (tmp_path / "notes.txt").write_text("ignored")  # .txt not tracked
        assert contract.code_state_hash() == base

    def test_test_command_binds_state_and_returns_no_note(self, tmp_path):
        contract = RevisionContract(workspace_root=tmp_path)
        note = contract.observe_tool(_bash_test_call(), "3 passed")
        assert note is None
        receipt = contract.receipts[-1]
        assert receipt.kind == "test_trace"
        assert receipt.admitted
        assert contract.verified_checkpoint == receipt.state_hash

    def test_revision_on_changed_state_is_flagged_stale(self, tmp_path):
        contract = RevisionContract(workspace_root=tmp_path)
        (tmp_path / "module.py").write_text("x = 1\n")
        contract.observe_tool(_bash_test_call(), "1 passed")
        # The agent now edits the code -- the captured trace is stale.
        (tmp_path / "module.py").write_text("x = 2\n")
        note = contract.observe_tool(_str_replace_call(), "ok")
        assert note is not None
        assert "STALE" in note
        receipt = contract.receipts[-1]
        assert receipt.kind == "revision"
        assert not receipt.admitted
        assert "stale test trace" in receipt.reason

    def test_revision_before_any_test_is_fresh(self, tmp_path):
        contract = RevisionContract(workspace_root=tmp_path)
        (tmp_path / "module.py").write_text("x = 1\n")
        note = contract.observe_tool(_str_replace_call(), "ok")
        assert note is None
        assert contract.receipts[-1].admitted

    def test_rerunning_tests_rebinds_evidence(self, tmp_path):
        contract = RevisionContract(workspace_root=tmp_path)
        (tmp_path / "module.py").write_text("x = 1\n")
        contract.observe_tool(_bash_test_call(), "1 passed")
        (tmp_path / "module.py").write_text("x = 2\n")
        assert contract.observe_tool(_str_replace_call(), "ok") is not None
        # Agent re-runs tests against the new state -> evidence fresh again.
        note = contract.observe_tool(_bash_test_call("t3", "pytest -q"), "1 passed")
        assert note is None
        assert contract.receipts[-1].admitted

    def test_reset_clears_evidence(self, tmp_path):
        contract = RevisionContract(workspace_root=tmp_path)
        contract.observe_tool(_bash_test_call(), "ok")
        assert contract.receipts
        contract.reset()
        assert contract.receipts == []
        assert contract.verified_checkpoint is None

    def test_audit_summary_counts_admissions(self, tmp_path):
        contract = RevisionContract(workspace_root=tmp_path)
        (tmp_path / "module.py").write_text("x = 1\n")
        contract.observe_tool(_bash_test_call(), "1 passed")
        (tmp_path / "module.py").write_text("x = 2\n")
        contract.observe_tool(_str_replace_call(), "ok")  # stale
        summary = contract.audit_summary()
        assert summary["receipts"] == 2
        assert summary["admitted"] == 1
        assert summary["stale"] == 1


class _ScriptedClient(LLMClient):
    """Returns a scripted sequence of model responses, no network."""

    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, messages, max_tokens, **kwargs):  # type: ignore[no-untyped-def]
        return self._responses.pop(0), {}


@pytest.fixture
def _no_shell_bash():
    """Keep the agent's bash tool from spawning a real pexpect shell."""
    with (
        patch("tools.bash_tool.start_persistent_shell") as mock_start,
        patch("tools.bash_tool.run_command") as mock_run,
    ):
        mock_start.return_value = (MagicMock(), "PROMPT>> ")
        mock_run.side_effect = lambda child, prompt, cmd: (
            "hello" if cmd == "echo hello" else "1 passed in 0.10s"
        )
        yield mock_run


class TestAgentWiring:
    def test_run_impl_threads_stale_note_into_dialog(self, tmp_path, _no_shell_bash):
        # Scripted loop: run tests -> edit code -> complete. The edit invalidates
        # the test trace, so run_impl must append a STALE note to the tool result
        # that reaches the model.
        fix_path = tmp_path / "fix.py"
        responses = [
            [
                ToolCall(
                    tool_call_id="t1",
                    tool_name="bash",
                    tool_input={"command": "echo pytest: 1 passed"},
                )
            ],
            [
                ToolCall(
                    tool_call_id="t2",
                    tool_name="str_replace_editor",
                    tool_input={
                        "command": "create",
                        "path": str(fix_path),
                        "file_text": "y = 2",
                    },
                )
            ],
            [TextResult(text="done")],
        ]
        agent = Agent(
            client=_ScriptedClient(responses),
            workspace_manager=WorkspaceManager(root=tmp_path),
            console=Console(quiet=True),
            logger_for_agent_logs=logging.getLogger("test_revision_contract"),
            ask_user_permission=False,
        )
        # __init__ wiring: the contract is bound to the agent's workspace.
        assert isinstance(agent.revision_contract, RevisionContract)
        assert agent.revision_contract.workspace_root == tmp_path

        agent.run_agent("repair the module")

        # The revision receipt was recorded as stale (admitted=False).
        revision_receipts = [
            r for r in agent.revision_contract.receipts if r.kind == "revision"
        ]
        assert any(
            not r.admitted and "STALE" in r.note_for_agent for r in revision_receipts
        )
        # And the note actually reached the model via the dialog (the wiring).
        assert "STALE" in str(agent.dialog)
        assert agent.revision_contract.audit_summary()["stale"] == 1
