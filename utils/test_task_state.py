"""Tests for the StructAgent-style unified task-progress state.

The unit tests exercise ``utils.task_state`` directly. The wiring test imports
the (non-new) call-site module ``tools.agent`` and drives its real ``run_impl``
loop with an injected fake client + a patched (shell-free) bash tool, asserting
the state is observed, failures are attributed, and the compact state snapshot
is surfaced back into the dialog the model sees.
"""

import json
import logging

import pytest
from rich.console import Console

import tools.agent as agent_mod
from tools.agent import Agent
from utils.common import LLMTool, ToolImplOutput
from utils.llm_client import LLMClient, TextResult, ToolCall
from utils.task_state import TaskState
from utils.workspace_manager import WorkspaceManager


# --------------------------------------------------------------------------- #
# Fake LLM client for the verifier-backed transition (reuses the pairwise
# reward path in utils.solution_verifier without real API calls).
# --------------------------------------------------------------------------- #
class _FixedScoreClient(LLMClient):
    """Always returns the same A/B score pair in the verifier's tag format."""

    def __init__(self, score_a: float, score_b: float):
        self.score_a = score_a
        self.score_b = score_b
        self.calls = 0

    def generate(self, messages, max_tokens, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        body = f"<score_A>{self.score_a}</score_A>\n<score_B>{self.score_b}</score_B>"
        return [TextResult(text=body)], {}


class _ScriptedAgentClient(LLMClient):
    """Returns a scripted sequence of assistant responses and records inputs."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen_messages: list = []

    def generate(self, messages, max_tokens, **kwargs):  # type: ignore[no-untyped-def]
        self.seen_messages.append(messages)
        return self.responses.pop(0), {}


class _FakeBashTool(LLMTool):
    """Shell-free stand-in for the bash tool, returning a fixed output."""

    name = "bash"
    description = "fake bash"
    input_schema = {"type": "object"}

    def __init__(self, output: str):
        super().__init__()
        self._output = output

    def run_impl(self, tool_input, dialog_messages=None):  # type: ignore[no-untyped-def]
        return ToolImplOutput(self._output, "fake bash")


# --------------------------------------------------------------------------- #
# TaskState unit tests
# --------------------------------------------------------------------------- #
class TestTaskStateObserve:
    def test_records_evidence_and_action_counts(self):
        state = TaskState()
        state.begin("fix the bug")
        state.observe("bash", {"command": "ls"}, "file_a.py\nfile_b.py")
        state.observe("str_replace_editor", {"path": "x"}, "The file has been edited.")

        assert state._turn == 2
        assert state.action_counts == {"bash": 1, "str_replace_editor": 1}
        assert [e.action for e in state.evidence] == ["bash", "str_replace_editor"]
        assert all(e.ok for e in state.evidence)
        assert state.status == "in_progress"

    def test_attributes_failures_and_flips_to_blocked(self):
        state = TaskState()
        state.begin("fix the bug")
        state.observe("bash", {"command": "make test"}, "=== FAILURES ===\nboom")

        assert state.status == "blocked"
        failed = [e for e in state.evidence if not e.ok]
        assert len(failed) == 1
        assert failed[0].action == "bash"
        assert state.attribution[-1].source == "test_failure"
        assert state.attribution[-1].turn == 1

    def test_passing_test_run_is_not_misfiled_as_failure(self):
        state = TaskState()
        state.begin("fix the bug")
        # A green pytest summary must read as success.
        state.observe("bash", {"command": "pytest"}, "3 passed in 0.4s")
        assert state.evidence[-1].ok is True
        assert state.status == "in_progress"
        assert state.attribution == []

    def test_complete_action_marks_completed(self):
        state = TaskState()
        state.begin("fix the bug")
        state.observe("bash", {"command": "make test"}, "1 passed")
        state.observe("complete", {"answer": "done"}, "Task completed")
        assert state.status == "completed"

    def test_evidence_ledger_is_bounded(self):
        state = TaskState()
        state.begin("fix the bug")
        for i in range(50):
            state.observe("bash", {"command": str(i)}, f"output {i}")
        # Compact over long runs: only the most recent evidence is retained.
        assert len(state.evidence) <= 12
        # Action counts still summarize the full history.
        assert state.action_counts["bash"] == 50


class TestTaskStateCompactStr:
    def test_snapshot_includes_progress_fields(self):
        state = TaskState()
        state.begin("a" * 200)  # long goal is truncated in the snapshot
        state.observe("bash", {"command": "make test"}, "Command timed out.")

        snapshot = state.to_compact_str()
        assert "TaskState" in snapshot
        assert "status=blocked" in snapshot
        assert "bash=1" in snapshot
        assert "FAILED" in snapshot
        assert "open failures: turn 1 bash:command_timeout" in snapshot


class TestTaskStateCheckpoint:
    def test_checkpoint_due_respects_cadence(self):
        state = TaskState(checkpoint_every=3)
        state.begin("fix the bug")
        state.observe("bash", {"command": "a"}, "ok")
        assert state.checkpoint_due() is False
        state.observe("bash", {"command": "b"}, "ok")
        assert state.checkpoint_due() is False
        state.observe("bash", {"command": "c"}, "ok")
        assert state.checkpoint_due() is True


class TestTaskStateVerifyProgress:
    def test_accepts_transition_on_grounded_progress(self):
        client = _FixedScoreClient(score_a=80.0, score_b=20.0)
        state = TaskState(verify_threshold=0.55)
        state.begin("fix the bug")
        state.observe("bash", {"command": "make test"}, "1 passed")

        verdict = state.verify_progress(client)

        assert verdict.score == pytest.approx(0.8)
        assert verdict.accepted is True
        assert state.last_verify is verdict
        # Accepted snapshot becomes the new baseline for the next transition.
        assert state._last_accepted_snapshot == state.snapshot_for_verifier()
        assert client.calls > 0

    def test_rejects_and_attributes_when_progress_not_grounded(self):
        client = _FixedScoreClient(score_a=20.0, score_b=80.0)
        state = TaskState(verify_threshold=0.55)
        state.begin("fix the bug")
        state.observe("bash", {"command": "make test"}, "1 passed")

        verdict = state.verify_progress(client)

        assert verdict.score == pytest.approx(0.2)
        assert verdict.accepted is False
        # Rejection is recorded as a failure attribution for recovery.
        assert state.attribution[-1].source == "no_grounded_progress"
        # Baseline is not advanced on a rejected transition.
        assert state._last_accepted_snapshot != state.snapshot_for_verifier()


# --------------------------------------------------------------------------- #
# Wiring test: drives the real Agent.run_impl loop with an injected client and
# a shell-free bash tool, asserting the state integration is live.
# --------------------------------------------------------------------------- #
class TestAgentWiring:
    def test_run_impl_observes_state_and_surfaces_snapshot(self, tmp_path, monkeypatch):
        # Avoid spawning a real persistent shell: swap the bash tool factory.
        monkeypatch.setattr(
            agent_mod,
            "create_bash_tool",
            lambda **kwargs: _FakeBashTool("Command timed out. Please try again."),
        )

        client = _ScriptedAgentClient(
            responses=[
                # Turn 1: model calls the (fake) bash tool, which "fails".
                [
                    ToolCall(
                        tool_call_id="call_1",
                        tool_name="bash",
                        tool_input={"command": "make test"},
                    )
                ],
                # Turn 2: model produces a plain answer -> run completes.
                [TextResult(text="I will retry the failing test.")],
            ]
        )
        agent = Agent(
            client=client,
            workspace_manager=WorkspaceManager(root=tmp_path),
            console=Console(),
            logger_for_agent_logs=logging.getLogger("test_task_state"),
            max_turns=2,
            enable_task_state=True,
            surface_task_state=True,
            enable_state_verification=False,
        )

        agent.run_impl({"instruction": "fix the flaky test"})

        # The loop observed the bash call and attributed its failure.
        assert agent.task_state is not None
        assert agent.task_state.goal == "fix the flaky test"
        assert any(e.action == "bash" for e in agent.task_state.evidence)
        assert agent.task_state.status == "blocked"
        assert agent.task_state.attribution[-1].source == "command_timeout"

        # The compact state snapshot was surfaced into the dialog the model
        # sees on the following turn (turn 2's input includes turn 1's tool
        # result, now carrying the [TaskState] block).
        turn2_messages = client.seen_messages[1]
        serialized = json.dumps(
            [[block.to_dict() for block in turn] for turn in turn2_messages]
        )
        assert "TaskState" in serialized
        assert "status=blocked" in serialized

    def test_task_state_can_be_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            agent_mod,
            "create_bash_tool",
            lambda **kwargs: _FakeBashTool("all good"),
        )
        client = _ScriptedAgentClient(
            responses=[
                [
                    ToolCall(
                        tool_call_id="call_1",
                        tool_name="bash",
                        tool_input={"command": "echo hi"},
                    )
                ],
                [TextResult(text="done")],
            ]
        )
        agent = Agent(
            client=client,
            workspace_manager=WorkspaceManager(root=tmp_path),
            console=Console(),
            logger_for_agent_logs=logging.getLogger("test_task_state"),
            max_turns=2,
            enable_task_state=False,
        )
        agent.run_impl({"instruction": "noop"})

        assert agent.task_state is None
        # With state disabled, no snapshot is appended to the tool result.
        turn2_messages = client.seen_messages[1]
        serialized = json.dumps(
            [[block.to_dict() for block in turn] for turn in turn2_messages]
        )
        assert "TaskState" not in serialized
