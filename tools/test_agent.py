"""Integration tests for the E3 (Estimate, Execute, Expand) wiring in Agent.

These construct a real ``tools.agent.Agent`` with a scripted (no-network) LLM
client and assert that ``run_impl`` consumes the estimator's minimum-viable turn
budget first and *expands* to the full cap only when that budget is exhausted
without completing. This exercises the wiring edit in ``tools/agent.py`` end to
end.
"""

import logging

from rich.console import Console

from tools.agent import Agent
from utils.common import LLMTool, ToolImplOutput
from utils.llm_client import LLMClient, TextResult, ToolCall
from utils.task_complexity_estimator import (
    ComplexityTier,
    TaskComplexityEstimator,
    TaskOperatingPoint,
)
from utils.workspace_manager import WorkspaceManager

# A valid sequential_thinking tool input -- lets the agent spend a turn without
# completing (should_stop stays False) and without side effects beyond memory.
_THINK_INPUT = {
    "thought": "planning",
    "nextThoughtNeeded": True,
    "thoughtNumber": 1,
    "totalThoughts": 2,
}


class _ScriptedClient(LLMClient):
    """Returns canned responses in order, then repeats the last one."""

    def __init__(self, responses):
        self._responses = responses
        self.call_count = 0

    def generate(
        self,
        messages,
        max_tokens,
        system_prompt=None,
        temperature=0.0,
        tools=None,
        tool_choice=None,
        thinking_tokens=None,
    ):
        idx = min(self.call_count, len(self._responses) - 1)
        response = self._responses[idx]
        self.call_count += 1
        return response, {}


class _NoopBash(LLMTool):
    """A bash tool stand-in that never spawns a shell (test hermeticity)."""

    name = "bash"
    description = "noop bash for tests"
    input_schema = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def run_impl(self, tool_input, dialog_messages=None):
        return ToolImplOutput(tool_output="noop", tool_result_message="noop")


def _think_call(call_id):
    return [
        ToolCall(
            tool_call_id=call_id,
            tool_name="sequential_thinking",
            tool_input=dict(_THINK_INPUT),
        )
    ]


def _make_agent(monkeypatch, tmp_path, client, estimator=None, max_turns=5):
    """Build an Agent without spawning a real bash shell."""
    import tools.agent as agent_module

    monkeypatch.setattr(agent_module, "create_bash_tool", lambda **kw: _NoopBash())
    return Agent(
        client=client,
        workspace_manager=WorkspaceManager(root=tmp_path),
        console=Console(),
        logger_for_agent_logs=logging.getLogger("test_agent"),
        max_turns=max_turns,
        ask_user_permission=False,
        task_complexity_estimator=estimator,
    )


class TestAgentE3Budgeting:
    """Verify the Estimate -> Execute -> Expand flow inside run_impl."""

    def test_agent_has_default_estimator(self, monkeypatch, tmp_path):
        client = _ScriptedClient([[TextResult(text="ok")]])
        agent = _make_agent(monkeypatch, tmp_path, client)
        assert isinstance(agent.task_complexity_estimator, TaskComplexityEstimator)

    def test_simple_task_completes_without_expand(self, monkeypatch, tmp_path):
        # A genuinely simple task: the estimator grants a small first-phase
        # budget, and the agent completes on turn 1 -- well inside it. No
        # expand should occur.
        client = _ScriptedClient([[TextResult(text="done")]])
        agent = _make_agent(monkeypatch, tmp_path, client, max_turns=5)
        result = agent.run_agent("Fix the typo in the README title.")
        assert result == "done"
        assert client.call_count == 1

    def test_expand_runs_when_minimum_viable_budget_exhausted(
        self, monkeypatch, tmp_path
    ):
        # first_phase = 2 (injected estimator) < max_turns = 5, so turns 3-4
        # can ONLY be reached by expanding. Completing on turn 4 therefore
        # proves the Expand step fired.
        estimator = _FixedEstimator(initial_turns=2)
        client = _ScriptedClient(
            [
                _think_call("c1"),  # turn 1 (phase 1)
                _think_call("c2"),  # turn 2 (phase 1) -- exhausts budget
                _think_call("c3"),  # turn 3 (expand)
                [TextResult(text="expanded-and-done")],  # turn 4 (expand) -- completes
            ]
        )
        agent = _make_agent(
            monkeypatch, tmp_path, client, estimator=estimator, max_turns=5
        )
        result = agent.run_agent("Refactor the revenue helper.")

        # Without expand the loop would stop at turn 2 and report failure; the
        # fact that we reached turn 4 and returned the completion text is the
        # expand behavior under test.
        assert result == "expanded-and-done"
        assert client.call_count == 4

    def test_full_cap_used_when_never_completing(self, monkeypatch, tmp_path):
        # The agent never completes: first_phase (2) + expand (5-2=3) = 5 =
        # max_turns total turns. Expand must top the budget up exactly once.
        estimator = _FixedEstimator(initial_turns=2)
        client = _ScriptedClient([_think_call("c")])  # never completes
        agent = _make_agent(
            monkeypatch, tmp_path, client, estimator=estimator, max_turns=5
        )
        result = agent.run_agent("Refactor the revenue helper.")
        assert "did not complete after max turns" in result.lower()
        assert client.call_count == 5  # 2 (phase 1) + 3 (expand)

    def test_no_expand_when_estimate_meets_cap(self, monkeypatch, tmp_path):
        # first_phase clamps to max_turns: nothing to expand into. The agent
        # uses exactly the cap when it never completes.
        estimator = _FixedEstimator(initial_turns=20)  # > max_turns
        client = _ScriptedClient([_think_call("c")])
        agent = _make_agent(
            monkeypatch, tmp_path, client, estimator=estimator, max_turns=4
        )
        result = agent.run_agent("Refactor the revenue helper.")
        assert "did not complete after max turns" in result.lower()
        assert client.call_count == 4

    def test_real_estimator_feeds_the_loop(self, monkeypatch, tmp_path):
        # End-to-end with the REAL estimator: a simple instruction gets a
        # small first-phase budget (initial_turns=3) under a max_turns cap of
        # 5, so completion on turn 4 must come from expansion.
        client = _ScriptedClient(
            [
                _think_call("c1"),
                _think_call("c2"),
                _think_call("c3"),
                [TextResult(text="done")],
            ]
        )
        agent = _make_agent(monkeypatch, tmp_path, client, max_turns=5)
        op = agent.task_complexity_estimator.estimate("Fix the typo in the README.")
        assert op.tier is ComplexityTier.SIMPLE  # initial_turns == 3 < 5

        result = agent.run_agent("Fix the typo in the README.")
        assert result == "done"
        assert client.call_count == 4  # turn 4 is in the expand phase


class _FixedEstimator(TaskComplexityEstimator):
    """Estimator stub returning a fixed operating point regardless of input."""

    def __init__(self, initial_turns):
        super().__init__()
        self._initial_turns = initial_turns

    def estimate(self, instruction, workspace_root=None):
        return TaskOperatingPoint(
            tier=ComplexityTier.MODERATE,
            initial_turns=self._initial_turns,
            score=0.5,
            signals=("fixed-test-estimator",),
        )
