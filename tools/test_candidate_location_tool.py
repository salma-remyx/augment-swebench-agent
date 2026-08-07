"""Tests for the candidate location explorer tool.

The first test class exercises the wiring in the existing call site
(``tools.agent.Agent``) -- it constructs the agent and confirms the new tool is
registered and shares the workspace, then drives an end-to-end exploration
through the agent's actual tool registry. The second class covers the tool's
own validation and behavior.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.agent import Agent
from tools.candidate_location_tool import CandidateLocationTool
from utils.workspace_manager import WorkspaceManager


class TestAgentWiresCandidateLocationTool(unittest.TestCase):
    """The agent's tool registry (the call site) must include the explorer."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir(parents=True, exist_ok=True)
        (self.root / "src" / "real.py").write_text("x = 1\n")
        self.workspace_manager = WorkspaceManager(root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_agent(self) -> Agent:
        # Avoid spawning a real pexpect bash shell during construction.
        with patch("tools.agent.create_bash_tool") as mock_bash:
            mock_bash.return_value = Mock()
            return Agent(
                client=Mock(),
                workspace_manager=self.workspace_manager,
                console=Mock(),
                logger_for_agent_logs=Mock(),
            )

    def test_tool_is_registered_with_unique_name(self):
        agent = self._make_agent()
        names = [tool.name for tool in agent.tools]
        self.assertIn("candidate_location_explorer", names)
        # No accidental duplicates (the agent enforces uniqueness at run time).
        self.assertEqual(len(names), len(set(names)))

    def test_registered_tool_end_to_end(self):
        agent = self._make_agent()
        tool = next(t for t in agent.tools if t.name == "candidate_location_explorer")
        self.assertIsInstance(tool, CandidateLocationTool)
        # The workspace manager wired in from the agent reaches the tool.
        self.assertIs(tool.workspace_manager, self.workspace_manager)

        # Multi-location sampling, grounded against the real workspace.
        grounded = json.loads(
            tool.run_impl(
                {
                    "action": "add_location",
                    "file_path": "src/real.py",
                    "rationale": "imports the buggy symbol",
                    "confidence": 4,
                }
            ).tool_output
        )
        self.assertTrue(grounded["grounded"])

        ungrounded = json.loads(
            tool.run_impl(
                {
                    "action": "add_location",
                    "file_path": "src/missing.py",
                    "rationale": "maybe the caller",
                    "confidence": 2,
                }
            ).tool_output
        )
        self.assertFalse(ungrounded["grounded"])
        self.assertEqual(ungrounded["counts"]["num_locations"], 2)

        # Iterative refinement: record a failed repair attempt.
        attempt = json.loads(
            tool.run_impl(
                {
                    "action": "record_attempt",
                    "file_path": "src/real.py",
                    "attempt": "added a null check",
                    "outcome": "failed",
                    "notes": "tests still red",
                }
            ).tool_output
        )
        self.assertEqual(attempt["counts"]["num_failed"], 1)

        # Distill an insight from the failed attempt.
        reflection = json.loads(
            tool.run_impl(
                {"action": "reflect", "insight": "the fault is in the caller"}
            ).tool_output
        )
        self.assertEqual(len(reflection["insights"]), 1)
        self.assertEqual(len(reflection["unresolved_attempts"]), 1)

        # Summary replays locations, attempts, and insights for final-round use.
        summary = json.loads(tool.run_impl({"action": "summary"}).tool_output)
        self.assertEqual(summary["counts"]["num_locations"], 2)
        self.assertEqual(summary["counts"]["num_attempts"], 1)
        self.assertEqual(len(summary["insights"]), 1)
        self.assertIn("src/real.py", summary["locations"])


class TestCandidateLocationToolUnits(unittest.TestCase):
    """Unit tests for the explorer's validation and exploration discipline."""

    def setUp(self):
        self.tool = CandidateLocationTool()

    def test_unknown_action_fails_gracefully(self):
        data = json.loads(self.tool.run_impl({"action": "bogus"}).tool_output)
        self.assertEqual(data["status"], "failed")

    def test_missing_required_field_fails_gracefully(self):
        data = json.loads(
            self.tool.run_impl(
                {"action": "add_location", "file_path": "a.py"}
            ).tool_output
        )
        self.assertEqual(data["status"], "failed")
        self.assertIn("rationale", data["error"])

    def test_invalid_outcome_fails_gracefully(self):
        data = json.loads(
            self.tool.run_impl(
                {
                    "action": "record_attempt",
                    "file_path": "a.py",
                    "attempt": "tweak",
                    "outcome": "nope",
                }
            ).tool_output
        )
        self.assertEqual(data["status"], "failed")

    def test_summary_nudges_location_sampling_first(self):
        data = json.loads(self.tool.run_impl({"action": "summary"}).tool_output)
        self.assertEqual(data["counts"]["num_locations"], 0)
        self.assertIn("candidate", data["next_step_hint"].lower())

    def test_reflect_requires_insight(self):
        data = json.loads(self.tool.run_impl({"action": "reflect"}).tool_output)
        self.assertEqual(data["status"], "failed")
        self.assertIn("insight", data["error"])

    def test_get_tool_start_message(self):
        msg = self.tool.get_tool_start_message(
            {"action": "record_attempt", "file_path": "a.py"}
        )
        self.assertIn("record_attempt", msg)
        self.assertIn("a.py", msg)


if __name__ == "__main__":
    unittest.main()
