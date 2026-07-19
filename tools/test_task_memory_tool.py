"""Tests for the task memory tool, including integration with utils.common.

These tests exercise the tool through the real ``LLMTool.run()`` contract
defined in ``utils.common`` (a non-new module) and confirm the agent
registers the tool in its ``self.tools`` list (the call-site wiring edit).
"""

import json
import logging
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from utils.common import LLMTool, ToolImplOutput
from utils.workspace_manager import WorkspaceManager
from tools.task_memory_tool import TaskMemoryTool


class TestTaskMemoryTool(unittest.TestCase):
    """Behavioral tests for TaskMemoryTool via the LLMTool.run() path."""

    def setUp(self):
        self.tool = TaskMemoryTool()

    def test_is_llm_tool(self):
        """The tool subclasses the shared LLMTool contract from utils.common."""
        self.assertIsInstance(self.tool, LLMTool)

    def test_add_and_query(self):
        """Adding entries then querying them returns what was stored."""
        out = self.tool.run(
            {"action": "add", "entry_type": "goal", "content": "Fix the bug"}
        )
        data = json.loads(out)
        self.assertEqual(data["created"]["type"], "goal")
        self.assertEqual(data["created"]["id"], 1)

        out = self.tool.run(
            {
                "action": "add",
                "entry_type": "subtask",
                "content": "Reproduce the failure",
                "depends_on": [1],
            }
        )
        data = json.loads(out)
        self.assertEqual(data["created"]["depends_on"], [1])

        out = self.tool.run({"action": "query", "query_type": "goal"})
        data = json.loads(out)
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["entries"][0]["id"], 1)

    def test_supersede_reports_dependents(self):
        """Revision-aware rollback flags downstream entries at risk."""
        self.tool.run(
            {"action": "add", "entry_type": "finding", "content": "root cause is X"}
        )
        self.tool.run(
            {
                "action": "add",
                "entry_type": "subtask",
                "content": "patch X",
                "depends_on": [1],
            }
        )
        out = self.tool.run(
            {"action": "supersede", "entry_id": 1, "reason": "X was wrong"}
        )
        data = json.loads(out)
        self.assertEqual(data["superseded"]["status"], "superseded")
        self.assertEqual(data["dependents_at_risk"], [2])

    def test_memory_survives_for_reorientation(self):
        """Structured memory is queryable independent of the linear dialog."""
        self.tool.run(
            {"action": "add", "entry_type": "goal", "content": "ship the feature"}
        )
        self.tool.run(
            {
                "action": "add",
                "entry_type": "subtask",
                "content": "write tests",
                "depends_on": [1],
            }
        )
        out = self.tool.run({"action": "summary"})
        data = json.loads(out)
        self.assertEqual(data["total_entries"], 2)
        self.assertEqual(len(data["open_subtasks"]), 1)
        # Subtask #2 depends on goal #1, which is still open -> blocked.
        self.assertEqual(len(data["blocked_subtasks"]), 1)

    def test_update_status_unblocks_dependent(self):
        """Completing a dependency removes a subtask from the blocked list."""
        self.tool.run(
            {"action": "add", "entry_type": "subtask", "content": "do part A"}
        )
        self.tool.run(
            {
                "action": "add",
                "entry_type": "subtask",
                "content": "do part B",
                "depends_on": [1],
            }
        )
        summary = json.loads(self.tool.run({"action": "summary"}))
        self.assertEqual(len(summary["blocked_subtasks"]), 1)
        self.tool.run({"action": "update", "entry_id": 1, "status": "done"})
        summary = json.loads(self.tool.run({"action": "summary"}))
        self.assertEqual(len(summary["blocked_subtasks"]), 0)

    def test_run_impl_returns_tool_impl_output(self):
        result = self.tool.run_impl({"action": "summary"})
        self.assertIsInstance(result, ToolImplOutput)

    def test_invalid_action_rejected_by_schema(self):
        """An out-of-enum action is caught by LLMTool.run()'s jsonschema check."""
        out = self.tool.run({"action": "nope"})
        self.assertIn("Invalid tool input", out)
        self.assertIn("nope", out)


class TestTaskMemoryWiring(unittest.TestCase):
    """Confirm the agent (the existing call site) registers the tool."""

    @patch("tools.agent.create_bash_tool")
    def test_agent_registers_task_memory(self, _mock_bash):
        from tools.agent import Agent
        from utils.llm_client import LLMClient

        agent = Agent(
            client=LLMClient(),
            workspace_manager=WorkspaceManager(Path("/tmp")),
            console=Console(),
            logger_for_agent_logs=logging.getLogger("test_task_memory"),
        )
        names = [tool.name for tool in agent.tools]
        self.assertIn("task_memory", names)


if __name__ == "__main__":
    unittest.main()
