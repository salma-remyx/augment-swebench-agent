"""Tool for indicating task completion."""

from typing import Any, Optional
from utils.common import (
    DialogMessages,
    LLMTool,
    ToolImplOutput,
)
from utils.evidence_gate import EvidenceGate, format_stop_message


class CompleteTool(LLMTool):
    name = "complete"
    """The model should call this tool when it is done with the task."""

    description = "Call this tool when you are done with the task, and supply your answer or summary."
    input_schema = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the question, or final summary of actions taken to accomplish the task.",
            },
        },
        "required": ["answer"],
    }

    def __init__(self, evidence_gate: Optional[EvidenceGate] = None):
        super().__init__()
        self.answer: str = ""
        self.evidence_gate = evidence_gate

    @property
    def should_stop(self):
        return self.answer != ""

    def reset(self):
        self.answer = ""

    def run_impl(
        self,
        tool_input: dict[str, Any],
        dialog_messages: Optional[DialogMessages] = None,
    ) -> ToolImplOutput:
        assert tool_input["answer"], "Model returned empty answer"
        # Proof-or-Stop: the DONE claim is honored only when the evidence gate
        # admits it. When the gate refuses, leave ``self.answer`` unset so
        # ``should_stop`` stays False -- the agent loop continues and the stop
        # message is returned to the model as the tool result.
        if self.evidence_gate is not None:
            verdict = self.evidence_gate(dialog_messages)
            if not verdict.admissible:
                return ToolImplOutput(
                    format_stop_message(verdict),
                    "Completion blocked pending verifiable evidence",
                )
        self.answer = tool_input["answer"]
        return ToolImplOutput("Task completed", "Task completed")

    def get_tool_start_message(self, tool_input: dict[str, Any]) -> str:
        return ""
