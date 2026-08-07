"""Candidate Location Explorer tool.

A stateful reasoning tool that structures the agent's repair-strategy
exploration along the two axes called out by PhoenixRepair
(https://arxiv.org/abs/2607.18859v1):

  1. *Multi-location sampling* -- systematically enumerate and explore
     several candidate edit locations before locking onto a fix, instead
     of committing to the first plausible spot.
  2. *Iterative reflection and refinement* -- record each repair attempt
     per location together with its test outcome, then distill insights
     from the full attempt history to guide the final patch.

This is an adapted port (Mode 2): the paper's multi-agent orchestrator is
substituted by the agent's native tool-dispatch loop (the driver calls this
tool the way it calls ``sequential_thinking``), and the paper's graph-based
localization augmentation is replaced by a parameter-free check that grounds
each proposed location against the real workspace. The driver performs the
reasoning; this tool structures, grounds, and remembers it.
"""

import json
from typing import Any, Dict, List, Optional

from utils.common import DialogMessages, LLMTool, ToolImplOutput
from utils.workspace_manager import WorkspaceManager

VALID_OUTCOMES = ("passed", "failed", "partial")
VALID_ACTIONS = ("add_location", "record_attempt", "reflect", "summary")


class CandidateLocationTool(LLMTool):
    """Track candidate edit locations, per-location repair attempts, and distilled insights.

    The tool holds structured state across calls so the agent is nudged to
    explore multiple locations and to reflect on failed attempts before
    producing its final patch.
    """

    name = "candidate_location_explorer"
    description = """\
A structured tool for exploring repair strategies before committing to a fix.

Use this tool to avoid locking onto the first plausible edit location. It keeps
running state across calls so you can systematically widen the search for a fix.

When to use this tool:
- Before making an edit, when the fault location is not yet certain.
- After a repair attempt fails its tests, to record the outcome and reflect.
- Before the final patch, to review all candidate locations, attempts, and
  distilled lessons in one place.

Supported actions (pass via the `action` field):
- add_location: Register a candidate edit location with a rationale and a
  confidence (1-5). The tool grounds the path against the workspace and reports
  whether the file actually exists.
- record_attempt: Record a repair attempt at a location along with its test
  outcome (passed / failed / partial) and optional notes.
- reflect: Distill a concrete lesson from the attempt history. Reflections are
  accumulated and replayed in the summary to guide the final patch.
- summary: Return the full structured state -- every location, its attempts and
  outcomes, and the distilled insights -- to guide final-round generation.

Call `summary` before writing your final patch so the distilled insights from
all prior attempts inform it.
"""

    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(VALID_ACTIONS),
                "description": "Which exploration action to perform.",
            },
            "file_path": {
                "type": "string",
                "description": "Candidate edit location (add_location, record_attempt).",
            },
            "rationale": {
                "type": "string",
                "description": "Why this location may contain the fault (add_location).",
            },
            "confidence": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Confidence that this location holds the fault (add_location).",
            },
            "attempt": {
                "type": "string",
                "description": "Description of the repair attempt (record_attempt).",
            },
            "outcome": {
                "type": "string",
                "enum": list(VALID_OUTCOMES),
                "description": "Test outcome of the attempt (record_attempt).",
            },
            "notes": {
                "type": "string",
                "description": "Optional details about the attempt (record_attempt).",
            },
            "insight": {
                "type": "string",
                "description": "A concrete lesson distilled from the attempt history (reflect).",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        workspace_manager: Optional[WorkspaceManager] = None,
        verbose: bool = False,
    ):
        """Initialize the candidate location explorer.

        Args:
            workspace_manager: Optional workspace manager used to ground proposed
                locations against the real filesystem. When omitted, grounding
                is skipped.
            verbose: If True, log exploration steps.
        """
        super().__init__()
        self.workspace_manager = workspace_manager
        self.verbose = verbose
        # file_path -> {"rationale", "confidence", "grounded", "attempts": [...]}
        self.locations: Dict[str, Dict[str, Any]] = {}
        self.insights: List[str] = []

    # ------------------------------------------------------------------ helpers

    def _ground_path(self, file_path: str) -> bool:
        """Return True if ``file_path`` resolves to an existing workspace file."""
        if self.workspace_manager is None:
            return True  # grounding unavailable; do not penalize the caller
        try:
            resolved = self.workspace_manager.workspace_path(file_path)
            return resolved.exists()
        except Exception:
            return False

    def _ensure_location(self, file_path: str) -> Dict[str, Any]:
        """Get or create the location record for ``file_path``."""
        if file_path not in self.locations:
            self.locations[file_path] = {
                "rationale": "",
                "confidence": 0,
                "grounded": self._ground_path(file_path),
                "attempts": [],
            }
        return self.locations[file_path]

    def _counts(self) -> Dict[str, int]:
        attempts = [a for loc in self.locations.values() for a in loc["attempts"]]
        return {
            "num_locations": len(self.locations),
            "num_attempts": len(attempts),
            "num_passed": sum(1 for a in attempts if a["outcome"] == "passed"),
            "num_failed": sum(1 for a in attempts if a["outcome"] == "failed"),
            "num_partial": sum(1 for a in attempts if a["outcome"] == "partial"),
            "num_insights": len(self.insights),
        }

    def _next_step_hint(self) -> str:
        """Suggest the most useful next exploration step (the paper's discipline)."""
        counts = self._counts()
        if counts["num_locations"] == 0:
            return (
                "Sample 2-3 distinct candidate edit locations before attempting a "
                "fix; add each with the add_location action."
            )
        unresolved = counts["num_failed"] + counts["num_partial"]
        if counts["num_passed"] == 0 and counts["num_locations"] < 2:
            return (
                "No passing attempt yet and only one location sampled; broaden the "
                "search by adding at least one more candidate location."
            )
        if unresolved > counts["num_insights"]:
            return (
                "There are failed/partial attempts without a corresponding reflection; "
                "use the reflect action to distill a lesson before retrying."
            )
        if counts["num_passed"] > 0:
            return (
                "A passing attempt exists; call summary, then finalize the patch guided "
                "by the distilled insights."
            )
        return (
            "Call summary to review every location, attempt, and insight before "
            "producing the final patch."
        )

    def _summary_view(self) -> Dict[str, Any]:
        view = {
            "locations": self.locations,
            "insights": self.insights,
            "counts": self._counts(),
            "next_step_hint": self._next_step_hint(),
        }
        return view

    # ------------------------------------------------------------------ actions

    def _add_location(self, tool_input: Dict[str, Any]) -> ToolImplOutput:
        file_path = tool_input.get("file_path")
        rationale = tool_input.get("rationale")
        if not file_path or not isinstance(file_path, str):
            raise ValueError("add_location requires a string 'file_path'")
        if not rationale or not isinstance(rationale, str):
            raise ValueError("add_location requires a string 'rationale'")
        confidence = tool_input.get("confidence", 3)
        if not isinstance(confidence, int) or not (1 <= confidence <= 5):
            raise ValueError("'confidence' must be an integer between 1 and 5")

        location = self._ensure_location(file_path)
        # Re-ground on update in case the file was created since first sampled.
        location["grounded"] = self._ground_path(file_path)
        location["rationale"] = rationale
        location["confidence"] = confidence

        response = {
            "action": "add_location",
            "file_path": file_path,
            "grounded": location["grounded"],
            "counts": self._counts(),
            "next_step_hint": self._next_step_hint(),
        }
        return ToolImplOutput(
            tool_output=json.dumps(response, indent=2),
            tool_result_message=(
                f"Recorded candidate location {file_path} "
                f"(grounded={location['grounded']})"
            ),
            auxiliary_data={"exploration": self._summary_view()},
        )

    def _record_attempt(self, tool_input: Dict[str, Any]) -> ToolImplOutput:
        file_path = tool_input.get("file_path")
        attempt = tool_input.get("attempt")
        outcome = tool_input.get("outcome")
        if not file_path or not isinstance(file_path, str):
            raise ValueError("record_attempt requires a string 'file_path'")
        if not attempt or not isinstance(attempt, str):
            raise ValueError("record_attempt requires a string 'attempt'")
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"'outcome' must be one of {VALID_OUTCOMES}")
        notes = tool_input.get("notes", "") or ""

        location = self._ensure_location(file_path)
        attempt_record = {
            "attempt": attempt,
            "outcome": outcome,
            "notes": notes,
        }
        location["attempts"].append(attempt_record)

        response = {
            "action": "record_attempt",
            "file_path": file_path,
            "attempt": attempt_record,
            "location_attempts": len(location["attempts"]),
            "counts": self._counts(),
            "next_step_hint": self._next_step_hint(),
        }
        return ToolImplOutput(
            tool_output=json.dumps(response, indent=2),
            tool_result_message=(
                f"Recorded {outcome} attempt #{len(location['attempts'])} "
                f"at {file_path}"
            ),
            auxiliary_data={"exploration": self._summary_view()},
        )

    def _reflect(self, tool_input: Dict[str, Any]) -> ToolImplOutput:
        insight = tool_input.get("insight")
        if not insight or not isinstance(insight, str):
            raise ValueError("reflect requires a string 'insight'")
        self.insights.append(insight)

        # Replay the unresolved attempts so reflection is anchored to evidence.
        unresolved = [
            {"file_path": fp, **a}
            for fp, loc in self.locations.items()
            for a in loc["attempts"]
            if a["outcome"] in ("failed", "partial")
        ]
        response = {
            "action": "reflect",
            "insight": insight,
            "unresolved_attempts": unresolved,
            "insights": self.insights,
            "counts": self._counts(),
            "next_step_hint": self._next_step_hint(),
        }
        return ToolImplOutput(
            tool_output=json.dumps(response, indent=2),
            tool_result_message=(f"Distilled insight #{len(self.insights)}: {insight}"),
            auxiliary_data={"exploration": self._summary_view()},
        )

    def _summary(self) -> ToolImplOutput:
        view = self._summary_view()
        return ToolImplOutput(
            tool_output=json.dumps(view, indent=2),
            tool_result_message=(
                "Exploration summary: "
                f"{view['counts']['num_locations']} location(s), "
                f"{view['counts']['num_attempts']} attempt(s), "
                f"{view['counts']['num_insights']} insight(s)."
            ),
            auxiliary_data={"exploration": view},
        )

    # ------------------------------------------------------------------ public

    def run_impl(
        self,
        tool_input: Dict[str, Any],
        dialog_messages: Optional[DialogMessages] = None,
    ) -> ToolImplOutput:
        """Dispatch on the requested exploration action."""
        try:
            action = tool_input.get("action")
            if action == "add_location":
                return self._add_location(tool_input)
            if action == "record_attempt":
                return self._record_attempt(tool_input)
            if action == "reflect":
                return self._reflect(tool_input)
            if action == "summary":
                return self._summary()
            raise ValueError(f"'action' must be one of {VALID_ACTIONS}")
        except Exception as e:
            error_response = {"error": str(e), "status": "failed"}
            return ToolImplOutput(
                tool_output=json.dumps(error_response, indent=2),
                tool_result_message=f"Error in candidate_location_explorer: {e}",
                auxiliary_data={"error": str(e)},
            )

    def get_tool_start_message(self, tool_input: Dict[str, Any]) -> str:
        action = tool_input.get("action", "?")
        file_path = tool_input.get("file_path")
        suffix = f" on {file_path}" if file_path else ""
        return f"Exploring repair strategy ({action}{suffix})"
