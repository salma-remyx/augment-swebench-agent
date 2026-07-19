"""Task Memory Tool — persistent, revision-aware task memory for the agent.

Keeps a structured record of goals, subtasks, findings, decisions, and
observations across turns, with dependency links and revision-aware
supersession (rollback). This directly addresses the agent's flat, linear
dialog context: when ``DialogMessages`` is truncated to the last few turns,
the structured memory survives on the tool instance and the model can
re-orient by querying it instead of re-reading the whole history.

Adapted from "Task Memory Engine: Spatial Memory for Robust Multi-Step LLM
Agents" (arXiv:2505.19436). This is a Mode 2 (adapted) port: the paper's
core mechanism — persistent goal/dependency tracking with revision-aware
rollback — is kept at fidelity, while the learned spatial retriever is
substituted with a parameter-free, tool-call-driven query, and the paper's
separate benchmark suite is cut (evaluation belongs in a downstream PR).
TME is training-free, and so is this tool.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.common import (
    DialogMessages,
    LLMTool,
    ToolImplOutput,
)

VALID_ENTRY_TYPES = ("goal", "subtask", "finding", "decision", "observation")
VALID_STATUSES = ("open", "done", "failed", "superseded")


@dataclass
class MemoryEntry:
    """A single typed task-memory entry."""

    id: int
    entry_type: str
    content: str
    status: str = "open"
    depends_on: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    superseded_by: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.entry_type,
            "content": self.content,
            "status": self.status,
            "depends_on": list(self.depends_on),
            "notes": list(self.notes),
            "superseded_by": self.superseded_by,
        }


class TaskMemoryTool(LLMTool):
    """A persistent, revision-aware task memory for multi-step agents.

    The memory lives on the tool instance (which persists for the agent's
    lifetime, like ``SequentialThinkingTool``'s thought history), so it is
    not lost when the linear dialog is truncated.
    """

    name = "task_memory"
    description = """\
A persistent, structured task memory that survives dialog truncation.

Use it to track evolving goals, subtask dependencies, findings, decisions,
and observations across many turns — instead of relying on the linear chat
history, which gets truncated. The memory is revision-aware: when you learn
a previous step was wrong, supersede it (a rollback) and the tool reports
which downstream entries depended on it.

When to use:
- The task has multiple subtasks or evolving goals.
- You want to record a decision or finding to revisit later.
- You are about to backtrack or change approach — supersede the old entry.
- You lost context after truncation — call action=summary to re-orient.

Actions:
- add: create an entry (entry_type, content, optional depends_on).
- update: change an entry's status (open/done/failed) or append a note.
- supersede: mark an entry superseded (rollback); reports dependents at risk.
- query: retrieve entries filtered by type/status/depends_on/keyword.
- summary: compact overview (goals, open/blocked subtasks, decisions).
"""

    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "update", "supersede", "query", "summary"],
                "description": "What to do with the task memory.",
            },
            "entry_type": {"type": "string", "enum": list(VALID_ENTRY_TYPES)},
            "content": {"type": "string"},
            "depends_on": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
            },
            "entry_id": {"type": "integer", "minimum": 1},
            "status": {"type": "string", "enum": list(VALID_STATUSES)},
            "note": {"type": "string"},
            "reason": {"type": "string"},
            "query_type": {"type": "string", "enum": list(VALID_ENTRY_TYPES)},
            "query_status": {"type": "string", "enum": list(VALID_STATUSES)},
            "keyword": {"type": "string"},
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        super().__init__()
        self.entries: dict[int, MemoryEntry] = {}
        self._next_id: int = 1

    # --- helpers -------------------------------------------------------

    def _add_entry(
        self,
        entry_type: str,
        content: str,
        depends_on: Optional[list[int]] = None,
    ) -> MemoryEntry:
        entry_id = self._next_id
        self._next_id += 1
        entry = MemoryEntry(
            id=entry_id,
            entry_type=entry_type,
            content=content,
            depends_on=list(depends_on or []),
        )
        self.entries[entry_id] = entry
        return entry

    def _dependents_of(self, entry_id: int) -> list[int]:
        return sorted(e.id for e in self.entries.values() if entry_id in e.depends_on)

    def _blocked_subtasks(self) -> list[MemoryEntry]:
        blocked: list[MemoryEntry] = []
        for entry in self.entries.values():
            if entry.entry_type != "subtask" or entry.status != "open":
                continue
            open_deps = [
                dep
                for dep in entry.depends_on
                if dep in self.entries and self.entries[dep].status != "done"
            ]
            if open_deps:
                blocked.append(entry)
        return blocked

    # --- actions -------------------------------------------------------

    def _handle_add(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        entry_type = tool_input.get("entry_type")
        content = tool_input.get("content")
        if entry_type not in VALID_ENTRY_TYPES:
            return {"error": f"Invalid or missing entry_type: {entry_type!r}"}
        if not content:
            return {"error": "content is required for action=add"}
        depends_on = tool_input.get("depends_on") or []
        entry = self._add_entry(entry_type, content, depends_on)
        result: dict[str, Any] = {
            "created": entry.to_dict(),
            "message": f"Added {entry_type} #{entry.id}.",
        }
        unknown = [dep for dep in depends_on if dep not in self.entries]
        if unknown:
            result["warning"] = f"depends_on references unknown ids: {unknown}"
        return result

    def _handle_update(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        entry = self.entries.get(tool_input.get("entry_id"))
        if entry is None:
            return {"error": f"No entry with id {tool_input.get('entry_id')}"}
        status = tool_input.get("status")
        if status is not None:
            if status not in VALID_STATUSES:
                return {"error": f"Invalid status: {status!r}"}
            entry.status = status
        note = tool_input.get("note")
        if note:
            entry.notes.append(note)
        return {"updated": entry.to_dict()}

    def _handle_supersede(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        entry = self.entries.get(tool_input.get("entry_id"))
        if entry is None:
            return {"error": f"No entry with id {tool_input.get('entry_id')}"}
        reason = tool_input.get("reason")
        entry.status = "superseded"
        if reason:
            entry.notes.append(f"Superseded: {reason}")
        dependents = self._dependents_of(entry.id)
        if dependents:
            message = (
                f"Entry #{entry.id} superseded. {len(dependents)} downstream "
                "entr(y/ies) depended on it; review and update or supersede "
                "them as needed."
            )
        else:
            message = f"Entry #{entry.id} superseded. No downstream dependents."
        return {
            "superseded": entry.to_dict(),
            "dependents_at_risk": dependents,
            "message": message,
        }

    def _handle_query(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        query_type = tool_input.get("query_type")
        query_status = tool_input.get("query_status")
        keyword = tool_input.get("keyword")
        dep_id = tool_input.get("depends_on")

        def matches(entry: MemoryEntry) -> bool:
            if query_type and entry.entry_type != query_type:
                return False
            if query_status and entry.status != query_status:
                return False
            if keyword and keyword.lower() not in entry.content.lower():
                return False
            if dep_id and dep_id not in entry.depends_on:
                return False
            return True

        found = [entry.to_dict() for entry in self.entries.values() if matches(entry)]
        return {"count": len(found), "entries": found}

    def _handle_summary(self) -> dict[str, Any]:
        goals = [e for e in self.entries.values() if e.entry_type == "goal"]
        open_subtasks = [
            e
            for e in self.entries.values()
            if e.entry_type == "subtask" and e.status == "open"
        ]
        decisions = [e for e in self.entries.values() if e.entry_type == "decision"]
        return {
            "total_entries": len(self.entries),
            "goals": [
                {"id": g.id, "content": g.content, "status": g.status} for g in goals
            ],
            "open_subtasks": [e.to_dict() for e in open_subtasks],
            "blocked_subtasks": [e.to_dict() for e in self._blocked_subtasks()],
            "decisions": [{"id": d.id, "content": d.content} for d in decisions],
        }

    # --- tool entry point ---------------------------------------------

    def run_impl(
        self,
        tool_input: dict[str, Any],
        dialog_messages: Optional[DialogMessages] = None,
    ) -> ToolImplOutput:
        action = tool_input.get("action")
        if action == "add":
            result = self._handle_add(tool_input)
            message = "Added a task memory entry."
        elif action == "update":
            result = self._handle_update(tool_input)
            message = "Updated a task memory entry."
        elif action == "supersede":
            result = self._handle_supersede(tool_input)
            message = "Superseded a task memory entry."
        elif action == "query":
            result = self._handle_query(tool_input)
            message = "Queried task memory."
        elif action == "summary":
            result = self._handle_summary()
            message = "Summarized task memory."
        else:
            result = {"error": f"Unknown action: {action!r}"}
            message = "Unknown task memory action."
        return ToolImplOutput(
            tool_output=json.dumps(result, indent=2),
            tool_result_message=message,
            auxiliary_data={"action": action, "result": result},
        )

    def get_tool_start_message(self, tool_input: dict[str, Any]) -> str:
        return f"Updating task memory ({tool_input.get('action', '?')})"
