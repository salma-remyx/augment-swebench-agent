"""State-bound evidence ledger and admission receipts for the generate-test-revise loop.

Adapted from "Looping Is Not Reliability: State-Bound Evidence and Typed Revision
Contracts for Agentic Code Repair" (arXiv:2607.24604). The paper shows that
repetition in a generate-test-revise loop gives no reliability guarantee: a
correct patch is frequently lost when a later revision rests on STALE test
traces that were captured against an older code state. Its remedy is to (1) bind
verifier/test evidence to exact code states, (2) preserve verified checkpoints,
and (3) emit auditable admission receipts that gate each revision on fresh,
mechanically-verifiable evidence.

This module ports the paper's *mechanically enforceable subset* into the agent's
tool-result loop (``tools/agent.py:Agent.run_impl``):

  * :meth:`RevisionContract.code_state_hash` binds evidence to a code state.
  * :meth:`RevisionContract.observe_tool` records each loop step, detects when a
    revision (``str_replace_editor``) invalidates the last captured test trace,
    and returns an advisory note that the loop appends to the tool result so the
    model sees that its evidence is stale and should re-run the tests.
  * :attr:`RevisionContract.receipts` / :meth:`RevisionContract.audit_summary`
    expose the auditable admission trail.

This is a conformance artifact -- it makes stale evidence *detectable* and
surfaces it; it does not by itself improve repair competence, which the paper is
explicit it does not claim.

Adaptations (Mode 2), spelled out so the reader can tell paper from port:

  * "exact code state" -> a bounded, deterministic content digest over tracked
    source files (capped file count / per-file bytes). Same change-detection
    signal; cheaper on large repos than a full snapshot.
  * verifier evidence -> the agent's own test-command output, identified by
    command pattern (``pytest`` / ``python -m unittest`` / ``tox`` ...), since
    the loop only threads the tool-output string, not the bash exit code.
  * the paper's separate sealed-study / benchmark harness is intentionally out
    of scope -- evaluation belongs in a downstream PR.
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from utils.common import ToolCallParameters

# Commands that produce "test traces" -- the verifier evidence the loop binds to
# a code state. Matched against the bash command string. ``\b`` keeps this from
# firing on incidental substrings like ``test_pytest_data.py``.
_TEST_COMMAND_RE = re.compile(
    r"\b(pytest|tox|nox|nosetests|python\s+-m\s+(pytest|unittest))\b"
)

# Source suffixes whose contents define the "code state" under repair.
_TRACKED_SUFFIXES = frozenset(
    (
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".kt",
        ".scala",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".sh",
        ".sql",
        ".md",
        ".rst",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".cfg",
        ".ini",
    )
)

# Directories never hashed (generated / vendored / VCS / virtualenvs).
_SKIP_DIRS = frozenset(
    (
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".venv",
        "venv",
        "env",
        ".eggs",
        "dist",
        "build",
        "target",
        ".next",
        ".cache",
    )
)


def _is_test_command(command: str) -> bool:
    return _TEST_COMMAND_RE.search(command) is not None


@dataclass
class AdmissionReceipt:
    """Auditable record of what the loop admitted at one step.

    Attributes:
        step: 1-indexed loop step this receipt was issued at.
        tool_name: The tool whose result this receipt covers.
        state_hash: The code state bound to this step's evidence.
        kind: ``"test_trace"`` | ``"revision"`` | ``"other"``.
        admitted: True if the step's evidence is fresh; False if it rests on a
            stale trace.
        reason: Human-readable explanation of the admission decision.
        verified_checkpoint: The most recent state at which tests were run, at
            the time this receipt was issued.
        note_for_agent: Advisory text threaded back to the model (empty when the
            step needs no warning).
    """

    step: int
    tool_name: str
    state_hash: str
    kind: str
    admitted: bool
    reason: str
    verified_checkpoint: Optional[str]
    note_for_agent: str = ""


class RevisionContract:
    """Binds test evidence to code states and gates revisions on fresh evidence.

    Adapted from arXiv:2607.24604. One instance lives on the ``Agent`` and
    observes every tool result via :meth:`observe_tool`, called from
    ``Agent.run_impl`` right after each tool runs.
    """

    def __init__(
        self,
        workspace_root: Optional[Path],
        max_files_to_hash: int = 5000,
        max_bytes_per_file: int = 262144,
    ):
        self.workspace_root = workspace_root
        self.max_files_to_hash = max_files_to_hash
        self.max_bytes_per_file = max_bytes_per_file

        self.receipts: list[AdmissionReceipt] = []
        self._step = 0
        self._last_test_state_hash: Optional[str] = None
        self._last_test_step: Optional[int] = None
        # The most recent code state at which tests were run (preserved
        # checkpoint); None until the first test trace is observed.
        self.verified_checkpoint: Optional[str] = None

    def reset(self) -> None:
        """Clear accumulated evidence for a fresh repair attempt."""
        self.receipts.clear()
        self._step = 0
        self._last_test_state_hash = None
        self._last_test_step = None
        self.verified_checkpoint = None

    # -- state binding -------------------------------------------------------

    def code_state_hash(self) -> str:
        """Deterministic digest of tracked source files under the workspace.

        Changes iff a tracked source file's path or contents change. Bounded by
        ``max_files_to_hash`` / ``max_bytes_per_file`` so it stays cheap on large
        repos; returns ``"unknown"`` if the workspace is unreadable.
        """
        if self.workspace_root is None:
            return "unknown"
        digest = hashlib.sha256()
        counted = 0
        try:
            paths = sorted(p for p in self.workspace_root.rglob("*") if p.is_file())
        except OSError:
            return "unknown"
        for path in paths:
            if counted >= self.max_files_to_hash:
                digest.update(b"<truncated>")
                break
            if not self._should_track(path):
                continue
            rel = path.relative_to(self.workspace_root)
            digest.update(rel.as_posix().encode("utf-8", "ignore"))
            digest.update(b"\0")
            try:
                size = path.stat().st_size
            except OSError:
                digest.update(b"<unstatable>")
                digest.update(b"\0")
                continue
            if size <= self.max_bytes_per_file:
                try:
                    digest.update(path.read_bytes())
                except OSError:
                    digest.update(b"<unreadable>")
            else:
                digest.update(b"<large>")
            digest.update(b"\0")
            counted += 1
        return digest.hexdigest()

    @staticmethod
    def _should_track(path: Path) -> bool:
        for part in path.parts:
            if part in _SKIP_DIRS or part.endswith(".egg-info"):
                return False
        return path.suffix.lower() in _TRACKED_SUFFIXES

    # -- loop observation ----------------------------------------------------

    def observe_tool(
        self,
        tool_call: ToolCallParameters,
        tool_output: str,
    ) -> Optional[str]:
        """Record one loop step; return an advisory note to show the model, if any.

        For a test command, the current code state is bound as the evidence
        anchor and preserved as a verified checkpoint. For a revision
        (``str_replace_editor``) made after a test trace was captured, if the
        code has since changed the captured trace is STALE and a note is
        returned so the loop can tell the model to re-run the tests against the
        current state.
        """
        self._step += 1
        current_state = self.code_state_hash()
        tool_name = tool_call.tool_name
        command = str(tool_call.tool_input.get("command", ""))

        note: Optional[str] = None
        admitted = True
        reason = "fresh evidence"
        kind = "other"

        if tool_name == "bash" and _is_test_command(command):
            kind = "test_trace"
            self._last_test_state_hash = current_state
            self._last_test_step = self._step
            self.verified_checkpoint = current_state
            reason = "test trace bound to current code state"
        elif (
            tool_name == "str_replace_editor"
            and self._last_test_state_hash is not None
            and current_state != self._last_test_state_hash
        ):
            kind = "revision"
            admitted = False
            reason = (
                f"stale test trace: tests at step {self._last_test_step} were "
                f"bound to state {self._last_test_state_hash[:8]} but the code "
                f"has since changed (now {current_state[:8]})"
            )
            note = (
                "[state-bound evidence] The last test run is now STALE: the code "
                "has changed since those tests ran. Re-run the tests against the "
                "current state before treating their result as evidence."
            )
        elif tool_name == "str_replace_editor":
            kind = "revision"
            reason = "revision on fresh (no prior test trace) state"

        self.receipts.append(
            AdmissionReceipt(
                step=self._step,
                tool_name=tool_name,
                state_hash=current_state,
                kind=kind,
                admitted=admitted,
                reason=reason,
                verified_checkpoint=self.verified_checkpoint,
                note_for_agent=note or "",
            )
        )
        return note

    # -- audit ---------------------------------------------------------------

    def audit_summary(self) -> dict[str, Any]:
        """Aggregate receipt counts for logging / an audit trail."""
        return {
            "steps": self._step,
            "receipts": len(self.receipts),
            "admitted": sum(1 for r in self.receipts if r.admitted),
            "stale": sum(1 for r in self.receipts if not r.admitted),
            "verified_checkpoint": self.verified_checkpoint,
        }
