"""Unified, verifier-backed task-progress state for the coding agent.

Adapted (Mode 2) port of *StructAgent: Harness Long-horizon Digital Agents
with Unified Causal Structure* (arXiv:2607.11388). StructAgent's central
claim is that a long-horizon agent should reason over a *compact, verifiable
State* of task progress rather than the raw interaction history, and that
progress updates should be gated by verification (verifier-backed state
transitions), which in turn enables targeted failure recovery and
evidence-driven completion.

This module keeps that core mechanism intact -- a unified ``TaskState`` that
distills the tool-call stream into compact structured progress, plus a
verifier-backed transition gate -- while substituting the paper's
computer-use-specific auxiliary machinery for target-native equivalents:

  * The paper's GUI actor / DOM probe over OSWorld & Mind2Web is replaced by
    evidence drawn from THIS agent's own tools: bash output, file edits, and
    the test/eval signals they produce. ``observe`` distills each tool result
    into a compact evidence entry and attributes failures by source.
  * The paper's bespoke state-transition verifier is replaced by the team's
    existing ``utils.solution_verifier.directed_reward`` pairwise reward
    (LLM-as-a-Verifier, arXiv:2607.05391) -- extending that confirmed
    ``utils/`` landing zone from best-of-N selection into runtime progress
    tracking.
  * The paper's OSWorld / Minecraft benchmark suite is intentionally cut;
    evaluation belongs downstream.

Preserved from the paper:
  * A unified State holding compact, verifiable task progress (goal, status,
    bounded evidence ledger, failure attribution, action counts).
  * Verifier-backed transitions: progress is grounded by a pairwise reward
    comparison of the current state against the last *accepted* state, so a
    transition is only accepted when the verifier confirms grounded progress.
  * Targeted failure recovery input: failed evidence is attributed by source
    so the agent can reason about *what* failed rather than re-reading raw
    output.

The state is surfaced to the agent as a compact, bounded snapshot so its
cognitive loop operates over structured progress instead of a growing transcript.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from utils.llm_client import LLMClient
from utils.solution_verifier import directed_reward

# Substrings in a tool-result string that unambiguously indicate the action
# failed, paired with the attribution source the failure is filed under.
_FAILURE_MARKERS: tuple[tuple[str, str], ...] = (
    ("Traceback (most recent call last)", "python_traceback"),
    ("Command timed out", "command_timeout"),
    ("Error executing command", "command_error"),
    ("Invalid tool input", "invalid_input"),
    ("=== FAILURES ===", "test_failure"),
    ("=== ERRORS ===", "test_error"),
    ("FAILED ", "test_failure"),
)

_EVIDENCE_SUMMARY_CHARS = 200
_GOAL_CHARS = 80
_MAX_EVIDENCE = 12
_MAX_ATTRIBUTION = 12

# The baseline the current state is compared against before any transition
# has been accepted -- "no progress yet". Comparing the first real state
# against this is what lets the verifier confirm grounded initial progress.
_NO_PROGRESS_BASELINE = "(no verifiable progress yet -- task just started)"

#: Overall task lifecycle status.
STATUS_IN_PROGRESS = "in_progress"
STATUS_BLOCKED = "blocked"
STATUS_COMPLETED = "completed"


@dataclass
class Evidence:
    """One compact evidence entry distilled from a tool result."""

    turn: int
    action: str
    ok: bool
    summary: str


@dataclass
class Attribution:
    """A failure attributed to a source (input to targeted recovery)."""

    turn: int
    action: str
    source: str


@dataclass
class VerifyVerdict:
    """Outcome of a verifier-backed state transition."""

    score: float
    accepted: bool
    note: str


@dataclass
class TaskState:
    """Unified, compact, verifiable task-progress state.

    Call ``begin`` at the start of a run, ``observe`` after every tool result,
    and ``verify_progress`` at checkpoints (cadence via ``checkpoint_every``)
    to gate transitions with the pairwise verifier reward.
    """

    goal: str = ""
    status: str = STATUS_IN_PROGRESS
    checkpoint_every: int = 5
    verify_threshold: float = 0.55
    n_evaluations: int = 1
    evidence: list[Evidence] = field(default_factory=list)
    attribution: list[Attribution] = field(default_factory=list)
    action_counts: dict[str, int] = field(default_factory=dict)
    last_verify: Optional[VerifyVerdict] = None
    _turn: int = 0
    _last_accepted_snapshot: str = _NO_PROGRESS_BASELINE

    def begin(self, goal: str) -> None:
        """Reset the state for a new run anchored to ``goal``."""
        self.goal = goal
        self.status = STATUS_IN_PROGRESS
        self.evidence.clear()
        self.attribution.clear()
        self.action_counts.clear()
        self.last_verify = None
        self._turn = 0
        self._last_accepted_snapshot = _NO_PROGRESS_BASELINE

    def observe(self, action: str, tool_input: dict[str, Any], result: Any) -> None:
        """Distill one tool result into compact evidence (+ failure attribution).

        ``result`` is the string the tool returned to the dialog. Failure is
        detected by unambiguous markers and attributed by source so the agent
        has structured recovery input instead of raw output to re-parse.
        """
        self._turn += 1
        text = result if isinstance(result, str) else str(result)
        ok, source = _classify(text)

        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self.evidence.append(Evidence(self._turn, action, ok, _summarize(text)))
        if len(self.evidence) > _MAX_EVIDENCE:
            # Keep the most recent evidence; older turns stay summarized in
            # action_counts so the snapshot stays compact over long runs.
            self.evidence = self.evidence[-_MAX_EVIDENCE:]

        if not ok:
            self.attribution.append(Attribution(self._turn, action, source))
            if len(self.attribution) > _MAX_ATTRIBUTION:
                self.attribution = self.attribution[-_MAX_ATTRIBUTION:]
            if self.status == STATUS_IN_PROGRESS:
                self.status = STATUS_BLOCKED

        # The complete tool signals evidence-driven task completion.
        if action == "complete":
            self.status = STATUS_COMPLETED

    def checkpoint_due(self) -> bool:
        """Whether the current turn lands on a verification checkpoint."""
        step = max(1, self.checkpoint_every)
        return self._turn > 0 and self._turn % step == 0

    def snapshot_for_verifier(self) -> str:
        """Textual state the pairwise verifier scores against the goal."""
        counts = ", ".join(f"{k}={v}" for k, v in self.action_counts.items()) or "none"
        failures = (
            "; ".join(f"turn {a.turn} {a.action}:{a.source}" for a in self.attribution)
            or "none"
        )
        recent = self.evidence[-1].summary if self.evidence else "(no actions yet)"
        return (
            f"goal: {self.goal}\n"
            f"status: {self.status} (turn {self._turn})\n"
            f"actions taken: {counts}\n"
            f"recent evidence: {recent}\n"
            f"attributed failures: {failures}"
        )

    def to_compact_str(self) -> str:
        """Bounded snapshot surfaced to the agent each turn.

        This is the paper's compact verifiable progress: the agent reasons
        over this structured state rather than re-reading the whole transcript.
        """
        goal_preview = (
            (self.goal[:_GOAL_CHARS] + "...")
            if len(self.goal) > _GOAL_CHARS
            else self.goal
        )
        counts = ", ".join(f"{k}={v}" for k, v in self.action_counts.items()) or "none"
        if self.evidence:
            last = self.evidence[-1]
            flag = "ok" if last.ok else "FAILED"
            last_line = f"{last.action} (turn {last.turn}, {flag}): {last.summary}"
        else:
            last_line = "(no actions yet)"
        failures = (
            "; ".join(f"turn {a.turn} {a.action}:{a.source}" for a in self.attribution)
            or "none"
        )
        if self.last_verify is not None:
            verify_line = (
                f"score={self.last_verify.score:.2f} "
                f"accepted={self.last_verify.accepted}"
            )
        else:
            verify_line = "not run"
        return (
            f"[TaskState] turn={self._turn} status={self.status} "
            f'goal="{goal_preview}"\n'
            f"actions: {counts}\n"
            f"last evidence: {last_line}\n"
            f"open failures: {failures}\n"
            f"verifier: {verify_line}"
        )

    def verify_progress(self, client: LLMClient) -> VerifyVerdict:
        """Gate the current state transition with the pairwise verifier reward.

        The current compact state is scored against the last *accepted* state
        for the goal using ``utils.solution_verifier.directed_reward`` (the
        team's LLM-as-a-Verifier reward). The transition is accepted only when
        the verifier confirms grounded progress (``score >= verify_threshold``);
        an unaccepted transition is recorded as a failure attribution so the
        agent can recover rather than silently stalling. With
        ``n_evaluations=1`` scoring is deterministic (temperature 0).
        """
        current = self.snapshot_for_verifier()
        score, _prev_score = directed_reward(
            self.goal,
            current,
            self._last_accepted_snapshot,
            client,
            n_evaluations=self.n_evaluations,
        )
        accepted = score >= self.verify_threshold
        note = f"verifier score={score:.2f} threshold={self.verify_threshold:.2f}"
        if accepted:
            self._last_accepted_snapshot = current
        else:
            self.attribution.append(
                Attribution(self._turn, "verifier", "no_grounded_progress")
            )
            if len(self.attribution) > _MAX_ATTRIBUTION:
                self.attribution = self.attribution[-_MAX_ATTRIBUTION:]
        self.last_verify = VerifyVerdict(score, accepted, note)
        return self.last_verify


def _classify(text: str) -> tuple[bool, str]:
    """Return ``(ok, source)`` for a tool-result string.

    ``ok`` is False only when an unambiguous failure marker is present, so a
    passing test run or a successful edit is never mis-filed as a failure.
    """
    for marker, source in _FAILURE_MARKERS:
        if marker in text:
            return False, source
    return True, "ok"


def _summarize(text: str) -> str:
    """Collapse whitespace and truncate a tool result to a compact summary."""
    collapsed = " ".join(text.split())
    if len(collapsed) > _EVIDENCE_SUMMARY_CHARS:
        return collapsed[:_EVIDENCE_SUMMARY_CHARS] + "..."
    return collapsed
