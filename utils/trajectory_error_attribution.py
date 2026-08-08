"""Trajectory error-lifecycle attribution for failed agent rollouts.

Adapted from "TRAJDEBUG: Tracing Error Lifecycle to Identify Critical
Failures in Long-Horizon Agent Trajectories" (arXiv:2608.06346).

Mode 2 (adapted port). The paper's core mechanism is preserved --
error-lifecycle tracing: (1) discover error steps in a long agent
trajectory, (2) trace each error's resolution status across the rest of the
trajectory, and (3) attribute the CRITICAL failure to the earliest
unresolved error whose impact propagates to the terminal failure, surfaced
with verbatim evidence. Three auxiliary components are substituted with
target-native equivalents:

  * Multi-granularity history compression -> the agent's own turn-segmented
    log. ``cli.py`` routes agent output to ``agent_logs.txt`` with explicit
    ``NEW TURN`` delimiters emitted by ``tools/agent.py:run_impl``; the log
    already segments at turn + tool-result granularity, so no learned
    compressor is needed.
  * LLM-based evidence error identification -> a parameter-free evidence
    detector that scans each turn's tool output for the observable error
    signals a SWE-bench agent actually emits (Python tracebacks, pytest
    failures, raised exceptions, str_replace / bash tool errors, timeouts).
    The paper notes step evidence is "scattered across distant instructions,
    observations, and prior context"; this detector captures the observable
    observation-side of that evidence without an LLM call.
  * TrajErrBench (486 hand-annotated trajectories) -> cut. Evaluating this
    diagnostic against ground-truth critical-error annotations is a
    downstream concern.

Preserved from the paper: error discovery, per-error resolution-status
tracing, and critical attribution to the earliest decisive unresolved error
with verbatim evidence -- the actionable feedback the paper shows improves
downstream agent success.

The repo already produces the exact artifact this consumes per failed
rollout -- ``agent_logs.txt`` plus an ``is_success`` verdict
(``run_agent_on_swebench_problem.py``). ``trajectory_debugger.py`` is the
analysis CLI that wires this in, paralleling ``majority_vote_ensembler.py``.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any

# Log-format contract emitted by tools/agent.py:run_impl. Kept here as the
# parser's source of truth so it tracks the producer without importing the
# (docker/pexpect-bearing) agent module at analysis time.
_DASHES = "-" * 45
_TURN_DELIMITER = f"{_DASHES} NEW TURN {_DASHES}"
_USER_INPUT_MARKER = f"{_DASHES} USER INPUT {_DASHES}"
_PLANNING_PREFIX = "Top-level agent planning next step:"
_TOOL_CALL_RE = re.compile(r"Calling tool (?P<name>\S+) with input:")
_TOOL_OUTPUT_MARKER = "Tool output:"
_NO_TOOLS_MARKER = "[no tools were called]"

# (kind, regex). Scanned per output line; first matching kind wins so a line
# is labelled once. Order is deliberate: most-specific first.
_ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("traceback", re.compile(r"Traceback \(most recent call last\)")),
    (
        "test_failure",
        re.compile(r"(FAILED\b|FAILURES\b|ERRORS\b|\d+\s+failed\b)", re.IGNORECASE),
    ),
    ("exception", re.compile(r"\b([A-Z]\w*(?:Error|Exception)):")),
    (
        "tool_error",
        re.compile(
            r"(No match found|More than one match|Command not executed due to banned"
            r"|Command timed out|Timeout exceeded|Failed to execute command)",
            re.IGNORECASE,
        ),
    ),
    (
        "generic_error",
        re.compile(
            r"(command not found|No such file or directory|fatal error"
            r"|ModuleNotFoundError|ImportError|SyntaxError|IndentationError)",
            re.IGNORECASE,
        ),
    ),
)

_PATH_RE = re.compile(r"([\w./-]+\.py)")
_TEST_NODE_RE = re.compile(r"([\w./-]+::[\w.]+)")


@dataclass
class TrajectoryTurn:
    """One agent turn parsed from ``agent_logs.txt``."""

    index: int
    raw: str
    planning_text: str = ""
    tool_name: str = ""
    tool_input: str = ""
    tool_output: str = ""

    @property
    def has_tool_call(self) -> bool:
        return bool(self.tool_name)


@dataclass
class ErrorSignal:
    """A detected error inside one turn's tool output."""

    kind: str
    snippet: str  # verbatim evidence line
    signature: tuple[str, ...] = ()  # artifacts used to trace resolution


@dataclass
class ErrorEvent:
    """An error step with its lifecycle status."""

    turn_index: int
    tool_name: str
    signals: list[ErrorSignal]
    resolved: bool = False
    terminal_impact: bool = False

    @property
    def kind(self) -> str:
        return self.signals[0].kind if self.signals else "unknown"

    @property
    def evidence(self) -> str:
        return self.signals[0].snippet if self.signals else ""

    def signature_tokens(self) -> set[str]:
        tokens: set[str] = set()
        for signal in self.signals:
            tokens.update(signal.signature)
        return tokens


@dataclass
class CriticalFailure:
    """The earliest decisive, unresolved error responsible for the failure."""

    turn_index: int
    tool_name: str
    kind: str
    evidence: str  # verbatim
    terminal_impact: bool


@dataclass
class AttributionReport:
    """Full error-lifecycle analysis of one trajectory."""

    problem_statement: str
    num_turns: int
    errors: list[ErrorEvent]
    critical: CriticalFailure | None = None

    @property
    def unresolved(self) -> list[ErrorEvent]:
        return [error for error in self.errors if not error.resolved]


def _extract_instruction(preamble: str) -> str:
    if _USER_INPUT_MARKER in preamble:
        return preamble.split(_USER_INPUT_MARKER, 1)[1].strip()
    return ""


def parse_agent_log(log_text: str) -> tuple[str, list[TrajectoryTurn]]:
    """Split ``agent_logs.txt`` text into the instruction and a list of turns.

    The producer (``tools/agent.py``) wraps each turn in
    ``\\n{NEW TURN delimiter}\\n``; each turn body carries an optional
    ``Top-level agent planning next step:`` line and a
    ``Calling tool <name> with input: ... Tool output: ...`` block.
    """
    segments = log_text.split(_TURN_DELIMITER)
    problem_statement = _extract_instruction(segments[0] if segments else "")

    turns: list[TrajectoryTurn] = []
    for index, segment in enumerate(segments[1:]):
        turn = TrajectoryTurn(index=index, raw=segment)
        for line in segment.splitlines():
            if line.startswith(_PLANNING_PREFIX):
                turn.planning_text = line[len(_PLANNING_PREFIX) :].strip()
                break
        call_match = _TOOL_CALL_RE.search(segment)
        if call_match:
            turn.tool_name = call_match.group("name")
            after = segment[call_match.end() :]
            if _TOOL_OUTPUT_MARKER in after:
                raw_input, raw_output = after.split(_TOOL_OUTPUT_MARKER, 1)
                turn.tool_input = raw_input.strip()
                turn.tool_output = raw_output.strip()
            else:
                turn.tool_input = after.strip()
        elif _NO_TOOLS_MARKER in segment:
            turn.tool_output = _NO_TOOLS_MARKER
        turns.append(turn)
    return problem_statement, turns


def _signature_tokens(*texts: str) -> tuple[str, ...]:
    tokens: set[str] = set()
    for text in texts:
        tokens.update(_PATH_RE.findall(text))
        tokens.update(_TEST_NODE_RE.findall(text))
    return tuple(sorted(tokens))


def _detect_signals(text: str) -> list[ErrorSignal]:
    """Return the observable error signals in one tool output (verbatim)."""
    signals: list[ErrorSignal] = []
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped in seen:
            continue
        for kind, pattern in _ERROR_PATTERNS:
            if pattern.search(stripped):
                seen.add(stripped)
                signals.append(ErrorSignal(kind=kind, snippet=stripped))
                break
        if len(signals) >= 5:
            break
    return signals


def _trace_lifecycle(turns: list[TrajectoryTurn], errors: list[ErrorEvent]) -> None:
    """Mark each error's resolution status and terminal impact in place.

    An error is *resolved* when a later turn with no error signal references
    the same artifact (file path / pytest node id) -- e.g. a re-run pytest
    that passes on the same node, or a later successful edit to the same
    file. It has *terminal impact* when that same artifact is still
    referenced in the final turn.
    """
    if not turns:
        return
    clean: dict[int, bool] = {
        turn.index: not _detect_signals(turn.tool_output) for turn in turns
    }
    last_tokens = set(_signature_tokens(turns[-1].tool_output, turns[-1].tool_input))
    for error in errors:
        signature = error.signature_tokens()
        if signature and signature & last_tokens:
            error.terminal_impact = True
        for later in turns[error.turn_index + 1 :]:
            if not clean[later.index]:
                continue  # later turn still erroring -> not a clean fix
            if signature and signature & set(
                _signature_tokens(later.tool_output, later.tool_input)
            ):
                error.resolved = True
                break


def attribute_critical_failure(
    errors: list[ErrorEvent], is_success: bool | None
) -> CriticalFailure | None:
    """Pick the earliest decisive error responsible for the final failure.

    A succeeded trajectory has no critical failure. Otherwise the critical
    error is the earliest unresolved error with terminal impact; if none
    qualify (the failure cause is only partially traced), fall back to the
    earliest unresolved error so the diagnosis still points somewhere.
    """
    if is_success is True:
        return None
    unresolved = [error for error in errors if not error.resolved]
    if not unresolved:
        return None
    critical = next((error for error in unresolved if error.terminal_impact), None)
    if critical is None:
        critical = unresolved[0]
    return CriticalFailure(
        turn_index=critical.turn_index,
        tool_name=critical.tool_name,
        kind=critical.kind,
        evidence=critical.evidence,
        terminal_impact=critical.terminal_impact,
    )


def analyze_trajectory(
    log_text: str, is_success: bool | None = None
) -> AttributionReport:
    """Run full error-lifecycle attribution over one agent trajectory."""
    problem_statement, turns = parse_agent_log(log_text)
    errors: list[ErrorEvent] = []
    for turn in turns:
        if not turn.has_tool_call:
            continue
        signals = _detect_signals(turn.tool_output)
        if not signals:
            continue
        signature = _signature_tokens(turn.tool_output, turn.tool_input)
        for signal in signals:
            signal.signature = signature
        errors.append(
            ErrorEvent(turn_index=turn.index, tool_name=turn.tool_name, signals=signals)
        )
    _trace_lifecycle(turns, errors)
    critical = attribute_critical_failure(errors, is_success)
    return AttributionReport(
        problem_statement=problem_statement,
        num_turns=len(turns),
        errors=errors,
        critical=critical,
    )


def format_report(report: AttributionReport) -> str:
    """Render a human-readable diagnostic from an attribution report."""
    lines = [
        f"Analyzed {report.num_turns} turn(s); found {len(report.errors)} error "
        f"step(s), {len(report.unresolved)} unresolved."
    ]
    critical = report.critical
    if critical is None:
        lines.append(
            "No critical failure attributed (trajectory succeeded or all "
            "errors were resolved)."
        )
        return "\n".join(lines)
    lines.append("")
    lines.append(
        f"CRITICAL FAILURE -> earliest decisive error at turn "
        f"{critical.turn_index} (tool={critical.tool_name}, kind={critical.kind})"
    )
    lines.append(f"terminal impact: {'yes' if critical.terminal_impact else 'no'}")
    lines.append("verbatim evidence:")
    lines.append(f"  {critical.evidence}")
    return "\n".join(lines)


def report_to_dict(report: AttributionReport) -> dict[str, Any]:
    return asdict(report)
