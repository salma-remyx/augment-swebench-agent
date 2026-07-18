"""Task-complexity-aware execution-scope estimation (E3 *Estimate* step).

Adapted from "Do AI Agents Know When a Task Is Simple? Toward Complexity-Aware
Reasoning and Execution" (arXiv:2607.13034). That paper's E3 (Estimate, Execute,
Expand) framework observes that agents habitually over-spend context on simple
tasks -- re-reading files they have already seen -- and proposes estimating a
*minimum-sufficient* operating point before committing the full budget: estimate
an initial operating point, execute a minimum viable path, and expand scope only
when that path does not verify.

This module implements the **Estimate** step: a parameter-free lexical-cue
classifier with an optional single grep probe that produces a
:class:`TaskOperatingPoint` (a complexity tier plus a minimum-viable turn
budget). :meth:`tools.agent.Agent.run_impl` consumes that budget as the
first-phase ("Execute the minimum viable path") turn count and runs the
**Expand** step in place: when the minimum-viable budget is exhausted without
the agent completing, the loop is granted the remaining turns up to the
agent's original ``max_turns`` cap -- so a correct estimate makes a simple task
cheap, while a wrong estimate can never do worse than the baseline budget.

Implementation mode (Mode 2 -- adapted port):

* The estimator itself is the paper's own parameter-free design kept at full
  fidelity: lexical-cue classification plus a single grep probe. No learned
  estimator and no LLM call is substituted in -- the paper does not use one.
* The paper's MSE-Bench / LLM-Case evaluation harness and its ACRR
  formalization are intentionally out of scope here; evaluation belongs in a
  downstream PR. What ships is the Estimate mechanism wired into the live
  agent loop.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ComplexityTier(Enum):
    """Coarse task-difficulty tiers.

    Each tier maps to a default minimum-viable turn budget via
    :data:`DEFAULT_TURNS_BY_TIER`.
    """

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


# Minimum-viable first-phase turn budgets per tier. These are deliberately
# *smaller* than a typical full cap (the agent's ``max_turns``): a correct
# estimate lets a simple task finish in a handful of turns, while the Expand
# phase in ``Agent.run_impl`` preserves the full cap as a safety net whenever
# the estimate turns out to be too tight.
DEFAULT_TURNS_BY_TIER: dict[ComplexityTier, int] = {
    ComplexityTier.SIMPLE: 3,
    ComplexityTier.MODERATE: 6,
    ComplexityTier.COMPLEX: 10,
}

# Score is anchored at NEUTRAL (MODERATE) and nudged up/down by cues.
NEUTRAL_SCORE = 0.5
SIMPLE_UPPER_BOUND = 0.35
COMPLEX_LOWER_BOUND = 0.65


@dataclass(frozen=True)
class TaskOperatingPoint:
    """The estimated minimum-sufficient operating point for a task.

    Attributes:
        tier: Estimated complexity tier.
        initial_turns: Minimum-viable first-phase turn budget. The caller is
            expected to clamp this against its own hard turn cap.
        score: Numeric complexity score in ``[0.0, 1.0]`` (higher = more
            complex); the aggregate of every cue that fired.
        signals: Human-readable cues that fired, for logging/debugging.
    """

    tier: ComplexityTier
    initial_turns: int
    score: float
    signals: tuple[str, ...] = ()


# Lexical cues. Each entry is ``(compiled regex, weight)``. A positive weight
# nudges the score toward COMPLEX; a negative weight nudges toward SIMPLE. The
# cue sets are the paper's "lexical-cue classification" signal.
_COMPLEX_CUES: tuple[tuple[re.Pattern[str], float, str], ...] = tuple(
    (re.compile(pattern), weight, label)
    for pattern, weight, label in (
        (r"\brefactor(?:ing|ed)?\b", 0.25, "refactor"),
        (r"\brewrite\b|\bredesign\b", 0.25, "rewrite/redesign"),
        (r"\bmigrat(?:e|ion)\b", 0.25, "migrate"),
        (r"\bfrom scratch\b|\bground[- ]?up\b", 0.25, "from-scratch"),
        (r"\bacross (?:the|this|all) \w*repo\w*\b", 0.20, "across-repo"),
        (
            r"\bentire codebase\b|\bwhole (?:repo|project|codebase)\b",
            0.20,
            "whole-codebase",
        ),
        (r"\bend[- ]to[- ]end\b|\be2e\b", 0.20, "end-to-end"),
        (r"\bnew feature\b|\badd (?:a |an )?feature\b", 0.20, "new-feature"),
        (
            r"\bconcurren(?:cy|t)\b|\brace[- ]condition\b|\bdeadlock\b|\bthread[- ]?safe\b",
            0.20,
            "concurrency",
        ),
        (r"\bmemory leak\b|\bsegfault\b|\bcrash\b", 0.20, "defect"),
        (r"\barchitect(?:ure|ural)\b|\bpipeline\b|\bworkflow\b", 0.15, "architecture"),
        (r"\bimplement(?:ation|ed|ing)?\b", 0.15, "implement"),
        (r"\bperformance\b|\boptimi[sz]e\b|\bprofiling\b", 0.15, "performance"),
        (r"\bcomprehensive\b|\bexhaustive\b|\bthorough\b", 0.10, "comprehensive"),
        (r"\b(?:multiple|several|many) files\b", 0.15, "many-files"),
        (r"\btest suite\b|\bunit tests?\b|\bintegration tests?\b", 0.10, "tests"),
    )
)

_SIMPLE_CUES: tuple[tuple[re.Pattern[str], float, str], ...] = tuple(
    (re.compile(pattern), weight, label)
    for pattern, weight, label in (
        (r"\btypo\b", 0.30, "typo"),
        (r"\brename(?:d|ing)?\b", 0.25, "rename"),
        (r"\bspelling\b", 0.25, "spelling"),
        (r"\bone[- ]liner\b|\bone line\b", 0.25, "one-liner"),
        (r"\bdocstring\b|\bcomment\b", 0.15, "doc/comment"),
        (r"\bwhitespace\b|\bindentation\b|\bformat(?:ting)?\b", 0.15, "format"),
        (r"\bimport\b", 0.15, "import"),
        (r"\bbump\b|\bversion\b", 0.15, "version-bump"),
        (r"\bmissing\b", 0.10, "missing"),
        (r"\blog(?:ging| line|s)?\b", 0.10, "logging"),
    )
)

# Words ignored when extracting grep-probe identifiers from the instruction.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "into",
        "your",
        "you",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "but",
        "not",
        "all",
        "any",
        "can",
        "will",
        "should",
        "would",
        "could",
        "than",
        "then",
        "them",
        "these",
        "those",
        "there",
        "their",
        "when",
        "where",
        "which",
        "what",
        "please",
        "make",
        "made",
        "need",
        "needs",
        "must",
        "may",
        "might",
        "use",
        "used",
        "using",
        "get",
        "set",
        "put",
        "add",
        "new",
        "old",
        "one",
        "two",
        "fix",
        "bug",
        "issue",
        "task",
        "code",
        "file",
        "files",
        "function",
        "method",
        "class",
        "repo",
        "project",
        "test",
        "tests",
        "line",
        "lines",
    }
)


def _tier_from_score(score: float) -> ComplexityTier:
    """Map a numeric complexity score to a coarse tier."""
    if score < SIMPLE_UPPER_BOUND:
        return ComplexityTier.SIMPLE
    if score < COMPLEX_LOWER_BOUND:
        return ComplexityTier.MODERATE
    return ComplexityTier.COMPLEX


@dataclass
class TaskComplexityEstimator:
    """Estimate a task's minimum-sufficient operating point (E3 *Estimate*).

    The estimator is parameter-free: it reads lexical cues off the instruction,
    optionally probes the workspace surface area with a *single* grep, and
    returns an operating point. It never calls the model -- the paper's
    estimate step is a cheap pre-filter, not an extra inference call.
    """

    turns_by_tier: dict[ComplexityTier, int] = field(
        default_factory=lambda: dict(DEFAULT_TURNS_BY_TIER)
    )

    def estimate(
        self,
        instruction: str,
        workspace_root: Optional[Path | str] = None,
    ) -> TaskOperatingPoint:
        """Estimate the operating point for ``instruction``.

        Args:
            instruction: The task instruction the agent is about to execute.
            workspace_root: Optional workspace root for the single grep probe.
                When omitted, only the lexical-cue classifier is used.

        Returns:
            A :class:`TaskOperatingPoint` with the estimated tier, a
            minimum-viable turn budget, the aggregate score, and the cues that
            fired.
        """
        text = (instruction or "").strip()
        lowered = text.lower()
        score = NEUTRAL_SCORE
        signals: list[str] = []

        for pattern, weight, label in _COMPLEX_CUES:
            if pattern.search(lowered):
                score += weight
                signals.append(f"+{label}")

        for pattern, weight, label in _SIMPLE_CUES:
            if pattern.search(lowered):
                score -= weight
                signals.append(f"-{label}")

        # Instruction length is a weak secondary signal: very short prompts
        # rarely need a code-base audit, very long ones usually do.
        word_count = len(text.split())
        if word_count <= 12:
            score -= 0.10
            signals.append("short-instruction")
        elif word_count >= 60:
            score += 0.15
            signals.append("long-instruction")

        # The single grep probe: if a workspace is available, gauge how many
        # files mention the salient identifiers named in the instruction. A
        # wide surface area bumps the estimate; a near-empty one nudges it down.
        if workspace_root is not None:
            surface = self._grep_probe(text, Path(workspace_root))
            if surface is not None:
                signals.append(f"grep-files={surface}")
                if surface >= 8:
                    score += 0.20
                elif surface <= 1:
                    score -= 0.10

        score = max(0.0, min(1.0, score))
        tier = _tier_from_score(score)
        return TaskOperatingPoint(
            tier=tier,
            initial_turns=self.turns_by_tier[tier],
            score=score,
            signals=tuple(signals),
        )

    @staticmethod
    def _extract_identifiers(instruction: str) -> list[str]:
        """Pull salient identifiers (to grep for) out of the instruction."""
        candidates: set[str] = set()
        for match in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", instruction):
            if match.lower() not in _STOPWORDS:
                candidates.add(match)
        for quoted in re.findall(r"[\"']([A-Za-z0-9_./\-]{2,})[\"']", instruction):
            candidates.add(quoted)
        # Prefer the longest identifiers -- they are the most specific targets.
        return sorted(candidates, key=len, reverse=True)[:3]

    @staticmethod
    def _grep_probe(instruction: str, workspace_root: Path) -> Optional[int]:
        """Run a *single* recursive grep and return the matching file count.

        Returns ``None`` when the probe cannot run (no identifiers to search
        for, missing workspace, or a grep error/timeout).
        """
        identifiers = TaskComplexityEstimator._extract_identifiers(instruction)
        if not identifiers or not workspace_root.exists():
            return None
        pattern = "|".join(re.escape(identifier) for identifier in identifiers)
        try:
            result = subprocess.run(
                [
                    "grep",
                    "-rlE",
                    "--exclude-dir=.git",
                    pattern,
                    str(workspace_root),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        # grep: returncode 0 => matches, 1 => no matches, >=2 => error.
        if result.returncode not in (0, 1):
            return None
        return sum(1 for line in result.stdout.splitlines() if line.strip())
