"""Coarse-to-fine, conditional refinement for best-of-N solution selection.

Adapted port of "MAgICoRe: Multi-Agent, Iterative, Coarse-to-Fine Refinement
for Reasoning" (arXiv:2409.12147). The repo's ensembler already implements the
paper's first two stages -- sample N candidate solutions and vote/select the
best (``majority_vote_ensembler.process_problem``). This module adds the
paper's *novel* third stage: a refinement loop that fires only on
low-consensus problems.

Refinement alone helps, but the paper's central finding is that *uniformly*
refining every problem over-corrects and lowers overall accuracy (its first
stated challenge). MAgICoRe therefore gates refinement on a confidence signal:
only problems whose candidate solutions disagree (low consensus / "hard") are
refined, while high-consensus ("easy") problems are returned as-is. The
refinement itself is multi-agent and iterative -- a *reviewer* critiques the
selected solution and a *solver* rewrites it from that critique, repeated up to
``max_rounds`` times and stopped early when the reviewer finds nothing left to
fix.

This module keeps the paper's mechanism intact -- the consensus gate, the
reviewer/solver roles, and the iterative early-stopping loop -- while
substituting two auxiliary components the ensembler stage cannot host:

  * The paper estimates problem difficulty by clustering the N sampled
    solutions and checking for a dominant cluster, using its own clustering /
    embedding pipeline. The ensembler has no such pipeline and no execution
    sandbox, so the gate is a parameter-free *candidate-similarity* proxy
    (:func:`consensus_from_candidates`): how strongly the selected candidate
    agrees with the rest of the pool. High agreement -> high consensus -> do
    not refine (the over-correction guard). This is the stand-in for the
    paper's cluster-dominance estimator and requires no learned model.

  * The paper's solver is a full tool-using agent that re-runs against the
    target repository (it can execute tests, edit files, observe failures).
    The ensembler stage operates on already-generated diffs with no live
    checkout, so the solver here is a single LLM turn that revises the diff
    from the reviewer's feedback via the repo's existing ``LLMClient``. The
    refinement is feedback-driven rather than execution-driven.

Preserved from the paper:
  * Consensus gate -- only low-consensus problems are refined; high-consensus
    problems pass through untouched (avoids excessive refinement).
  * Multi-agent refinement -- a reviewer produces the critique and a separate
    solver produces the edit, rather than one self-correcting call.
  * Iterative early stop -- the loop runs up to ``max_rounds`` rounds and halts
    as soon as the reviewer reports the solution needs no further change.

``refine`` fits the ensembler's selection contract: it takes the selected
candidate (and the pool it was drawn from) and returns either the original
diff (high consensus / reviewer satisfied) or a refined one.
"""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from prompts.refiner_prompt import build_review_prompt, build_solve_prompt
from utils.llm_client import LLMClient, TextPrompt, get_client

MAX_TOKENS = 16384
DEFAULT_CONSENSUS_THRESHOLD = 0.5
DEFAULT_MAX_ROUNDS = 3
DEFAULT_SAMPLE_TEMPERATURE = 0.7

_NEEDS_RE = re.compile(
    r"<needs_refinement>\s*(true|false)\s*</needs_refinement>",
    re.IGNORECASE,
)
_FEEDBACK_RE = re.compile(r"<feedback>(.*?)</feedback>", re.DOTALL | re.IGNORECASE)
_DIFF_RE = re.compile(r"<refined_diff>(.*?)</refined_diff>", re.DOTALL | re.IGNORECASE)


@dataclass
class ReviewResult:
    """One reviewer turn: whether the candidate still needs work, and why."""

    needs_refinement: bool
    feedback: str


@dataclass
class RefineResult:
    """Outcome of the coarse-to-fine refine loop.

    ``refined`` is True only when the returned ``diff`` differs from the
    originally-selected candidate. ``stopped_reason`` is one of
    ``high_consensus`` (gate closed), ``reviewer_satisfied`` (early stop),
    ``solver_empty`` (solver returned nothing usable), ``max_rounds`` (loop
    budget exhausted), or ``no_candidate`` (nothing to refine).
    """

    diff: Optional[str]
    refined: bool
    rounds: int
    consensus: Optional[float]
    stopped_reason: str


def _change_lines(diff: str) -> str:
    """The added/removed lines of a unified diff (the actual change), minus the
    ``+++``/``---`` file headers. Falls back to the full text when there are no
    change lines so similarity is still defined for non-diff inputs."""
    lines = []
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            lines.append(line)
    return "\n".join(lines) if lines else diff


def candidate_similarity(a: str, b: str) -> float:
    """Normalized agreement in ``[0, 1]`` between two candidate diffs, computed
    over their change lines so shared unchanged context does not inflate it."""
    return SequenceMatcher(None, _change_lines(a), _change_lines(b)).ratio()


def consensus_from_candidates(
    candidates: list[str], selected_index: Optional[int]
) -> float:
    """Parameter-free consensus proxy in ``[0, 1]``: the mean similarity of the
    selected candidate to the rest of the pool.

    This stands in for MAgICoRe's cluster-dominance difficulty estimator: a
    winner that closely matches most other samples signals high agreement
    (an "easy" problem -> do not refine), while a winner surrounded by very
    different samples signals disagreement (a "hard" problem -> refine). A
    single-candidate pool is trivially unanimous (1.0).
    """
    if not candidates or selected_index is None:
        return 1.0
    if len(candidates) <= 1:
        return 1.0
    if not 0 <= selected_index < len(candidates):
        return 1.0
    selected = candidates[selected_index]
    scores = [
        candidate_similarity(selected, c)
        for i, c in enumerate(candidates)
        if i != selected_index
    ]
    return sum(scores) / len(scores) if scores else 1.0


def _first_text(response: list) -> str:
    """Pull the text off the first response block, mirroring the ensembler."""
    first = response[0]
    return first.text if hasattr(first, "text") else str(first)  # pyright: ignore[reportAttributeAccessIssue]


def review(
    instruction: str,
    candidate_diff: str,
    client: LLMClient,
    sample_temperature: float = DEFAULT_SAMPLE_TEMPERATURE,
) -> ReviewResult:
    """One reviewer turn: critique the candidate and flag whether it needs work.

    Returns ``needs_refinement=False`` (no feedback) when the tag is missing or
    reads ``false`` -- the safe default is to stop refining rather than churn
    an already-good solution (the over-correction guard).
    """
    prompt = build_review_prompt(instruction, candidate_diff)
    messages = [[TextPrompt(text=prompt)]]
    response, _metadata = client.generate(
        messages=messages,  # type: ignore
        max_tokens=MAX_TOKENS,
        temperature=sample_temperature,
    )
    text = _first_text(response)
    match = _NEEDS_RE.search(text)
    needs = bool(match and match.group(1).lower() == "true")
    feedback_match = _FEEDBACK_RE.search(text)
    feedback = feedback_match.group(1).strip() if feedback_match else ""
    return ReviewResult(needs_refinement=needs, feedback=feedback)


def solve(
    instruction: str,
    candidate_diff: str,
    feedback: str,
    client: LLMClient,
    sample_temperature: float = DEFAULT_SAMPLE_TEMPERATURE,
) -> Optional[str]:
    """One solver turn: produce an improved diff acting on the reviewer's
    feedback, or ``None`` if no ``<refined_diff>`` block was emitted."""
    prompt = build_solve_prompt(instruction, candidate_diff, feedback)
    messages = [[TextPrompt(text=prompt)]]
    response, _metadata = client.generate(
        messages=messages,  # type: ignore
        max_tokens=MAX_TOKENS,
        temperature=sample_temperature,
    )
    text = _first_text(response)
    match = _DIFF_RE.search(text)
    if not match:
        return None
    refined = match.group(1).strip()
    return refined or None


def refine(
    instruction: str,
    candidates: list[str],
    selected_index: Optional[int],
    selected_diff: Optional[str],
    consensus: Optional[float] = None,
    client: Optional[LLMClient] = None,
    threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    sample_temperature: float = DEFAULT_SAMPLE_TEMPERATURE,
) -> RefineResult:
    """Run the coarse-to-fine refine loop on the selected candidate.

    Stage gate (coarse): if the candidate pool consensus is at or above
    ``threshold``, the problem is "easy" and the candidate is returned untouched
    -- this is the guard against excessive refinement. Refinement (fine) only
    runs on low-consensus problems, iterating reviewer -> solver up to
    ``max_rounds`` times and stopping early when the reviewer is satisfied.

    ``consensus`` defaults to :func:`consensus_from_candidates` (the
    parameter-free proxy). ``client`` defaults to the repo's Anthropic direct
    client (Claude Sonnet 4), matching the ensembler's model convention.
    """
    if selected_diff is None:
        return RefineResult(None, False, 0, None, "no_candidate")
    if client is None:
        client = get_client("anthropic-direct")
    if consensus is None:
        consensus = consensus_from_candidates(candidates, selected_index)

    # Coarse stage: high consensus -> do not refine (over-correction guard).
    if consensus >= threshold:
        return RefineResult(selected_diff, False, 0, consensus, "high_consensus")

    # Fine stage: iterative reviewer -> solver, only on low-consensus problems.
    current = selected_diff
    rounds = 0
    for _ in range(max(1, max_rounds)):
        verdict = review(instruction, current, client, sample_temperature)
        rounds += 1
        if not verdict.needs_refinement:
            return RefineResult(
                current,
                current != selected_diff,
                rounds,
                consensus,
                "reviewer_satisfied",
            )
        refined = solve(
            instruction, current, verdict.feedback, client, sample_temperature
        )
        if not refined:
            return RefineResult(
                current, current != selected_diff, rounds, consensus, "solver_empty"
            )
        current = refined
    return RefineResult(
        current, current != selected_diff, rounds, consensus, "max_rounds"
    )


def refine_or_none(
    instruction: str,
    candidates: list[str],
    selected_index: Optional[int],
    selected_diff: Optional[str],
    client: Optional[LLMClient] = None,
    threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> Optional[str]:
    """Run :func:`refine` and return the diff only when it was actually refined.

    This is the ``(instruction, candidates, selected_index, selected_diff) ->
    diff | None`` contract the ensembler's refiner hook expects: ``None`` means
    "keep the originally-selected candidate" (high consensus, reviewer
    satisfied, solver empty, or no candidate), a string means "use this refined
    diff instead".
    """
    result = refine(
        instruction,
        candidates,
        selected_index,
        selected_diff,
        client=client,
        threshold=threshold,
        max_rounds=max_rounds,
    )
    return result.diff if result.refined else None
