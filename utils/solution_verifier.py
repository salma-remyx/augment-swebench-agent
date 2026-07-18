"""Continuous per-candidate verification scoring for solution selection.

Adapted from "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(arXiv:2607.05391). The paper scores candidate solutions with a *continuous*
signal -- an expectation over the distribution of scoring-token logits -- and
ranks them, instead of asking a judge to emit a single discrete pick. That
continuous, calibrated, decomposable score is what an argmax selects on, and
the paper shows it beats single-shot discrete majority voting.

This module keeps the paper's three scaling axes and its cost-efficient
ranking intact, while substituting one auxiliary component:

  * The paper's logprob-expectation estimator is replaced by a direct,
    parameter-free numeric elicitation. The repo's Anthropic Messages client
    does not expose token logprobs, so the continuous score is the model's
    elicited expected quality per criterion, averaged over repeated samples.
    Both formulations yield a continuous, calibrated, decomposable score; the
    selection property the paper relies on is preserved.

Preserved from the paper:
  * Score granularity -- a fine-grained 0-100 scale rather than a discrete pick.
  * Repeated evaluation -- when ``num_samples > 1`` the candidate is re-scored
    with stochastic sampling (a nonzero ``sample_temperature``) and the scores
    are averaged, so the mean is a Monte Carlo estimate of the score
    expectation. This is the parameter-free stand-in for the paper's expectation
    over scoring-token logits, and it is what actually reduces scoring variance:
    at ``temperature == 0`` repeated calls are deterministic and average to
    themselves, giving no variance reduction.
  * Criteria decomposition -- scoring each candidate on multiple criteria and
    aggregating reduces task complexity.
  * Cost-efficient ranking -- each candidate is scored independently (no
    all-pairs comparisons) and ranked by aggregate score, so cost scales
    linearly in the number of candidates. ``rank_candidates`` exposes the full
    continuous ranking; ``select`` returns its argmax.

``select`` exposes the same ``(instruction, candidates) -> selected index``
contract as the existing majority-vote ensembler, so it drops in as an
alternative selector.
"""

import re
from typing import Optional

from prompts.verifier_prompt import DEFAULT_CRITERIA, build_verifier_prompt
from utils.llm_client import LLMClient, TextPrompt, get_client

_SCORE_TAG_RE = re.compile(
    r'<score\s+crit(?:erion|eria)="[^"]*"\s*>\s*(\d+(?:\.\d+)?)\s*</score>',
    re.IGNORECASE,
)


def _extract_scores(response_text: str) -> list[float]:
    """Pull all numeric scores out of ``<score ...>N</score>`` tags."""
    return [float(m) for m in _SCORE_TAG_RE.findall(response_text)]


def score_candidate(
    instruction: str,
    candidate: str,
    client: LLMClient,
    num_samples: int = 1,
    criteria: tuple[str, ...] = DEFAULT_CRITERIA,
    max_score: int = 100,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    sample_temperature: float = 0.7,
) -> float:
    """Score a single candidate on a continuous 0-``max_score`` scale.

    The candidate is scored ``num_samples`` times (repeated evaluation) and on
    each criterion in ``criteria`` (criteria decomposition); the returned
    continuous score is the mean across samples and criteria. Returns 0.0 if
    the model emits no parseable score tags.

    A single sample uses ``temperature`` (deterministic by default). Repeated
    evaluation (``num_samples > 1``) instead samples at ``sample_temperature``
    so the per-sample scores vary and their mean estimates the score
    expectation -- averaging identical deterministic samples would reduce no
    variance.
    """
    prompt = build_verifier_prompt(instruction, candidate, criteria, max_score)
    messages = [[TextPrompt(text=prompt)]]

    effective_temperature = sample_temperature if num_samples > 1 else temperature

    sample_means: list[float] = []
    for _ in range(max(1, num_samples)):
        response, _metadata = client.generate(
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=effective_temperature,
        )
        first = response[0]
        text = first.text if hasattr(first, "text") else str(first)  # pyright: ignore[reportAttributeAccessIssue]
        scores = [min(max(s, 0.0), float(max_score)) for s in _extract_scores(text)]
        if scores:
            sample_means.append(sum(scores) / len(scores))

    if not sample_means:
        return 0.0
    return sum(sample_means) / len(sample_means)


def score_candidates(
    instruction: str,
    candidates: list[str],
    client: LLMClient,
    num_samples: int = 1,
    criteria: tuple[str, ...] = DEFAULT_CRITERIA,
    max_score: int = 100,
    sample_temperature: float = 0.7,
) -> list[float]:
    """Score every candidate independently, preserving input order.

    Independent per-candidate scoring is the paper's cost-efficient ranking
    primitive: no pairwise comparisons, so cost scales linearly in the number
    of candidates rather than quadratically.
    """
    return [
        score_candidate(
            instruction,
            candidate,
            client,
            num_samples,
            criteria,
            max_score,
            sample_temperature=sample_temperature,
        )
        for candidate in candidates
    ]


def rank_candidates(
    instruction: str,
    candidates: list[str],
    client: LLMClient,
    num_samples: int = 1,
    criteria: tuple[str, ...] = DEFAULT_CRITERIA,
    max_score: int = 100,
    sample_temperature: float = 0.7,
) -> list[tuple[int, float]]:
    """Rank candidates by continuous verification score, best first.

    Returns ``(original_index, score)`` pairs sorted by descending score; ties
    preserve input order (stable sort). Exposing the full continuous ranking --
    not just the winner -- is what the paper's fine-grained score is for: it can
    drive candidate selection, serve as a task-progress proxy, or feed a reward
    signal, all of which a single discrete pick discards.
    """
    scores = score_candidates(
        instruction,
        candidates,
        client,
        num_samples,
        criteria,
        max_score,
        sample_temperature=sample_temperature,
    )
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda pair: pair[1], reverse=True)
    return indexed


def select(
    instruction: str,
    candidates: list[str],
    client: Optional[LLMClient] = None,
    num_samples: int = 1,
    criteria: tuple[str, ...] = DEFAULT_CRITERIA,
    max_score: int = 100,
    sample_temperature: float = 0.7,
) -> Optional[int]:
    """Select the best candidate by continuous verification score.

    Returns the 0-based index of the highest-scoring candidate, or ``None`` if
    there are no candidates. Ties are broken toward the earliest candidate.

    ``client`` defaults to the repo's Anthropic direct client (Claude Sonnet 4
    by default), matching the ensembler's model convention.
    """
    if not candidates:
        return None
    if client is None:
        client = get_client("anthropic-direct")

    ranking = rank_candidates(
        instruction,
        candidates,
        client,
        num_samples,
        criteria,
        max_score,
        sample_temperature=sample_temperature,
    )
    return ranking[0][0]
