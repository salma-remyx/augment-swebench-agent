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
  * Repeated evaluation -- averaging ``num_samples`` independent scores reduces
    scoring variance.
  * Criteria decomposition -- scoring each candidate on multiple criteria and
    aggregating reduces task complexity.
  * Cost-efficient ranking -- each candidate is scored independently (no
    all-pairs comparisons) and selected by argmax over aggregate scores.

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
) -> float:
    """Score a single candidate on a continuous 0-``max_score`` scale.

    The candidate is scored ``num_samples`` times (repeated evaluation) and on
    each criterion in ``criteria`` (criteria decomposition); the returned
    continuous score is the mean across samples and criteria. Returns 0.0 if
    the model emits no parseable score tags.
    """
    prompt = build_verifier_prompt(instruction, candidate, criteria, max_score)
    messages = [[TextPrompt(text=prompt)]]

    sample_means: list[float] = []
    for _ in range(max(1, num_samples)):
        response, _metadata = client.generate(
            messages=messages,  # type: ignore
            max_tokens=max_tokens,
            temperature=temperature,
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
) -> list[float]:
    """Score every candidate independently, preserving input order.

    Independent per-candidate scoring is the paper's cost-efficient ranking
    primitive: no pairwise comparisons, so cost scales linearly in the number
    of candidates rather than quadratically.
    """
    return [
        score_candidate(
            instruction, candidate, client, num_samples, criteria, max_score
        )
        for candidate in candidates
    ]


def select(
    instruction: str,
    candidates: list[str],
    client: Optional[LLMClient] = None,
    num_samples: int = 1,
    criteria: tuple[str, ...] = DEFAULT_CRITERIA,
    max_score: int = 100,
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

    scores = score_candidates(
        instruction, candidates, client, num_samples, criteria, max_score
    )
    best_index = 0
    best_score = scores[0]
    for offset, score in enumerate(scores[1:], start=1):
        if score > best_score:
            best_score = score
            best_index = offset
    return best_index
