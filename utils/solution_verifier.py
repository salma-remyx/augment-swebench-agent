"""Pairwise fine-grained verification for best-of-N solution selection.

Ported from "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(arXiv:2607.05391). The paper selects among candidate trajectories with a
*pairwise* fine-grained reward model: the verifier sees two trajectories A and B
together and emits a fine-grained score for each. Those directed comparisons are
aggregated by a Probabilistic Pivot Tournament (PPT) -- a ring pass that cancels
slot bias, pivot selection, pivot rounds, and Bradley-Terry soft wins -- so the
best of N candidates is found in O(Nk) verifier calls rather than O(N^2).

This module keeps the paper's mechanism intact -- pairwise reward, per-criterion
isolation, repeated evaluation, and the PPT / Bradley-Terry aggregation -- while
substituting one auxiliary component:

  * The paper reads the verifier's probability distribution over an ordered set
    of score tokens and takes its expectation (needing a logprob-exposing
    backend such as Vertex, vLLM, or SGLang). The repo's Anthropic Messages
    client does not expose token logprobs, so the reward is a Monte Carlo
    estimate of that expectation instead: the pair is scored ``n_evaluations``
    times with stochastic sampling and the per-slot scores are averaged. This is
    the parameter-free stand-in for the score-token expectation, and it is what
    reduces scoring variance -- at temperature 0 repeated calls are
    deterministic and average to themselves.

Preserved from the paper:
  * Pairwise reward -- candidates are scored two-at-a-time in a directed A/B
    comparison, not independently.
  * Criteria decomposition -- each criterion is scored in its OWN isolated
    verifier call and the rewards are averaged, reducing per-call complexity.
  * Repeated evaluation -- ``n_evaluations`` stochastic samples per
    (criterion, directed pair), averaged (the Monte Carlo expectation above).
  * Probabilistic Pivot Tournament -- ring pass + pivots + Bradley-Terry
    aggregation, so cost is O(Nk) rather than O(N^2).

``select`` exposes the same ``(instruction, candidates) -> selected index``
contract as the existing majority-vote ensembler, so it drops in as an
alternative selector.
"""

import random
import re
from typing import Optional

from prompts.verifier_prompt import DEFAULT_CRITERIA, build_verifier_prompt
from utils import pivot_tournament as ppt
from utils.llm_client import LLMClient, TextPrompt, get_client

MAX_TOKENS = 4096
DEFAULT_N_EVALUATIONS = 8
DEFAULT_SAMPLE_TEMPERATURE = 0.7


def _extract_score(response_text: str, slot: str, max_score: int) -> Optional[float]:
    """Pull the numeric score for one slot out of ``<score_X>N</score_X>``.

    Returns the score normalized to ``[0, 1]``, or ``None`` if the tag is
    absent so the caller can fall back to a tie.
    """
    match = re.search(
        rf"<score_{slot}>\s*(\d+(?:\.\d+)?)\s*</score_{slot}>",
        response_text,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = min(max(float(match.group(1)), 0.0), float(max_score))
    return value / max_score


def score_pair_criterion(
    problem: str,
    trace_a: str,
    trace_b: str,
    criterion: dict,
    client: LLMClient,
    n_evaluations: int = DEFAULT_N_EVALUATIONS,
    max_score: int = 100,
    sample_temperature: float = DEFAULT_SAMPLE_TEMPERATURE,
) -> tuple[float, float]:
    """Fine-grained rewards ``(R_a, R_b)`` in ``[0, 1]`` for one directed pair
    on a SINGLE criterion.

    The verifier sees ``trace_a`` in slot A and ``trace_b`` in slot B and scores
    both on ``criterion`` only (criteria decomposition). The pair is scored
    ``n_evaluations`` times and the per-slot scores are averaged -- the Monte
    Carlo estimate of the paper's score-token expectation. A slot with no
    parseable score defaults to a tie (0.5). Repeated evaluation
    (``n_evaluations > 1``) samples at ``sample_temperature`` so the per-sample
    scores vary; a single evaluation is deterministic (temperature 0.0).
    """
    prompt = build_verifier_prompt(problem, trace_a, trace_b, criterion, max_score)
    messages = [[TextPrompt(text=prompt)]]

    effective_temperature = sample_temperature if n_evaluations > 1 else 0.0

    a_samples: list[float] = []
    b_samples: list[float] = []
    for _ in range(max(1, n_evaluations)):
        response, _metadata = client.generate(
            messages=messages,  # type: ignore
            max_tokens=MAX_TOKENS,
            temperature=effective_temperature,
        )
        first = response[0]
        text = first.text if hasattr(first, "text") else str(first)  # pyright: ignore[reportAttributeAccessIssue]
        ra = _extract_score(text, "A", max_score)
        rb = _extract_score(text, "B", max_score)
        if ra is not None:
            a_samples.append(ra)
        if rb is not None:
            b_samples.append(rb)

    r_a = sum(a_samples) / len(a_samples) if a_samples else 0.5
    r_b = sum(b_samples) / len(b_samples) if b_samples else 0.5
    return r_a, r_b


def directed_reward(
    problem: str,
    trace_a: str,
    trace_b: str,
    client: LLMClient,
    criteria: tuple[dict, ...] = DEFAULT_CRITERIA,
    n_evaluations: int = DEFAULT_N_EVALUATIONS,
    max_score: int = 100,
    sample_temperature: float = DEFAULT_SAMPLE_TEMPERATURE,
) -> tuple[float, float]:
    """Fine-grained rewards ``(R_a, R_b)`` for the directed comparison (``a`` in
    slot A, ``b`` in slot B), averaged over all criteria.

    Each criterion is scored in its own isolated verifier call
    (:func:`score_pair_criterion`); the returned reward averages those
    per-criterion rewards. Returns a tie (0.5, 0.5) if there are no criteria.
    """
    sa = sb = 0.0
    cnt = 0
    for criterion in criteria:
        ra, rb = score_pair_criterion(
            problem,
            trace_a,
            trace_b,
            criterion,
            client,
            n_evaluations,
            max_score,
            sample_temperature,
        )
        sa += ra
        sb += rb
        cnt += 1
    return (sa / cnt, sb / cnt) if cnt else (0.5, 0.5)


def select(
    instruction: str,
    candidates: list[str],
    client: Optional[LLMClient] = None,
    criteria: tuple[dict, ...] = DEFAULT_CRITERIA,
    n_evaluations: int = DEFAULT_N_EVALUATIONS,
    pivots: int = ppt.DEFAULT_PIVOTS,
    seed: int = 0,
    max_score: int = 100,
    sample_temperature: float = DEFAULT_SAMPLE_TEMPERATURE,
) -> Optional[int]:
    """Select the best of N candidates with a Probabilistic Pivot Tournament.

    Directed pairs of candidates are scored with the pairwise fine-grained
    reward and aggregated by PPT (ring pass + pivots + Bradley-Terry soft wins
    normalized by comparison count), so the cost is O(Nk) verifier comparisons
    rather than the O(N^2) of a full round-robin. Returns the 0-based index of
    the winning candidate, or ``None`` if there are no candidates. Identical
    inputs with the same ``seed`` run the identical tournament.

    ``client`` defaults to the repo's Anthropic direct client (Claude Sonnet 4
    by default), matching the ensembler's model convention.
    """
    if not candidates:
        return None
    n = len(candidates)
    if n == 1:
        return 0
    if client is None:
        client = get_client("anthropic-direct")

    # Memoize directed comparisons so a pair scored in the ring pass is not
    # re-scored if it recurs in the pivot rounds (the reference achieves this
    # with an on-disk score cache).
    memo: dict[tuple[int, int], tuple[float, float]] = {}

    def score(a: int, b: int) -> tuple[float, float]:
        if (a, b) not in memo:
            memo[(a, b)] = directed_reward(
                instruction,
                candidates[a],
                candidates[b],
                client,
                criteria,
                n_evaluations,
                max_score,
                sample_temperature,
            )
        return memo[(a, b)]

    rng = random.Random(seed)
    ring = ppt.ring_cycle(n, rng)
    best, _n_comparisons = ppt.select_best(n, ring, pivots, score)
    return best
