"""Multi-judge voting and consensus selection for best-of-N solution choice.

Adapted from "Voting or Consensus? Decision-Making in Multi-Agent Debate"
(arXiv:2502.19130). The paper's controlled study isolates the DECISION-MAKING
PROTOCOL -- how the answers of several agents are aggregated into one final
answer -- and finds that, for reasoning tasks, VOTING (each agent independently
picks an answer; the plurality wins) beats a single decision-maker and matches
or beats CONSENSUS (a chair agent aggregates the agents' answers). SWE-bench
patch selection is a reasoning task, so this ports the paper's two headline
protocols onto the ensembler's existing ``(instruction, candidates) -> selected
index`` selector contract, alongside the single-shot o1 pick and the
LLM-as-a-Verifier tournament.

This is a Mode 2 (adapted) port: the protocol is the paper's actual contribution,
while the multi-agent DEBATE that surrounds it (agents exchanging messages over
rounds) is auxiliary machinery this repo does not host. That debate is replaced
by the repo's native primitive -- ``n_judges`` independent judging calls, each of
which independently picks one candidate. The two aggregation protocols the paper
compares are kept intact:

  * Voting -- each judge's pick is a single vote; the candidate with the most
    valid votes wins (plurality), ties broken toward the earliest candidate. This
    is the protocol the paper recommends for reasoning tasks.
  * Consensus -- the judges' individual picks are surfaced to a chair agent that
    makes the final selection, the "single answer synthesized from many opinions"
    shape the paper contrasts against voting. Falls back to the plurality winner
    if the chair is unparseable, so consensus never fails where voting succeeds.

Preserved from the paper:
  * Protocol isolation -- the aggregation rule is the only thing that changes
    between voting and consensus; the judges and their prompt are identical, so
    the comparison is controlled exactly as in the paper.
  * Voting (plurality) aggregation with ties toward a stable default.
  * Consensus aggregation via a chair that sees the panel's picks.
  * Independent judges whose disagreement (via sampling temperature) is the
    diversity that makes voting non-degenerate.

``select`` exposes the same ``(instruction, candidates) -> selected index``
contract as the existing majority-vote ensembler, so it drops in as an
alternative selector.
"""

import re
from typing import Optional

from prompts.judge_prompt import build_judge_prompt
from utils.llm_client import LLMClient, TextPrompt, get_client

MAX_TOKENS = 1024
DEFAULT_N_JUDGES = 5
DEFAULT_SAMPLE_TEMPERATURE = 0.7


def _extract_vote(response_text: str) -> Optional[int]:
    """Pull the 1-based ``<vote>N</vote>`` from a response and make it 0-based.

    Returns ``None`` if the tag is absent so the caller can skip the vote.
    """
    match = re.search(r"<vote>\s*(\d+)\s*</vote>", response_text, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)) - 1


def judge_votes(
    instruction: str,
    candidates: list[str],
    client: LLMClient,
    n_judges: int = DEFAULT_N_JUDGES,
    sample_temperature: float = DEFAULT_SAMPLE_TEMPERATURE,
) -> list[Optional[int]]:
    """Collect one independent vote per judge.

    Each judge sees the same instruction + candidates and independently picks a
    single candidate. Voting with ``n_judges > 1`` samples at
    ``sample_temperature`` so the judges disagree -- the diversity that makes
    voting non-degenerate; a single judge is deterministic (temperature 0.0).
    Votes that are absent or out of range are returned as ``None`` and excluded
    from aggregation.
    """
    if not candidates:
        return []
    prompt = build_judge_prompt(instruction, candidates)
    messages = [[TextPrompt(text=prompt)]]

    effective_temperature = sample_temperature if n_judges > 1 else 0.0

    votes: list[Optional[int]] = []
    for _ in range(max(1, n_judges)):
        response, _metadata = client.generate(
            messages=messages,  # type: ignore
            max_tokens=MAX_TOKENS,
            temperature=effective_temperature,
        )
        first = response[0]
        text = first.text if hasattr(first, "text") else str(first)  # pyright: ignore[reportAttributeAccessIssue]
        idx = _extract_vote(text)
        if idx is not None and 0 <= idx < len(candidates):
            votes.append(idx)
        else:
            votes.append(None)
    return votes


def aggregate_votes(votes: list[Optional[int]], n_candidates: int) -> Optional[int]:
    """Plurality voting: the candidate with the most valid votes wins, ties
    broken toward the earliest candidate. Returns ``None`` if there are no valid
    votes.
    """
    counts = [0] * n_candidates
    for v in votes:
        if v is not None and 0 <= v < n_candidates:
            counts[v] += 1
    if sum(counts) == 0:
        return None
    return max(range(n_candidates), key=lambda i: (counts[i], -i))


def consensus_pick(
    instruction: str,
    candidates: list[str],
    votes: list[Optional[int]],
    client: LLMClient,
) -> Optional[int]:
    """Consensus aggregation: a chair agent makes the final selection.

    The chair sees the task, the candidate solutions, and the tally of the
    judges' individual picks, then emits a final ``<vote>``. This is the paper's
    "single answer synthesized from many opinions" shape, contrasted with voting.
    Falls back to the plurality winner if the chair is unparseable, so consensus
    never fails where voting would succeed. Returns ``None`` if no judge cast a
    valid vote to tally.
    """
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    counts = [valid.count(i) for i in range(len(candidates))]
    tally_lines = "\n".join(
        f"- Candidate {i + 1}: {counts[i]} vote(s)"
        for i in range(len(candidates))
        if counts[i] > 0
    )
    candidate_blocks = ""
    for i, diff in enumerate(candidates):
        candidate_blocks += (
            f"\n<candidate_solution index={i + 1}>\n"
            f"{diff}\n</candidate_solution index={i + 1}>\n"
        )
    prompt = f"""\
You are the chair of a panel of expert reviewers. Each reviewer independently
picked the single best candidate solution to the task below; their votes are
summarized in the tally. Your job is to reach a final consensus selection.

<instruction>
{instruction}
</instruction>

<candidate_solutions>
{candidate_blocks}
</candidate_solutions>

<vote_tally>
{tally_lines}
</vote_tally>

Weigh the panel's votes against the task and the candidates themselves, then pick
the single best candidate. Emit its 1-based index and nothing else inside the
tag:

<vote>N</vote>
"""
    messages = [[TextPrompt(text=prompt)]]
    response, _metadata = client.generate(
        messages=messages,  # type: ignore
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    first = response[0]
    text = first.text if hasattr(first, "text") else str(first)  # pyright: ignore[reportAttributeAccessIssue]
    idx = _extract_vote(text)
    if idx is not None and 0 <= idx < len(candidates):
        return idx
    return aggregate_votes(votes, len(candidates))


def select(
    instruction: str,
    candidates: list[str],
    client: Optional[LLMClient] = None,
    protocol: str = "voting",
    n_judges: int = DEFAULT_N_JUDGES,
    sample_temperature: float = DEFAULT_SAMPLE_TEMPERATURE,
) -> Optional[int]:
    """Select the best of N candidates by multi-judge voting or consensus.

    ``n_judges`` independent judges each pick one candidate. Under ``"voting"``
    (the paper's recommended protocol for reasoning tasks) the plurality winner
    is returned. Under ``"consensus"`` a chair agent makes the final call from
    the judges' vote tally. Returns the 0-based index of the selected candidate,
    or ``None`` if there are no candidates. ``client`` defaults to the repo's
    Anthropic direct client (Claude Sonnet 4), matching the ensembler's model
    convention.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return 0
    if protocol not in ("voting", "consensus"):
        raise ValueError(f"Unknown protocol: {protocol!r}")
    if client is None:
        client = get_client("anthropic-direct")

    votes = judge_votes(instruction, candidates, client, n_judges, sample_temperature)
    if protocol == "consensus":
        return consensus_pick(instruction, candidates, votes, client)
    return aggregate_votes(votes, len(candidates))
