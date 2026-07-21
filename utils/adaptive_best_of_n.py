"""Adaptive best-of-N selection by answer agreement.

Adapted (Mode 2) from *Best-of-∞ -- Asymptotic Performance of Test-Time
Compute* (arXiv:2509.21091). The paper studies best-of-N selection by majority
voting and prescribes an *adaptive* generation budget: stop sampling once the
candidate answers agree, rather than spending a fixed N on every problem. That
operational core is what this module ports.

The paper's core mechanism is kept intact -- majority agreement is the
selection signal, and an agreement-based stopping rule picks the sampling
budget N per problem. One auxiliary component is substituted:

  * The paper's "answer" is a single value (e.g. a multiple-choice label or a
    numeric answer) that N samples vote on. In this repo the "answer" is a
    candidate patch, so agreement is measured over *normalized patch content*
    -- the added/removed code with diff hunk headers and line numbers stripped.
    Two rollouts that produce the same net change share a signature, which is
    the parameter-free stand-in for the paper's "same answer".

  * When the persisted ``eval_outcomes`` are available, agreement is measured
    over the *successful* rollouts first (the repo's eval signal stands in for
    the paper's answer-correctness oracle), falling back to all rollouts when
    none succeed. The majority pick is tie-broken toward a successful rollout
    so the eval signal refines the selection.

Intentionally scoped out (does not fit this repo):
  * The paper's "optimal weighted multi-LLM ensemble" combines the majority
    votes of several LLMs, each weighted by its reliability. This repo runs a
    single driver model (Claude Sonnet 4), so there is no multi-LLM ensemble
    to weight -- only the single-model adaptive-N core is ported.
  * The paper's N -> infinity asymptotic analysis is the *justification* for
    why agreement-based stopping is sound; it is not reproduced as code.

``select`` exposes the same ``(instruction, candidates) -> selected index``
contract as ``utils.solution_verifier.select`` and the majority-vote
ensembler, so it drops in as an LLM-free alternative selector. ``adaptive_n``
consumes a problem's ordered rollouts plus their ``eval_outcomes`` to report
the smallest N at which agreement was reached -- the per-problem budget the
paper says is sufficient -- and how many rollouts that saves.
"""

from dataclasses import dataclass
from typing import Optional

DEFAULT_AGREEMENT_THRESHOLD = 0.5
"""Leader must hold strictly more than this share of the votes seen so far for
the stopping rule to fire. ``0.5`` is a strict majority."""

DEFAULT_MIN_SAMPLES = 2
"""Smallest prefix the stopping rule may commit on. Agreement is undefined for
a single sample, so the rule never fires before two rollouts."""


def normalize_patch(diff: str) -> str:
    """Content signature of a candidate patch, ignoring diff bookkeeping.

    Strips markdown code fences, file headers (``diff --git``, ``index``,
    ``---``, ``+++``) and hunk headers (``@@ ... @@`` -- whose line numbers
    vary across rollouts even for equivalent changes), drops the leading
    ``+``/``-``/`` `` marker from each content line, and trims trailing
    whitespace. Two rollouts that produce the same net code change therefore
    share a signature: the parameter-free stand-in for the paper's "same
    answer".
    """
    kept: list[str] = []
    for raw in diff.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.lstrip().startswith("```"):
            continue
        if line.startswith(("@@", "diff ", "index ")):
            continue
        if line.startswith(("--- ", "+++ ")) or line in ("---", "+++"):
            continue
        if line[:1] in ("+", "-", " "):
            line = line[1:].rstrip()
        if line:
            kept.append(line)
    return "\n".join(kept)


def agreement_groups(candidates: list[str]) -> list[list[int]]:
    """Group candidate indices by normalized patch, largest group first.

    Indices within a group keep their original order. Groups are sorted by
    descending size; ties break toward the group whose earliest member
    appeared first. An empty input returns an empty list.
    """
    order: dict[str, list[int]] = {}
    for idx, cand in enumerate(candidates):
        order.setdefault(normalize_patch(cand), []).append(idx)
    groups = list(order.values())
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def select(instruction: str, candidates: list[str]) -> Optional[int]:
    """Select the candidate the majority of rollouts agree on.

    Conforms to the ``(instruction, candidates) -> selected index`` contract
    shared with ``utils.solution_verifier.select`` and the majority-vote
    ensembler, so it is a drop-in, LLM-free alternative selector (``instruction``
    is accepted for contract conformance; agreement selection is content-only).
    The winner is the earliest member of the largest agreement group -- a
    plurality vote on the normalized patch, ties toward the earliest group.
    Returns ``None`` for no candidates and ``0`` for a single candidate.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return 0
    return agreement_groups(candidates)[0][0]


def _successes(diffs: list[str], eval_outcomes: object) -> list[bool]:
    """Per-rollout success flags, aligned to ``diffs`` and padded defensively.

    A non-list ``eval_outcomes`` (e.g. the ensembler's ``{}`` default) yields
    all-False, which makes ``adaptive_n`` measure agreement over every rollout
    -- "no eval signal to anchor on".
    """
    n = len(diffs)
    if not isinstance(eval_outcomes, list):
        return [False] * n
    out: list[bool] = []
    for eo in eval_outcomes[:n]:
        out.append(bool(eo.get("is_success", False)) if isinstance(eo, dict) else False)
    if len(out) < n:
        out += [False] * (n - len(out))
    return out


def _plurality_winner(
    diffs: list[str], indices: list[int], successes: list[bool]
) -> int:
    """Index into ``diffs`` of the plurality answer among ``indices``.

    Group size wins; ties break toward the group containing a successful
    rollout, then toward the earliest-appearing group. Within the winning
    group the earliest successful rollout is returned (else the earliest
    member), so the persisted eval signal refines the majority pick.
    """
    groups: dict[str, list[int]] = {}
    for i in indices:
        groups.setdefault(normalize_patch(diffs[i]), []).append(i)

    best_members: list[int] = []
    best_key: Optional[tuple[int, bool, int]] = None
    for members in groups.values():
        has_success = any(successes[i] for i in members)
        key = (len(members), has_success, -members[0])
        if best_key is None or key > best_key:
            best_key = key
            best_members = members

    successful = [i for i in best_members if successes[i]]
    return (successful or best_members)[0]


def _group_share(diffs: list[str], indices: list[int], member_index: int) -> float:
    """Share of ``indices`` whose normalized patch matches ``member_index``."""
    if not indices:
        return 0.0
    key = normalize_patch(diffs[member_index])
    same = sum(1 for i in indices if normalize_patch(diffs[i]) == key)
    return same / len(indices)


@dataclass
class AdaptiveNDecision:
    """Agreement-based adaptive best-of-N result for one problem.

    Attributes:
        n: rollouts available (``len(diffs)``).
        adaptive_n: smallest prefix length at which the winning answer had
            already secured a strict majority and could be committed to.
            Equals ``n`` when agreement was never reached (no safe early stop).
        rollouts_saved: ``n - adaptive_n`` -- generations the adaptive scheme
            would not have spent on this problem.
        selected_index: 0-based index the adaptive scheme commits to (the
            prefix leader when agreement was reached, else the full-N winner).
        leader_share: the leader's vote share at the committed prefix.
        n_successful: how many rollouts passed eval.
        counted_votes: number of votes agreement was measured over (successful
            rollouts when any succeeded, else all rollouts).
        agreement_reached: the stopping rule fired before ``n``.
        matches_full_n: the adaptive pick equals the full-N plurality pick --
            False when early stopping settled on a different answer than using
            all N rollouts would have.
    """

    n: int
    adaptive_n: int
    rollouts_saved: int
    selected_index: int
    leader_share: float
    n_successful: int
    counted_votes: int
    agreement_reached: bool
    matches_full_n: bool


def adaptive_n(
    diffs: list[str],
    eval_outcomes: Optional[list[dict]] = None,
    agreement_threshold: float = DEFAULT_AGREEMENT_THRESHOLD,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Optional[AdaptiveNDecision]:
    """Decide the per-problem sampling budget from answer agreement.

    Walks the rollouts in order and, for each prefix, measures the share of
    the leading answer (the largest normalized-patch group). The adaptive
    budget is the smallest prefix (of at least ``min_samples`` counted votes)
    at which that leader already holds a strict majority (share strictly
    greater than ``agreement_threshold``) -- the point at which, per the
    paper, further sampling would not change the selection.

    Votes are counted over the *successful* rollouts when any succeeded (the
    repo's persisted eval signal stands in for the paper's correctness
    oracle); over all rollouts otherwise. The committed answer is tie-broken
    toward a successful rollout. Returns ``None`` when there are no rollouts.
    """
    n = len(diffs)
    if n == 0:
        return None

    successes = _successes(diffs, eval_outcomes)
    n_successful = sum(successes)
    if n_successful > 0:
        counted_idx = [i for i in range(n) if successes[i]]
    else:
        counted_idx = list(range(n))
    counted_votes = len(counted_idx)

    full_winner = _plurality_winner(diffs, counted_idx, successes)

    selected_index = full_winner
    leader_share = _group_share(diffs, counted_idx, full_winner)
    adaptive_n_val = n
    reached = False
    for k in range(min_samples, counted_votes + 1):
        prefix = counted_idx[:k]
        leader = _plurality_winner(diffs, prefix, successes)
        share = _group_share(diffs, prefix, leader)
        if share > agreement_threshold:
            # Map the counted-prefix length back to a rollout budget: reaching
            # the k-th counted vote requires generating every rollout up to and
            # including counted_idx[k - 1] (a 0-based rollout index).
            adaptive_n_val = counted_idx[k - 1] + 1
            selected_index = leader
            leader_share = share
            reached = True
            break

    return AdaptiveNDecision(
        n=n,
        adaptive_n=adaptive_n_val,
        rollouts_saved=n - adaptive_n_val,
        selected_index=selected_index,
        leader_share=leader_share,
        n_successful=n_successful,
        counted_votes=counted_votes,
        agreement_reached=reached,
        matches_full_n=(selected_index == full_winner),
    )
