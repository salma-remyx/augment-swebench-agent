"""Probabilistic Pivot Tournament (PPT): O(Nk) best-of-N selection.

Ported from the reference LLM-as-a-Verifier implementation. A round-robin
tournament compares all C(N, 2) pairs of candidates -- O(N^2) verifier calls.
PPT reaches the same selection with O(Nk) comparisons (k = number of pivots,
k << N) in three steps:

  1) Ring pass. Sample a uniformly random Hamiltonian cycle over the N
     candidates and score the N adjacent directed pairs. Because the cycle is a
     single loop, every candidate appears exactly once in the "A" slot and once
     in the "B" slot of the verifier prompt, so any systematic slot preference
     of the verifier cancels in expectation across the ring.

  2) Pivot selection. Rank candidates by their ring-pass mean preference
     w_i / c_i and take the top-k as the pivot set P.

  3) Pivot rounds. With P fixed, score every non-pivot-vs-pivot directed pair
     and every pivot-vs-pivot pair. All comparisons are aggregated into the same
     w_i, c_i and the winner is argmax_i w_i / c_i. Normalizing by c_i removes
     the bias that pivots take part in more comparisons than non-pivots.

  Total comparisons: N + k(N - k) + C(k, 2), i.e. linear in N for fixed k.

A comparison's two fine-grained rewards (R_a, R_b) become a soft win via the
Bradley-Terry model, p(a beats b) = sigmoid(R_a - R_b). This module is
scorer-agnostic: the caller supplies a directed ``score(a, b) -> (R_a, R_b)``
that scores candidate ``a`` in slot A and ``b`` in slot B.
"""

import math
from itertools import combinations

DEFAULT_PIVOTS = 2


def ring_cycle(n, rng):
    """Return the N directed adjacent pairs of a uniformly random Hamiltonian
    cycle over ``n`` candidates. For n <= 1 there are no comparisons."""
    if n <= 1:
        return []
    perm = list(range(n))
    rng.shuffle(perm)
    return [(perm[t], perm[(t + 1) % n]) for t in range(n)]


def bradley_terry(ra, rb):
    """p(a beats b) under the Bradley-Terry model on rewards in [0, 1]."""
    return 1.0 / (1.0 + math.exp(-(ra - rb)))


def accumulate(pairs, score, w, c):
    """Score each directed pair and aggregate soft wins into w, c in place."""
    for a, b in pairs:
        ra, rb = score(a, b)
        p = bradley_terry(ra, rb)
        w[a] += p
        c[a] += 1
        w[b] += 1.0 - p
        c[b] += 1


def select_pivots(w, c, k):
    """Top-k candidates by mean preference w_i / c_i (ties broken by index)."""
    n = len(w)
    k = min(k, n)
    order = sorted(
        range(n),
        key=lambda i: (-(w[i] / c[i] if c[i] else 0.0), i))
    return order[:k]


def pivot_round_pairs(n, pivots):
    """Directed pairs for step 3: every non-pivot vs pivot, plus pivot vs
    pivot. Non-pivots take slot A; within P the lower index takes slot A."""
    pivot_set = set(pivots)
    non_pivots = [i for i in range(n) if i not in pivot_set]
    pairs = [(i, p) for i in non_pivots for p in pivots]
    pairs += list(combinations(sorted(pivots), 2))
    return pairs


def select_best(n, ring, k, score):
    """Run the full PPT given a pre-sampled ``ring`` (list of directed pairs)
    and a directed ``score(a, b) -> (R_a, R_b)``. Returns
    ``(best_index, n_comparisons)``. Use ``ring_cycle(n, rng)`` to produce
    ``ring``."""
    w = [0.0] * n
    c = [0] * n

    # Step 1: ring pass.
    accumulate(ring, score, w, c)

    # Step 2: pivots = empirical leaders from the ring pass.
    pivots = select_pivots(w, c, k)

    # Step 3: pivot rounds, aggregated into the same w, c.
    pr_pairs = pivot_round_pairs(n, pivots)
    accumulate(pr_pairs, score, w, c)

    best = max(range(n), key=lambda i: (w[i] / c[i] if c[i] else 0.0, -i))
    return best, len(ring) + len(pr_pairs)
