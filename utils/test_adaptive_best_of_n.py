"""Tests for the agreement-based adaptive best-of-N selector.

These tests import the (non-new) call-site module ``majority_vote_ensembler``
and wire the new agreement selector into its existing ``process_problem``
``selector`` hook -- the same Selector contract used by
``utils.solution_verifier.select`` -- to prove the integration without editing
the ensembler. They also cross-check the parameter-free agreement selector
against the (non-new) ``utils.pivot_tournament`` winner on an unambiguous
problem, and exercise the adaptive-N budget logic directly.
"""

import random

import majority_vote_ensembler
from utils import adaptive_best_of_n as abn
from utils import pivot_tournament as ppt


def _make_problem(diffs: list[str], eval_success: list[bool]) -> dict:
    return {
        "id": "test-problem",
        "instruction": "Fix the bug",
        "diffs": diffs,
        "eval_outcomes": [{"is_success": s} for s in eval_success],
    }


class TestNormalizePatch:
    def test_strips_hunk_header_and_markers(self):
        # Two patches that differ only in hunk line numbers normalize equal --
        # the varying bookkeeping does not change the net code change.
        a = "@@ -10,3 +10,7 @@\n def square(x):\n+def f():\n+    return 1\n"
        b = "@@ -40,9 +40,13 @@\n def square(x):\n+def f():\n+    return 1\n"
        assert abn.normalize_patch(a) == abn.normalize_patch(b)
        assert "def f()" in abn.normalize_patch(a)
        assert "@@" not in abn.normalize_patch(a)

    def test_strips_code_fence_wrapping(self):
        wrapped = "```diff\n@@ @@\n+good fix\n```"
        assert abn.normalize_patch(wrapped) == "good fix"

    def test_different_changes_differ(self):
        assert abn.normalize_patch("+return x") != abn.normalize_patch("+return y")


class TestSelect:
    def test_empty_returns_none(self):
        assert abn.select("p", []) is None

    def test_single_returns_zero(self):
        assert abn.select("p", ["only"]) == 0

    def test_plurality_picks_earliest_of_majority(self):
        # Indices 0 and 2 agree ("good fix"); 1 disagrees.
        diffs = ["@@ @@\n+good fix", "@@ @@\n+other fix", "@@ @@\n+good fix"]
        assert abn.select("p", diffs) == 0

    def test_ties_break_to_earliest(self):
        assert abn.select("p", ["a", "b", "c"]) == 0


class TestAdaptiveN:
    def test_empty_returns_none(self):
        assert abn.adaptive_n([], []) is None

    def test_single_rollout_cannot_commit(self):
        # Agreement is undefined for one sample: the rule never fires.
        d = abn.adaptive_n(["+x"], [{"is_success": True}])
        assert d is not None
        assert d.adaptive_n == 1
        assert d.rollouts_saved == 0
        assert d.agreement_reached is False

    def test_early_agreement_saves_rollouts(self):
        # 4 rollouts, 3 successful and all "fix A"; the 2nd successful A is
        # reached after generating rollouts 0,1,2, so the budget is 3.
        diffs = ["+A", "+B", "+A", "+A"]
        evals = [{"is_success": s} for s in [True, False, True, True]]
        d = abn.adaptive_n(diffs, evals)
        assert d is not None
        assert d.agreement_reached is True
        assert d.adaptive_n == 3
        assert d.rollouts_saved == 1
        assert d.selected_index == 0
        assert d.matches_full_n is True

    def test_no_agreement_uses_full_budget(self):
        # Every rollout a distinct answer: no majority ever emerges.
        diffs = ["+A", "+B", "+C", "+D"]
        d = abn.adaptive_n(diffs, [{"is_success": True}] * 4)
        assert d is not None
        assert d.agreement_reached is False
        assert d.adaptive_n == 4
        assert d.rollouts_saved == 0

    def test_min_samples_delays_commit(self):
        # Two agreeing successes are seen by rollout 2, but min_samples=3 keeps
        # the rule from firing until a third agreeing vote -- which never
        # comes, so the full budget is spent.
        diffs = ["+A", "+A", "+B", "+C", "+D"]
        evals = [{"is_success": s} for s in [True, True, False, False, False]]
        d = abn.adaptive_n(diffs, evals, min_samples=3)
        assert d is not None
        assert d.agreement_reached is False
        assert d.adaptive_n == 5
        # The default min_samples=2 would have stopped at budget 2.
        early = abn.adaptive_n(diffs, evals)
        assert early is not None
        assert early.adaptive_n == 2

    def test_no_successes_measure_agreement_over_all(self):
        # With no eval signal to anchor on, agreement is measured over every
        # rollout. Two identical failing patches trigger an early stop.
        diffs = ["+A", "+A", "+B", "+B"]
        d = abn.adaptive_n(diffs, [{"is_success": False}] * 4)
        assert d is not None
        assert d.n_successful == 0
        assert d.agreement_reached is True
        assert d.adaptive_n == 2

    def test_early_stop_can_pick_a_different_answer_than_full_n(self):
        # The first two rollouts agree on A (a strict majority of the prefix),
        # so the rule commits to A early -- but the full-N majority is B. The
        # matches_full_n flag surfaces exactly this risk.
        diffs = ["+A", "+A", "+B", "+B", "+B", "+B"]
        d = abn.adaptive_n(diffs, [{"is_success": True}] * 6)
        assert d is not None
        assert d.agreement_reached is True
        assert d.adaptive_n == 2
        assert d.selected_index == 0  # committed to A
        assert d.matches_full_n is False  # full-N winner is B (index 2)


class TestEnsemblerWiring:
    def test_select_drops_into_process_problem_selector_hook(self):
        # The new agreement selector plugs into the existing (non-new)
        # majority_vote_ensembler.process_problem selector hook with no edit.
        diffs = [
            "@@ @@\n+good fix",
            "@@ @@\n+other fix",
            "@@ @@\n+good fix",
        ]
        problem = _make_problem(diffs, [True, False, True])
        result = majority_vote_ensembler.process_problem(
            problem, 0, 1, selector=abn.select
        )
        assert result["selected_diff_index"] == 0
        assert result["selected_diff"] == diffs[0]
        assert result["is_eval_success"] is True
        # The single-shot majority-vote response is empty on the selector path.
        assert result["response"] == ""


class TestCrossCheckWithPivotTournament:
    def test_agreement_selector_matches_tournament_answer(self):
        # On an unambiguous problem (one dominant answer), the parameter-free
        # agreement selector and the pairwise pivot tournament pick the SAME
        # answer, even if they land on different identical-candidate indices.
        diffs = ["+fix A", "+fix A", "+fix A", "+fix B"]

        agreement_pick = abn.select("p", diffs)
        assert agreement_pick is not None

        rewards = [0.9, 0.9, 0.9, 0.1]

        def score(a: int, b: int) -> tuple[float, float]:
            return rewards[a], rewards[b]

        ring = ppt.ring_cycle(len(diffs), random.Random(0))
        tournament_pick, n_comparisons = ppt.select_best(len(diffs), ring, 2, score)
        assert n_comparisons > 0

        # Both land on the dominant "fix A" answer.
        agreement_norm = abn.normalize_patch(diffs[agreement_pick])
        assert agreement_norm == "fix A"
        assert abn.normalize_patch(diffs[tournament_pick]) == agreement_norm
