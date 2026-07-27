"""Tests for the MAgICoRe coarse-to-fine refiner and its ensembler wiring.

These tests import the (non-new) call-site module ``majority_vote_ensembler``
and the (non-new) ``utils.llm_client`` to build an injected fake client, then
exercise the consensus gate and reviewer/solver refine loop end-to-end without
making real API calls. Mirrors ``utils/test_solution_verifier.py``.
"""

import majority_vote_ensembler
from utils.llm_client import LLMClient, TextResult
from utils.solution_refiner import (
    RefineResult,
    candidate_similarity,
    consensus_from_candidates,
    refine,
    refine_or_none,
    review,
    solve,
)


class _FakeRefineClient(LLMClient):
    """Plays the reviewer/solver turns of the refine loop from scripted sequences.

    A *review* call (prompt with no ``<reviewer_feedback>``) pops the next
    ``(needs_refinement, feedback)`` verdict; a *solve* call pops the next
    refined diff (``None`` -> a response with no ``<refined_diff>`` tag).
    """

    def __init__(self, review_verdicts, solve_diffs):
        self.review_verdicts = list(review_verdicts)
        self.solve_diffs = list(solve_diffs)
        self.review_calls = 0
        self.solve_calls = 0

    def generate(self, messages, max_tokens, **kwargs):  # type: ignore[no-untyped-def]
        prompt = messages[0][0].text
        if "<reviewer_feedback>" in prompt:
            self.solve_calls += 1
            diff = self.solve_diffs[
                min(self.solve_calls - 1, len(self.solve_diffs) - 1)
            ]
            if diff is None:
                return [TextResult(text="i forgot the tag")], {}
            return [TextResult(text=f"<refined_diff>{diff}</refined_diff>")], {}
        self.review_calls += 1
        needs, feedback = self.review_verdicts[
            min(self.review_calls - 1, len(self.review_verdicts) - 1)
        ]
        body = f"<needs_refinement>{str(needs).lower()}</needs_refinement>"
        if needs:
            body += f"\n<feedback>{feedback}</feedback>"
        return [TextResult(text=body)], {}


def _make_problem(diffs, eval_success):
    return {
        "id": "test-problem",
        "instruction": "Fix the bug",
        "diffs": diffs,
        "eval_outcomes": [{"is_success": s} for s in eval_success],
    }


# Candidate pools used to drive the consensus gate. Identical change lines ->
# high consensus (do not refine); disjoint change lines -> low consensus (refine).
_HIGH_CONSENSUS = ["@@ hunk @@\n+return x"] * 3
_LOW_CONSENSUS = ["@@ a @@\n+return x", "@@ b @@\n+raise Error", "@@ c @@\n+pass"]


class TestConsensusProxy:
    def test_identical_changes_are_perfectly_similar(self):
        assert candidate_similarity("+return x", "+return x") == 1.0

    def test_similarity_is_in_unit_interval(self):
        s = candidate_similarity("+return x", "+raise Error")
        assert 0.0 <= s < 1.0

    def test_change_lines_ignore_unchanged_context(self):
        # Shared unchanged context must not inflate similarity.
        a = "--- f\n+++ f\n ctx\n+return x"
        b = "--- f\n+++ f\n ctx\n+raise"
        assert candidate_similarity(a, b) < 1.0

    def test_consensus_identical_pool_is_unanimous(self):
        assert consensus_from_candidates(_HIGH_CONSENSUS, 0) == 1.0

    def test_consensus_single_candidate_is_unanimous(self):
        assert consensus_from_candidates(["only"], 0) == 1.0

    def test_consensus_none_index_is_unanimous(self):
        assert consensus_from_candidates(_LOW_CONSENSUS, None) == 1.0

    def test_consensus_disjoint_pool_is_low(self):
        assert consensus_from_candidates(_LOW_CONSENSUS, 0) < 0.5


class TestReviewSolve:
    def test_review_parses_needs_and_feedback(self):
        client = _FakeRefineClient([], [])
        client.review_verdicts = [(True, "add a null check")]
        verdict = review("instruction", "candidate diff", client)
        assert verdict.needs_refinement is True
        assert verdict.feedback == "add a null check"

    def test_review_no_feedback_when_satisfied(self):
        client = _FakeRefineClient([(False, "")], [])
        verdict = review("instruction", "candidate diff", client)
        assert verdict.needs_refinement is False
        assert verdict.feedback == ""

    def test_solve_returns_refined_diff(self):
        client = _FakeRefineClient([], ["@@ refined @@\n+return x or default"])
        assert solve("instruction", "candidate", "feedback", client) == (
            "@@ refined @@\n+return x or default"
        )

    def test_solve_returns_none_without_tag(self):
        client = _FakeRefineClient([], [None])
        assert solve("instruction", "candidate", "feedback", client) is None


class TestRefineGate:
    def test_high_consensus_skips_refinement(self):
        # The over-correction guard: an easy problem is returned untouched and
        # no reviewer/solver turn is ever taken.
        client = _FakeRefineClient([(True, "x")], ["should not be used"])
        result = refine(
            "instruction",
            _HIGH_CONSENSUS,
            0,
            _HIGH_CONSENSUS[0],
            client=client,
            threshold=0.5,
            max_rounds=3,
        )
        assert result.refined is False
        assert result.diff == _HIGH_CONSENSUS[0]
        assert result.rounds == 0
        assert result.stopped_reason == "high_consensus"
        assert client.review_calls == 0
        assert client.solve_calls == 0

    def test_low_consensus_refines_until_reviewer_satisfied(self):
        client = _FakeRefineClient(
            [(True, "add null check"), (False, "")], ["REFINED_V1"]
        )
        result = refine(
            "instruction",
            _LOW_CONSENSUS,
            0,
            _LOW_CONSENSUS[0],
            client=client,
            threshold=0.5,
            max_rounds=3,
        )
        assert result.refined is True
        assert result.diff == "REFINED_V1"
        assert result.rounds == 2
        assert result.stopped_reason == "reviewer_satisfied"
        assert client.solve_calls == 1

    def test_low_consensus_but_reviewer_satisfied_first_round_skips_refine(self):
        # Even on a hard problem, if the reviewer immediately finds nothing to
        # fix, nothing is changed (no over-correction).
        client = _FakeRefineClient([(False, "")], [])
        result = refine(
            "instruction",
            _LOW_CONSENSUS,
            0,
            _LOW_CONSENSUS[0],
            client=client,
            threshold=0.5,
            max_rounds=3,
        )
        assert result.refined is False
        assert result.diff == _LOW_CONSENSUS[0]
        assert result.rounds == 1
        assert result.stopped_reason == "reviewer_satisfied"
        assert client.solve_calls == 0

    def test_max_rounds_exhausted(self):
        client = _FakeRefineClient(
            [(True, "f1"), (True, "f2"), (True, "f3")], ["V1", "V2", "V3"]
        )
        result = refine(
            "instruction",
            _LOW_CONSENSUS,
            0,
            _LOW_CONSENSUS[0],
            client=client,
            threshold=0.5,
            max_rounds=3,
        )
        assert result.refined is True
        assert result.diff == "V3"
        assert result.rounds == 3
        assert result.stopped_reason == "max_rounds"
        assert client.solve_calls == 3

    def test_solver_empty_stops_loop(self):
        client = _FakeRefineClient([(True, "f1")], [None])
        result = refine(
            "instruction",
            _LOW_CONSENSUS,
            0,
            _LOW_CONSENSUS[0],
            client=client,
            threshold=0.5,
            max_rounds=3,
        )
        assert result.refined is False
        assert result.diff == _LOW_CONSENSUS[0]
        assert result.rounds == 1
        assert result.stopped_reason == "solver_empty"

    def test_no_candidate_is_a_noop(self):
        client = _FakeRefineClient([(True, "x")], ["x"])
        result = refine("instruction", _LOW_CONSENSUS, 0, None, client=client)
        assert isinstance(result, RefineResult)
        assert result.refined is False
        assert result.stopped_reason == "no_candidate"

    def test_refine_or_none_returns_diff_only_when_refined(self):
        client = _FakeRefineClient(
            [(True, "add null check"), (False, "")], ["REFINED_V1"]
        )
        assert (
            refine_or_none(
                "instruction",
                _LOW_CONSENSUS,
                0,
                _LOW_CONSENSUS[0],
                client=client,
                threshold=0.5,
            )
            == "REFINED_V1"
        )

    def test_refine_or_none_returns_none_on_high_consensus(self):
        client = _FakeRefineClient([(True, "x")], ["should not be used"])
        assert (
            refine_or_none(
                "instruction", _HIGH_CONSENSUS, 0, _HIGH_CONSENSUS[0], client=client
            )
            is None
        )
        assert client.review_calls == 0


class TestEnsemblerWiring:
    def test_process_problem_refines_low_consensus_candidate(self):
        # Exercises the call-site edit in majority_vote_ensembler: the verifier
        # selector picks a candidate, then the refiner rewrites it when the pool
        # disagrees -- the integrated MAgICoRe coarse-to-fine behavior.
        client = _FakeRefineClient(
            [(True, "handle the None case"), (False, "")],
            ["@@ refined @@\n+return x or default"],
        )
        problem = _make_problem(_LOW_CONSENSUS, eval_success=[True, False, False])

        def selector(instruction, candidates):  # type: ignore[no-untyped-def]
            return 0

        def refiner(instruction, candidates, selected_index, selected_diff):  # type: ignore[no-untyped-def]
            r = refine(
                instruction,
                candidates,
                selected_index,
                selected_diff,
                client=client,
                threshold=0.5,
                max_rounds=3,
            )
            return r.diff if r.refined else None

        result = majority_vote_ensembler.process_problem(
            problem, 0, 1, selector=selector, refiner=refiner
        )
        assert result["selected_diff_index"] == 0
        assert result["was_refined"] is True
        assert result["selected_diff"] == "@@ refined @@\n+return x or default"

    def test_process_problem_leaves_high_consensus_candidate_untouched(self):
        client = _FakeRefineClient([(True, "should not run")], ["should not be used"])
        problem = _make_problem(_HIGH_CONSENSUS, eval_success=[True, True, True])

        def selector(instruction, candidates):  # type: ignore[no-untyped-def]
            return 0

        def refiner(instruction, candidates, selected_index, selected_diff):  # type: ignore[no-untyped-def]
            r = refine(
                instruction,
                candidates,
                selected_index,
                selected_diff,
                client=client,
                threshold=0.5,
            )
            return r.diff if r.refined else None

        result = majority_vote_ensembler.process_problem(
            problem, 0, 1, selector=selector, refiner=refiner
        )
        assert result["was_refined"] is False
        assert result["selected_diff"] == _HIGH_CONSENSUS[0]
        assert client.review_calls == 0

    def test_process_problem_without_refiner_is_unchanged(self):
        # The refiner parameter is optional; the verifier path still works.
        problem = _make_problem(_LOW_CONSENSUS, eval_success=[True, False, False])

        def selector(instruction, candidates):  # type: ignore[no-untyped-def]
            return 1

        result = majority_vote_ensembler.process_problem(
            problem, 0, 1, selector=selector
        )
        assert result["selected_diff_index"] == 1
        assert result["was_refined"] is False
        assert result["selected_diff"] == _LOW_CONSENSUS[1]
