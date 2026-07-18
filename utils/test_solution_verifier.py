"""Tests for the LLM-as-a-Verifier solution selector and its ensembler wiring.

These tests import the (non-new) call-site module ``majority_vote_ensembler``
and the (non-new) ``utils.llm_client`` to build an injected fake client, then
exercise the verifier wiring end-to-end without making real API calls.
"""

import majority_vote_ensembler
from utils.llm_client import LLMClient, TextResult
from utils.solution_verifier import score_candidate, select


class _FakeVerifierClient(LLMClient):
    """Returns canned score tags keyed off the candidate content in the prompt."""

    def __init__(self, scores_by_marker: dict[str, float]):
        self.scores_by_marker = scores_by_marker

    def generate(self, messages, max_tokens, **kwargs):  # type: ignore[no-untyped-def]
        text = messages[0][0].text
        # Default low score; the first matching marker wins.
        score = 1.0
        for marker, value in self.scores_by_marker.items():
            if marker in text:
                score = value
                break
        body = (
            '<score criterion="correctness">SCORE</score>\n'
            '<score criterion="completeness">SCORE</score>\n'
            '<score criterion="minimality">SCORE</score>'
        ).replace("SCORE", str(score))
        return [TextResult(text=body)], {}


def _make_problem(diffs: list[str], eval_success: list[bool]) -> dict:
    return {
        "id": "test-problem",
        "instruction": "Fix the bug",
        "diffs": diffs,
        "eval_outcomes": [{"is_success": s} for s in eval_success],
    }


class TestSolutionVerifier:
    def test_score_candidate_returns_continuous_mean(self):
        client = _FakeVerifierClient({"x": 80.0})
        # 80.0 on each of 3 criteria -> mean 80.0 (continuous, not a discrete pick).
        assert score_candidate("instruction", "candidate x", client) == 80.0

    def test_select_picks_highest_scored_candidate(self):
        client = _FakeVerifierClient({"good fix": 95.0, "broken fix": 10.0})
        diffs = ["@@ good fix @@\n+return x", "@@ broken fix @@\n+return 1/0"]
        assert select("instruction", diffs, client=client) == 0

    def test_select_ties_toward_earliest_candidate(self):
        client = _FakeVerifierClient({"a": 50.0, "b": 50.0})
        diffs = ["@@ a @@\n+pass", "@@ b @@\n+pass"]
        assert select("instruction", diffs, client=client) == 0

    def test_process_problem_wires_verifier_selector(self):
        # Exercises the call-site edit in majority_vote_ensembler via the real
        # verifier path with an injected (non-new) client.
        client = _FakeVerifierClient({"good fix": 90.0, "broken fix": 20.0})
        diffs = ["@@ good fix @@\n+pass", "@@ broken fix @@\n+raise"]
        problem = _make_problem(diffs, eval_success=[True, False])

        def selector(instruction, candidates):  # type: ignore[no-untyped-def]
            return select(instruction, candidates, client=client)

        result = majority_vote_ensembler.process_problem(
            problem, 0, 1, selector=selector
        )
        assert result["selected_diff_index"] == 0
        assert result["selected_diff"] == diffs[0]
        assert result["is_eval_success"] is True
        # The single-shot majority-vote response is empty on the verifier path.
        assert result["response"] == ""

    def test_process_problem_default_path_uses_no_selector(self):
        # Without a selector, process_problem still accepts the call (the o1
        # branch is only reached if the client call succeeds; here we assert the
        # selector parameter is optional and the no-diffs guard still works).
        problem = _make_problem([], eval_success=[])
        result = majority_vote_ensembler.process_problem(problem, 0, 1)
        assert result["selected_diff_index"] is None
        assert result["selected_diff"] is None
