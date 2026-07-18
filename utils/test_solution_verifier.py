"""Tests for the LLM-as-a-Verifier solution selector and its ensembler wiring.

These tests import the (non-new) call-site module ``majority_vote_ensembler``
and the (non-new) ``utils.llm_client`` to build an injected fake client, then
exercise the verifier wiring end-to-end without making real API calls.
"""

import majority_vote_ensembler
from utils.llm_client import LLMClient, TextResult
from utils.solution_verifier import rank_candidates, score_candidate, select


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


class _RecordingClient(LLMClient):
    """Records the temperature of each call and returns scores from a sequence.

    Each ``generate`` call pops the next value from ``score_sequence`` (cycling),
    emitting it on every criterion. This simulates stochastic sampling so the
    repeated-evaluation mean can be checked.
    """

    def __init__(self, score_sequence: list[float]):
        self.score_sequence = score_sequence
        self.temperatures: list[float] = []
        self._call = 0

    def generate(self, messages, max_tokens, **kwargs):  # type: ignore[no-untyped-def]
        self.temperatures.append(float(kwargs["temperature"]))
        score = self.score_sequence[self._call % len(self.score_sequence)]
        self._call += 1
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

    def test_rank_candidates_orders_by_descending_score(self):
        client = _FakeVerifierClient({"best": 90.0, "mid": 50.0, "worst": 10.0})
        diffs = ["@@ mid @@", "@@ best @@", "@@ worst @@"]
        ranking = rank_candidates("instruction", diffs, client)
        # Preserves original indices, sorted best-first: best(1), mid(0), worst(2).
        assert [idx for idx, _ in ranking] == [1, 0, 2]
        assert ranking[0][1] == 90.0

    def test_single_sample_is_deterministic_temperature(self):
        client = _RecordingClient([42.0])
        assert (
            score_candidate("instruction", "candidate", client, num_samples=1) == 42.0
        )
        # A single sample stays deterministic (temperature 0.0).
        assert client.temperatures == [0.0]

    def test_repeated_evaluation_samples_and_averages(self):
        # Three stochastic samples (40, 50, 60) -> mean 50.0, and each sampling
        # call must use a nonzero temperature so repeated evaluation is not a
        # degenerate repeat of one deterministic call.
        client = _RecordingClient([40.0, 50.0, 60.0])
        result = score_candidate("instruction", "candidate", client, num_samples=3)
        assert result == 50.0
        assert len(client.temperatures) == 3
        assert all(t > 0.0 for t in client.temperatures)

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

    def test_process_problem_with_repeated_evaluation_selector(self):
        # Mirrors main()'s --verifier-samples plumbing: the selector forwards
        # num_samples into select(), so the ensembler call-site drives repeated
        # evaluation with sampling.
        client = _RecordingClient([80.0])
        diffs = ["@@ only @@\n+pass"]
        problem = _make_problem(diffs, eval_success=[True])

        def selector(instruction, candidates):  # type: ignore[no-untyped-def]
            return select(instruction, candidates, client=client, num_samples=2)

        result = majority_vote_ensembler.process_problem(
            problem, 0, 1, selector=selector
        )
        assert result["selected_diff_index"] == 0
        # Two sampling calls, both at a nonzero temperature.
        assert len(client.temperatures) == 2
        assert all(t > 0.0 for t in client.temperatures)

    def test_process_problem_default_path_uses_no_selector(self):
        # Without a selector, process_problem still accepts the call (the o1
        # branch is only reached if the client call succeeds; here we assert the
        # selector parameter is optional and the no-diffs guard still works).
        problem = _make_problem([], eval_success=[])
        result = majority_vote_ensembler.process_problem(problem, 0, 1)
        assert result["selected_diff_index"] is None
        assert result["selected_diff"] is None
