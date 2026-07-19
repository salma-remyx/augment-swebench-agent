"""Tests for the LLM-as-a-Verifier solution selector and its ensembler wiring.

These tests import the (non-new) call-site module ``majority_vote_ensembler``
and the (non-new) ``utils.llm_client`` to build an injected fake client, then
exercise the pairwise verifier + pivot-tournament wiring end-to-end without
making real API calls.
"""

import random
import re

import majority_vote_ensembler
from prompts.verifier_prompt import DEFAULT_CRITERIA
from utils import pivot_tournament as ppt
from utils.llm_client import LLMClient, TextResult
from utils.solution_verifier import (
    directed_reward,
    score_pair_criterion,
    select,
)

_A_RE = re.compile(r"<candidate_A>(.*?)</candidate_A>", re.DOTALL)
_B_RE = re.compile(r"<candidate_B>(.*?)</candidate_B>", re.DOTALL)


class _FakePairwiseClient(LLMClient):
    """Scores each slot from the candidate content in the pairwise prompt.

    The prompt shows candidate A and candidate B; each slot is scored by the
    first marker whose text appears in that slot (default 1.0), and the response
    carries a ``<score_A>``/``<score_B>`` tag pair.
    """

    def __init__(self, scores_by_marker: dict[str, float], default: float = 1.0):
        self.scores_by_marker = scores_by_marker
        self.default = default
        self.calls = 0

    def _score_for(self, text: str) -> float:
        for marker, value in self.scores_by_marker.items():
            if marker in text:
                return value
        return self.default

    def generate(self, messages, max_tokens, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        prompt = messages[0][0].text
        slot_a = _A_RE.search(prompt).group(1)  # type: ignore[union-attr]
        slot_b = _B_RE.search(prompt).group(1)  # type: ignore[union-attr]
        sa = self._score_for(slot_a)
        sb = self._score_for(slot_b)
        body = f"<score_A>{sa}</score_A>\n<score_B>{sb}</score_B>"
        return [TextResult(text=body)], {}


class _RecordingClient(LLMClient):
    """Records the temperature of each call and returns scores from a sequence.

    Each ``generate`` call pops the next value from ``score_sequence`` (cycling)
    and emits it on both slots. This simulates stochastic sampling so the
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
        body = f"<score_A>{score}</score_A>\n<score_B>{score}</score_B>"
        return [TextResult(text=body)], {}


def _make_problem(diffs: list[str], eval_success: list[bool]) -> dict:
    return {
        "id": "test-problem",
        "instruction": "Fix the bug",
        "diffs": diffs,
        "eval_outcomes": [{"is_success": s} for s in eval_success],
    }


class TestPivotTournament:
    def test_bradley_terry_is_symmetric_at_a_tie(self):
        assert ppt.bradley_terry(0.5, 0.5) == 0.5

    def test_ring_cycle_visits_each_slot_once(self):
        ring = ppt.ring_cycle(4, random.Random(0))
        # A Hamiltonian cycle: N directed pairs, each candidate once in slot A
        # and once in slot B.
        assert len(ring) == 4
        assert sorted(a for a, _ in ring) == [0, 1, 2, 3]
        assert sorted(b for _, b in ring) == [0, 1, 2, 3]

    def test_select_best_picks_highest_reward(self):
        rewards = [0.2, 0.5, 0.9, 0.3]

        def score(a, b):
            return rewards[a], rewards[b]

        ring = ppt.ring_cycle(len(rewards), random.Random(0))
        best, n_comparisons = ppt.select_best(len(rewards), ring, 2, score)
        assert best == 2
        assert n_comparisons > 0


class TestSolutionVerifier:
    def test_score_pair_criterion_returns_normalized_rewards(self):
        client = _FakePairwiseClient({"good": 80.0, "bad": 20.0})
        ra, rb = score_pair_criterion(
            "instruction",
            "good candidate",
            "bad candidate",
            DEFAULT_CRITERIA[0],
            client,
            n_evaluations=1,
        )
        # 0-100 scores normalized to [0, 1].
        assert ra == 0.8
        assert rb == 0.2

    def test_directed_reward_isolates_each_criterion(self):
        client = _FakePairwiseClient({"good": 90.0, "bad": 10.0})
        ra, rb = directed_reward(
            "instruction",
            "good candidate",
            "bad candidate",
            client,
            n_evaluations=1,
        )
        assert ra == 0.9
        assert abs(rb - 0.1) < 1e-9
        # One isolated verifier call per criterion (criteria decomposition).
        assert client.calls == len(DEFAULT_CRITERIA)

    def test_select_picks_best_candidate(self):
        client = _FakePairwiseClient({"good fix": 95.0, "broken fix": 10.0})
        diffs = ["@@ good fix @@\n+return x", "@@ broken fix @@\n+return 1/0"]
        assert select("instruction", diffs, client=client, n_evaluations=1) == 0

    def test_select_ties_toward_earliest_candidate(self):
        client = _FakePairwiseClient({"a": 50.0, "b": 50.0})
        diffs = ["@@ a @@\n+pass", "@@ b @@\n+pass"]
        assert select("instruction", diffs, client=client, n_evaluations=1) == 0

    def test_select_single_candidate_returns_zero(self):
        client = _FakePairwiseClient({})
        # A single candidate needs no comparison at all.
        assert select("instruction", ["only"], client=client) == 0
        assert client.calls == 0

    def test_select_empty_returns_none(self):
        client = _FakePairwiseClient({})
        assert select("instruction", [], client=client) is None

    def test_single_evaluation_is_deterministic_temperature(self):
        client = _RecordingClient([42.0])
        ra, _rb = score_pair_criterion(
            "instruction", "a", "b", DEFAULT_CRITERIA[0], client, n_evaluations=1
        )
        assert ra == 0.42
        # A single evaluation stays deterministic (temperature 0.0).
        assert client.temperatures == [0.0]

    def test_repeated_evaluation_samples_and_averages(self):
        # Three stochastic samples (40, 50, 60) -> mean 50 -> 0.5 normalized, and
        # each sampling call must use a nonzero temperature so repeated
        # evaluation is not a degenerate repeat of one deterministic call.
        client = _RecordingClient([40.0, 50.0, 60.0])
        ra, rb = score_pair_criterion(
            "instruction", "a", "b", DEFAULT_CRITERIA[0], client, n_evaluations=3
        )
        assert ra == 0.5
        assert rb == 0.5
        assert len(client.temperatures) == 3
        assert all(t > 0.0 for t in client.temperatures)

    def test_process_problem_wires_verifier_selector(self):
        # Exercises the call-site edit in majority_vote_ensembler via the real
        # verifier path with an injected (non-new) client.
        client = _FakePairwiseClient({"good fix": 90.0, "broken fix": 20.0})
        diffs = ["@@ good fix @@\n+pass", "@@ broken fix @@\n+raise"]
        problem = _make_problem(diffs, eval_success=[True, False])

        def selector(instruction, candidates):  # type: ignore[no-untyped-def]
            return select(instruction, candidates, client=client, n_evaluations=1)

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
        # n_evaluations into select(), so the ensembler call-site drives repeated
        # evaluation with sampling.
        client = _RecordingClient([80.0])
        diffs = ["@@ a @@\n+pass", "@@ b @@\n+raise"]
        problem = _make_problem(diffs, eval_success=[True, False])

        def selector(instruction, candidates):  # type: ignore[no-untyped-def]
            return select(instruction, candidates, client=client, n_evaluations=2)

        result = majority_vote_ensembler.process_problem(
            problem, 0, 1, selector=selector
        )
        assert result["selected_diff_index"] == 0
        # Every sampling call ran at a nonzero temperature.
        assert len(client.temperatures) > 0
        assert all(t > 0.0 for t in client.temperatures)

    def test_process_problem_default_path_uses_no_selector(self):
        # Without a selector, process_problem still accepts the call (the o1
        # branch is only reached if the client call succeeds; here we assert the
        # selector parameter is optional and the no-diffs guard still works).
        problem = _make_problem([], eval_success=[])
        result = majority_vote_ensembler.process_problem(problem, 0, 1)
        assert result["selected_diff_index"] is None
        assert result["selected_diff"] is None
