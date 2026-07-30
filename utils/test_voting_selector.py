"""Tests for the multi-judge voting/consensus selector and its ensembler wiring.

These tests import the (non-new) call-site module ``majority_vote_ensembler``
and the (non-new) ``utils.llm_client`` to build an injected fake client, then
exercise the voting and consensus selectors end-to-end without making real API
calls.
"""

import re

import majority_vote_ensembler
from utils.llm_client import LLMClient, TextResult
from utils.voting_selector import (
    aggregate_votes,
    consensus_pick,
    judge_votes,
    select,
)

_BLOCK_RE = re.compile(
    r"<candidate_solution index=(\d+)>\n(.*?)\n</candidate_solution",
    re.DOTALL,
)


class _VoteClient(LLMClient):
    """A judge that votes by candidate marker; doubles as a consensus chair.

    For judge prompts (containing ``<candidate_solution>`` blocks), votes
    1-based for the first candidate whose block text contains
    ``preferred_marker``. Records the temperature of every call. When the prompt
    contains ``<vote_tally>`` (the consensus chair) and ``chair_vote`` is set,
    emits that fixed vote instead; if ``chair_vote`` is None the chair returns
    unparseable text so the fallback path is exercised.
    """

    def __init__(self, preferred_marker: str, chair_vote: int | None = None):
        self.preferred_marker = preferred_marker
        self.chair_vote = chair_vote
        self.temperatures: list[float] = []
        self.calls = 0

    def generate(self, messages, max_tokens, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.temperatures.append(float(kwargs["temperature"]))
        prompt = messages[0][0].text
        if "<vote_tally>" in prompt:
            if self.chair_vote is not None:
                return [TextResult(text=f"<vote>{self.chair_vote}</vote>")], {}
            return [TextResult(text="I cannot decide")], {}
        for idx_str, body in _BLOCK_RE.findall(prompt):
            if self.preferred_marker in body:
                return [TextResult(text=f"<vote>{idx_str}</vote>")], {}
        return [TextResult(text="no clear best")], {}


class _SequenceVoteClient(LLMClient):
    """Returns a predetermined 0-based vote sequence (cycling) for judge calls."""

    def __init__(self, votes: list[int | None]):
        self.votes = votes
        self.calls = 0

    def generate(self, messages, max_tokens, **kwargs):  # type: ignore[no-untyped-def]
        v = self.votes[self.calls % len(self.votes)]
        self.calls += 1
        if v is None:
            return [TextResult(text="unsure")], {}
        return [TextResult(text=f"<vote>{v + 1}</vote>")], {}


def _make_problem(diffs: list[str], eval_success: list[bool]) -> dict:
    return {
        "id": "test-problem",
        "instruction": "Fix the bug",
        "diffs": diffs,
        "eval_outcomes": [{"is_success": s} for s in eval_success],
    }


class TestAggregateVotes:
    def test_plurality_winner(self):
        assert aggregate_votes([0, 1, 0, 0, 1], 2) == 0

    def test_tie_toward_earliest(self):
        # Two votes each: the earliest candidate wins the tie.
        assert aggregate_votes([0, 1, 0, 1], 2) == 0

    def test_no_valid_votes_returns_none(self):
        assert aggregate_votes([None, None], 2) is None

    def test_out_of_range_votes_are_ignored(self):
        # 5 and None are not valid for 2 candidates; the lone vote for 0 wins.
        assert aggregate_votes([5, None, 0], 2) == 0


class TestJudgeVotes:
    def test_collects_one_vote_per_judge(self):
        client = _VoteClient(preferred_marker="good fix")
        diffs = ["@@ good fix @@\n+pass", "@@ bad @@\n+raise"]
        votes = judge_votes("instruction", diffs, client, n_judges=3)
        assert votes == [0, 0, 0]
        assert client.calls == 3
        # n_judges > 1 sampled at a nonzero temperature.
        assert all(t > 0.0 for t in client.temperatures)

    def test_single_judge_is_deterministic(self):
        client = _VoteClient(preferred_marker="good fix")
        diffs = ["@@ good fix @@\n+pass", "@@ bad @@\n+raise"]
        votes = judge_votes("instruction", diffs, client, n_judges=1)
        assert votes == [0]
        # A single judge stays deterministic (temperature 0.0).
        assert client.temperatures == [0.0]

    def test_unparseable_vote_recorded_as_none(self):
        # No candidate carries the marker, so the judge emits no <vote>.
        client = _VoteClient(preferred_marker="missing")
        votes = judge_votes("instruction", ["@@ a @@\n+x", "@@ b @@\n+y"], client)
        assert votes == [None, None, None, None, None]


class TestSelect:
    def test_voting_picks_majority_candidate(self):
        client = _VoteClient(preferred_marker="good fix")
        diffs = ["@@ good fix @@\n+pass", "@@ broken fix @@\n+raise"]
        assert select("instruction", diffs, client=client, n_judges=3) == 0

    def test_voting_respects_plurality(self):
        # Judges split 2-1 for candidate 1 via the fixed sequence.
        client = _SequenceVoteClient([1, 1, 0])
        diffs = ["@@ a @@\n+x", "@@ b @@\n+y", "@@ c @@\n+z"]
        assert select("instruction", diffs, client=client, n_judges=3) == 1

    def test_consensus_can_differ_from_voting(self):
        # All three judges vote candidate 0 (plurality), but the chair overrides
        # to candidate 1 -- showing consensus aggregation is distinct from
        # voting, the paper's central comparison.
        diffs = ["@@ good fix @@\n+pass", "@@ other @@\n+raise", "@@ t @@\n+z"]
        voting_client = _VoteClient(preferred_marker="good fix")
        assert select("instruction", diffs, client=voting_client, n_judges=3) == 0
        consensus_client = _VoteClient(preferred_marker="good fix", chair_vote=2)
        assert (
            select(
                "instruction",
                diffs,
                client=consensus_client,
                n_judges=3,
                protocol="consensus",
            )
            == 1
        )

    def test_consensus_falls_back_to_plurality_when_chair_unparseable(self):
        # chair_vote=None -> the chair is unparseable -> fall back to plurality.
        client = _VoteClient(preferred_marker="good fix", chair_vote=None)
        diffs = ["@@ good fix @@\n+pass", "@@ bad @@\n+raise"]
        assert (
            select(
                "instruction",
                diffs,
                client=client,
                n_judges=3,
                protocol="consensus",
            )
            == 0
        )

    def test_single_candidate_returns_zero(self):
        client = _VoteClient(preferred_marker="only")
        assert select("instruction", ["@@ only @@\n+pass"], client=client) == 0
        # No judging call is needed for a single candidate.
        assert client.calls == 0

    def test_empty_returns_none(self):
        client = _VoteClient(preferred_marker="x")
        assert select("instruction", [], client=client) is None

    def test_unknown_protocol_raises(self):
        client = _VoteClient(preferred_marker="x")
        try:
            select("instruction", ["a", "b"], client=client, protocol="relay")
        except ValueError:
            return
        raise AssertionError("expected ValueError for unknown protocol")


class TestConsensusPick:
    def test_chair_pick_wins_over_plurality(self):
        client = _VoteClient(preferred_marker="good fix", chair_vote=2)
        diffs = ["@@ good fix @@\n+pass", "@@ other @@\n+raise", "@@ t @@\n+z"]
        # Judges unanimously voted 0; the chair overrides to index 1.
        assert consensus_pick("instruction", diffs, [0, 0, 0], client) == 1

    def test_falls_back_to_plurality_when_chair_unparseable(self):
        client = _VoteClient(preferred_marker="good fix", chair_vote=None)
        diffs = ["@@ good fix @@\n+pass", "@@ bad @@\n+raise"]
        assert consensus_pick("instruction", diffs, [0, 1, 0], client) == 0

    def test_no_valid_votes_returns_none(self):
        client = _VoteClient(preferred_marker="x")
        assert consensus_pick("instruction", ["a", "b"], [None, None], client) is None


class TestEnsemblerWiring:
    def test_process_problem_wires_voting_selector(self):
        # Exercises the call-site path in majority_vote_ensembler via the voting
        # selector with an injected (non-new) client.
        client = _VoteClient(preferred_marker="good fix")
        diffs = ["@@ good fix @@\n+pass", "@@ broken fix @@\n+raise"]
        problem = _make_problem(diffs, eval_success=[True, False])

        def selector(instruction, candidates):  # type: ignore[no-untyped-def]
            return select(instruction, candidates, client=client, n_judges=3)

        result = majority_vote_ensembler.process_problem(
            problem, 0, 1, selector=selector
        )
        assert result["selected_diff_index"] == 0
        assert result["selected_diff"] == diffs[0]
        assert result["is_eval_success"] is True
        # The single-shot majority-vote response is empty on the selector path.
        assert result["response"] == ""
