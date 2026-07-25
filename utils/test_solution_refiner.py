"""Tests for the PhoenixRepair-inspired reflection refiner.

These tests import the (non-new) ``utils.llm_client`` (to build an injected
fake client) and read the pipeline's real JSONL format
(``example_ensembler_data.jsonl`` -- the same file ``majority_vote_ensembler.py``
consumes), then exercise the cross-candidate reflection + final-round generation
wiring end-to-end without making real API calls.
"""

import json
from pathlib import Path

from utils.llm_client import LLMClient, TextResult
from utils.solution_refiner import distill_insights, partition_by_eval, refine

_EXAMPLE_DATA = Path(__file__).resolve().parent.parent / "example_ensembler_data.jsonl"


def _load_first_example() -> dict:
    with open(_EXAMPLE_DATA) as f:
        return json.loads(f.readline())


def _make_problem(diffs, eval_success):  # type: ignore[no-untyped-def]
    return {
        "id": "test-problem",
        "instruction": "Fix the bug",
        "diffs": diffs,
        "eval_outcomes": [{"is_success": s} for s in eval_success],
    }


class _FakeRefinerClient(LLMClient):
    """Returns canned reflection / final-generation responses.

    The final-generation prompt is the only one that contains the literal
    ``<refined_diff>`` tag, so the fake branches on that. ``refined=None``
    simulates a model answer with no parseable diff, exercising the
    verified-passing fallback.
    """

    def __init__(
        self,
        insights: str | None = "the passing fixes share X",
        refined: str | None = "@@ a @@\n+pass",
    ):
        self.insights = insights
        self.refined = refined
        self.prompts: list[str] = []

    def generate(self, messages, max_tokens, **kwargs):  # type: ignore[no-untyped-def]
        prompt = messages[0][0].text
        self.prompts.append(prompt)
        if "<refined_diff>" in prompt:
            body = (
                f"<refined_diff>{self.refined}</refined_diff>"
                if self.refined is not None
                else "I could not produce a diff."
            )
        else:
            body = f"<insights>{self.insights}</insights>"
        return [TextResult(text=body)], {}


class TestPartitionByEval:
    def test_splits_passing_and_failing(self):
        diffs = ["d0", "d1", "d2"]
        eval_outcomes = [
            {"is_success": True},
            {"is_success": False},
            {"is_success": True},
        ]
        passing, failing = partition_by_eval(diffs, eval_outcomes)
        assert passing == [0, 2]
        assert failing == [1]

    def test_non_list_eval_treats_all_as_failing(self):
        # majority_vote_ensembler defaults eval_outcomes to {} when absent; an
        # unverified candidate must never be trusted as a passing anchor.
        passing, failing = partition_by_eval(["d0", "d1"], {})
        assert passing == []
        assert failing == [0, 1]

    def test_short_eval_list_defaults_missing_to_failing(self):
        passing, failing = partition_by_eval(["d0", "d1"], [{"is_success": True}])
        assert passing == [0]
        assert failing == [1]


class TestDistillInsights:
    def test_parses_insights_tag(self):
        client = _FakeRefinerClient(insights="shared approach: use a guard")
        diffs = ["@@ a @@\n+pass", "@@ b @@\n+raise"]
        insights = distill_insights(
            "Fix the bug", diffs, [{"is_success": True}, {"is_success": False}], client
        )
        assert insights == "shared approach: use a guard"
        # Single reflection call over the whole candidate history.
        assert len(client.prompts) == 1
        # The reflection prompt surfaces the verified pass/fail label.
        assert "PASSED" in client.prompts[0]
        assert "FAILED" in client.prompts[0]

    def test_falls_back_to_raw_text_without_tag(self):
        client = _FakeRefinerClient()
        client.insights = None  # type: ignore[assignment]

        def generate(messages, max_tokens, **kwargs):  # type: ignore[no-untyped-def]
            return [TextResult(text="freeform reflection with no tag")], {}

        client.generate = generate  # type: ignore[assignment]
        insights = distill_insights(
            "Fix the bug", ["d0"], [{"is_success": True}], client
        )
        # No <insights> tag -> the raw reflection is carried forward, not lost.
        assert insights == "freeform reflection with no tag"


class TestRefine:
    def test_generated_when_model_emits_diff(self):
        client = _FakeRefinerClient(refined="@@ refined @@\n+return x")
        diffs = ["@@ good fix @@\n+pass", "@@ broken fix @@\n+raise"]
        result = refine(
            "Fix the bug",
            diffs,
            [{"is_success": True}, {"is_success": False}],
            client=client,
        )
        assert result["source"] == "generated"
        assert result["refined_diff"] == "@@ refined @@\n+return x"
        assert "passing fixes share" in result["insights"]
        # reflection + final-generation = two LLM calls.
        assert len(client.prompts) == 2

    def test_falls_back_to_first_passing_candidate(self):
        # Model emits no parseable diff -> fall back to the verified-passing
        # candidate so the result is never worse than selection.
        client = _FakeRefinerClient(refined=None)
        diffs = ["@@ good fix @@\n+pass", "@@ broken fix @@\n+raise"]
        result = refine(
            "Fix the bug",
            diffs,
            [{"is_success": True}, {"is_success": False}],
            client=client,
        )
        assert result["source"] == "best_pass_fallback"
        assert result["refined_diff"] == diffs[0]

    def test_all_fail_with_no_generation_returns_none(self):
        client = _FakeRefinerClient(refined=None)
        diffs = ["@@ a @@\n+raise", "@@ b @@\n+raise"]
        result = refine(
            "Fix the bug",
            diffs,
            [{"is_success": False}, {"is_success": False}],
            client=client,
        )
        assert result["source"] == "no_pass_no_generation"
        assert result["refined_diff"] is None

    def test_empty_diffs_short_circuits_without_client_call(self):
        client = _FakeRefinerClient()
        result = refine("Fix the bug", [], [], client=client)
        assert result["source"] == "no_candidate"
        assert result["refined_diff"] is None
        assert client.prompts == []


class TestEnsemblerDataIntegration:
    def test_refine_consumes_pipeline_jsonl(self):
        # Proves the refiner integrates with the existing pipeline: it reads the
        # exact JSONL majority_vote_ensembler consumes (diffs + eval_outcomes)
        # and produces a refined candidate per problem.
        problem = _load_first_example()
        client = _FakeRefinerClient(refined="@@ refined @@\n+pass")
        result = refine(
            problem["instruction"],
            problem["diffs"],
            problem["eval_outcomes"],
            client=client,
        )
        # problem-1 has eval_outcomes [True, False, False]; generation wins.
        assert result["source"] == "generated"
        assert result["refined_diff"] == "@@ refined @@\n+pass"
        passing, _failing = partition_by_eval(
            problem["diffs"], problem["eval_outcomes"]
        )
        # The verified anchor the fallback would use is candidate 0.
        assert passing == [0]

    def test_process_problem_emits_source_and_anchor_signal(self):
        # Exercises the CLI's per-problem wiring with an injected client.
        import reflection_refiner

        client = _FakeRefinerClient(refined="@@ refined @@\n+pass")
        problem = _make_problem(
            ["@@ good fix @@\n+pass", "@@ broken fix @@\n+raise"],
            eval_success=[True, False],
        )
        # Patch get_client so refine() uses the fake client.
        original = reflection_refiner.refine.__globals__["get_client"]
        reflection_refiner.refine.__globals__["get_client"] = lambda *a, **k: client
        try:
            result = reflection_refiner.process_problem(problem, 0, 1)
        finally:
            reflection_refiner.refine.__globals__["get_client"] = original
        assert result["source"] == "generated"
        assert result["refined_diff"] == "@@ refined @@\n+pass"
        assert result["best_anchor_eval_success"] is True
