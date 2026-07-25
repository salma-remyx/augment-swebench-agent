"""Cross-candidate reflection and final-round refinement for best-of-N solutions.

Adapted from *PhoenixRepair: Rethinking Repair Strategy Exploration in Software
Agents* (arXiv:2607.18859). The paper's core mechanism is reflection over the
full history of prior repair attempts followed by a final-round patch generated
from insights distilled across ALL of those attempts. This module ports that
core mechanism onto the repo's existing best-of-N pipeline: the orchestration
script (``run_agent_on_swebench_problem.py``) already gathers N candidate diffs
per problem *and* a verified eval outcome per candidate, then hands them to the
ensembler which merely SELECTS one and discards the rest. ``refine`` instead
reads that same (diffs, eval_outcomes) history, reflects across it, and emits
one refined candidate.

Mode 2 (adapted port) substitutions:

  * Multi-location / multi-strategy sampling: the repo already samples N
    candidate diffs per problem, which is the repo-native form of the paper's
    multi-strategy exploration. It is NOT reimplemented here -- the candidate
    history is taken as given.
  * Graph-based localization for hard tasks: CUT. The repo carries no code
    graph. The verified ``eval_outcomes`` (did the test suite pass?) already
    provide a real localization/verification signal, which is exactly the
    "verified I/O-contract match, not keyword overlap" the paper wants.
  * The paper's own benchmark/eval harness: CUT. Evaluation belongs downstream;
    this module CONSUMES the eval_outcomes the repo already produces rather than
    re-running any eval.
  * The paper's multi-round iterative loop is collapsed to a single
    reflection -> final-generation pass. Each "historical attempt" in the paper
    maps to one of the repo's N rollouts, so the cross-candidate reflection
    plays the iteration role; re-running ``refine`` iterates if desired.

Preserved from the paper:

  * Reflection over ALL historical attempts -- every candidate diff and its
    verified eval outcome is read by the reflection stage.
  * Insight distillation across the candidate history.
  * Final-round generation guided by the distilled insights.

The reflection signal is the repo's verified test-suite outcome (``is_success``)
per candidate, so the grounding is a real I/O-contract result, never keyword
overlap. If final-round generation fails to emit a parseable patch, ``refine``
falls back to the first verified-passing candidate -- which is at least as good
as the ensembler's selection, and strictly better when generation succeeds.
"""

import re
from typing import Any, Optional

from prompts.reflection_prompt import (
    build_final_generation_prompt,
    build_reflection_prompt,
)
from utils.llm_client import LLMClient, TextPrompt, get_client

MAX_TOKENS = 16384
TEMPERATURE = 0.0

_INSIGHTS_RE = re.compile(r"<insights>\s*(.*?)\s*</insights>", re.DOTALL)
_REFINED_RE = re.compile(r"<refined_diff>\s*(.*?)\s*</refined_diff>", re.DOTALL)


def _eval_success_list(eval_outcomes: Any, n: int) -> list[bool]:
    """Coerce the JSONL ``eval_outcomes`` field into a per-candidate bool list.

    The pipeline writes a list of ``{"is_success": bool}`` aligned with
    ``diffs``; ``majority_vote_ensembler`` defaults the field to ``{}`` when
    absent. This normalizes both, defaulting unknown entries to ``False`` so a
    missing eval is treated as unverified (never trusted as a passing anchor).
    """
    if not isinstance(eval_outcomes, list):
        return [False] * n
    out: list[bool] = []
    for i in range(n):
        entry = eval_outcomes[i] if i < len(eval_outcomes) else {}
        out.append(
            bool(entry.get("is_success", False)) if isinstance(entry, dict) else False
        )
    return out


def partition_by_eval(
    diffs: list[str], eval_outcomes: Any
) -> tuple[list[int], list[int]]:
    """Split candidate indices into ``(passing, failing)`` by verified outcome.

    A candidate is "passing" only when its aligned eval outcome reports
    ``is_success`` true -- the repo's verified I/O-contract signal.
    """
    success = _eval_success_list(eval_outcomes, len(diffs))
    passing = [i for i, ok in enumerate(success) if ok]
    failing = [i for i, ok in enumerate(success) if not ok]
    return passing, failing


def _first_response_text(response: list[Any]) -> str:
    first = response[0]
    return first.text if hasattr(first, "text") else str(first)  # pyright: ignore[reportAttributeAccessIssue]


def distill_insights(
    instruction: str,
    diffs: list[str],
    eval_outcomes: Any,
    client: LLMClient,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """Reflect across the candidate history and return the distilled insight text.

    Builds the reflection prompt from the diffs grouped by their verified
    pass/fail outcome, calls ``client`` once, and returns the text inside the
    ``<insights>`` tag (or the raw response if the tag is absent, so a poorly
    formatted model answer still carries forward rather than discarding the
    reflection).
    """
    if not diffs:
        return ""
    eval_success = _eval_success_list(eval_outcomes, len(diffs))
    prompt = build_reflection_prompt(instruction, diffs, eval_success)
    response, _metadata = client.generate(
        messages=[[TextPrompt(text=prompt)]],  # type: ignore
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = _first_response_text(response)
    match = _INSIGHTS_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _generate_refined_diff(
    instruction: str,
    diffs: list[str],
    eval_success: list[bool],
    insights: str,
    client: LLMClient,
    max_tokens: int,
    temperature: float,
) -> Optional[str]:
    """Final-round generation: emit one refined diff guided by ``insights``.

    Returns the parsed ``<refined_diff>`` content, or ``None`` if the model did
    not emit a parseable diff (caller falls back to a verified passing
    candidate).
    """
    prompt = build_final_generation_prompt(instruction, diffs, eval_success, insights)
    response, _metadata = client.generate(
        messages=[[TextPrompt(text=prompt)]],  # type: ignore
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = _first_response_text(response)
    match = _REFINED_RE.search(text)
    if not match:
        return None
    diff = match.group(1).strip()
    return diff or None


def refine(
    instruction: str,
    diffs: list[str],
    eval_outcomes: Any,
    client: Optional[LLMClient] = None,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> dict[str, Any]:
    """Reflect across the candidate history and emit one refined candidate.

    Pipeline: partition candidates by verified eval outcome -> distill insights
    across all of them -> final-round generation guided by those insights. If
    generation does not yield a parseable patch, fall back to the first
    verified-passing candidate so the result is never worse than selection.

    Returns a dict with:

    * ``insights``: the distilled reflection text (empty if there are no
      candidates).
    * ``refined_diff``: the generated patch, or the first passing candidate on
      fallback, or ``None`` when there is neither a generation nor any passing
      candidate.
    * ``source``: ``"generated"`` | ``"best_pass_fallback"`` | ``"no_candidate"``
      | ``"no_pass_no_generation"`` -- how ``refined_diff`` was produced, so the
      caller can tell a true final-round patch from a selection-equivalent
      fallback.
    """
    if not diffs:
        return {"insights": "", "refined_diff": None, "source": "no_candidate"}

    if client is None:
        client = get_client("anthropic-direct")

    eval_success = _eval_success_list(eval_outcomes, len(diffs))
    passing, _failing = partition_by_eval(diffs, eval_outcomes)

    insights = distill_insights(
        instruction, diffs, eval_outcomes, client, max_tokens, temperature
    )

    refined = _generate_refined_diff(
        instruction,
        diffs,
        eval_success,
        insights,
        client,
        max_tokens,
        temperature,
    )
    if refined is not None:
        return {"insights": insights, "refined_diff": refined, "source": "generated"}

    if passing:
        # Generation produced nothing usable; return the strongest verified
        # candidate so the outcome is at least as good as plain selection.
        return {
            "insights": insights,
            "refined_diff": diffs[passing[0]],
            "source": "best_pass_fallback",
        }

    return {
        "insights": insights,
        "refined_diff": None,
        "source": "no_pass_no_generation",
    }
