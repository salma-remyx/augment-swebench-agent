"""Pairwise, single-criterion verification prompt.

Used by ``utils.solution_verifier`` to elicit a fine-grained reward for a
*directed pair* of candidate trajectories on ONE evaluation criterion. Scoring
each criterion in its own isolated verifier call (rather than bundling every
criterion into one prompt) is the paper's criteria-decomposition axis: it keeps
each verification focused on a single, low-complexity judgement.
"""

# Rich per-criterion descriptions supplied to the verifier. The reference ships
# no built-in criteria (each benchmark supplies its own); these three generic
# code-review criteria make the CLI selector zero-config, and each carries a
# description so the isolated per-criterion call has real guidance to score on.
DEFAULT_CRITERIA = (
    {
        "id": "correctness",
        "name": "correctness",
        "description": (
            "Does the candidate actually solve the task described in the "
            "instruction? Judge whether the change modifies the code path "
            "responsible for the requested behavior and produces the right "
            "result, with no logic errors, wrong APIs, or broken control flow."
        ),
    },
    {
        "id": "completeness",
        "name": "completeness",
        "description": (
            "Does the candidate address the whole task rather than only part "
            "of it? Penalize solutions that handle the happy path but miss "
            "edge cases, or that leave parts of the instruction unimplemented."
        ),
    },
    {
        "id": "minimality",
        "name": "minimality",
        "description": (
            "Does the candidate change only what is needed to solve the task, "
            "without unrelated edits, dead code, or silent regressions in code "
            "paths the instruction did not mention? Smaller, focused changes "
            "score higher, but never at the expense of correctness."
        ),
    },
)


def build_verifier_prompt(
    problem: str,
    trace_a: str,
    trace_b: str,
    criterion: dict,
    max_score: int = 100,
) -> str:
    """Build a prompt that scores TWO candidates on ONE criterion.

    The verifier sees candidate ``trace_a`` in slot A and ``trace_b`` in slot B
    (a *directed* comparison) and scores each on the single ``criterion`` only,
    producing the fine-grained pairwise reward that drives the tournament.
    """
    return f"""\
You are an expert evaluator of candidate solutions to a software task. You will
see a task instruction and two candidate solutions. Your job is to evaluate them
on ONE specific criterion: **{criterion['name']}**.

<instruction>
{problem}
</instruction>

<candidate_A>
{trace_a}
</candidate_A>

<candidate_B>
{trace_b}
</candidate_B>

**Evaluation Guideline — {criterion['name']}:**
{criterion['description']}

Score each candidate ONLY on this specific criterion from 0 to {max_score},
where higher is better. Ignore aspects of the candidates not relevant to
"{criterion['name']}". Use the full 0-{max_score} range for fine-grained
calibration rather than rounding to coarse buckets.

Respond ONLY with one score tag per candidate and nothing else:

<score_A>N</score_A>
<score_B>N</score_B>
"""
