"""Verification scoring prompt for per-candidate solution evaluation.

Used by ``utils.solution_verifier`` to elicit fine-grained, criterion-
decomposed continuous scores for a single candidate solution.
"""

DEFAULT_CRITERIA = ("correctness", "completeness", "minimality")


def build_verifier_prompt(
    instruction: str,
    candidate: str,
    criteria: tuple[str, ...] = DEFAULT_CRITERIA,
    max_score: int = 100,
) -> str:
    """Build a prompt that asks the verifier to score ONE candidate.

    The candidate is scored independently (not compared against its siblings)
    and on each criterion separately, producing the fine-grained continuous
    signal that drives ranking.
    """
    criteria_list = "\n".join(f"- {c}" for c in criteria)
    score_tags = "\n".join(f'<score criterion="{c}">N</score>' for c in criteria)
    return f"""\
You are a strict code reviewer verifying a candidate solution to a software task.

<instruction>
{instruction}
</instruction>

<candidate_solution>
{candidate}
</candidate_solution>

Score the candidate on each criterion below from 0 to {max_score}, where higher
is better. Use the full 0-{max_score} range for fine-grained calibration rather
than rounding to coarse buckets.

Criteria:
{criteria_list}

Respond ONLY with one score tag per criterion and nothing else:

{score_tags}
"""
