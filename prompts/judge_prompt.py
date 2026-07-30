"""Independent-judge prompt for multi-judge voting / consensus selection.

Used by ``utils.voting_selector``. Each judge independently evaluates the
candidate solutions to a task and picks the single best one, emitting its 1-based
index in a ``<vote>`` tag. Judges share this identical prompt so that -- per the
paper -- the only thing that differs between the voting and consensus protocols
is how the picks are AGGREGATED, not the judges themselves. That keeps the
protocol comparison controlled, which is the paper's central methodological
point.
"""


def build_judge_prompt(instruction: str, candidates: list[str]) -> str:
    """Build a prompt asking one judge to pick the single best candidate.

    The judge sees the task instruction and every candidate solution, then emits
    the 1-based index of its pick inside a ``<vote>`` tag and nothing else, so
    the pick is cheap to parse for vote aggregation.
    """
    prompt = f"""\
You are an expert software engineer reviewing candidate solutions to a task. Pick
the single best candidate.

<instruction>
{instruction}
</instruction>

There are {len(candidates)} candidate solutions:
"""
    for i, diff in enumerate(candidates):
        prompt += f"""
<candidate_solution index={i + 1}>
{diff}
</candidate_solution index={i + 1}>
"""
    prompt += """
Analyze the candidates and pick the single best one. Emit its 1-based index and
nothing else inside the tag:

<vote>N</vote>
"""
    return prompt
