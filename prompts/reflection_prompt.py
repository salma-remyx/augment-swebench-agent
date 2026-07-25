"""Cross-candidate reflection and final-round refinement prompts.

Used by ``utils.solution_refiner`` to (1) reflect over a best-of-N candidate
set *together with its verified eval outcomes* and distill structured insights,
then (2) generate one refined patch guided by those insights. Each candidate is
shown alongside whether the repo's eval harness verified it (``is_success``),
so the reflection is grounded on a real I/O-contract signal rather than keyword
overlap.
"""

# Hard cap on how many candidates of each kind are shown to the model. The
# repo's best-of-N default is 8; long diffs blow up the context, so we surface a
# bounded window and note that the rest were elided.
MAX_PASSING_SHOWN = 4
MAX_FAILING_SHOWN = 4


def _format_candidate(index: int, diff: str) -> str:
    return f"<candidate index={index}>\n{diff}\n</candidate index={index}>\n"


def build_reflection_prompt(
    instruction: str,
    diffs: list[str],
    eval_success: list[bool],
) -> str:
    """Build a prompt that reflects over the candidate history.

    The candidates are split by their verified eval outcome (passing vs.
    failing) and shown to the model with that label, so reflection is anchored
    on what the test suite actually confirmed. The model returns its distilled
    insight inside an ``<insights>`` tag (parsed by ``utils.solution_refiner``).
    """
    passing = [i for i, ok in enumerate(eval_success) if ok]
    failing = [i for i, ok in enumerate(eval_success) if not ok]

    sections: list[str] = []
    sections.append(
        "I am a software engineer fixing a bug in my codebase. Here is the task:"
    )
    sections.append(f"<instruction>\n{instruction}\n</instruction>")
    sections.append(
        f"I generated {len(diffs)} candidate patches and ran the test suite on "
        "each. The eval outcome below is a VERIFIED result: 'passing' means the "
        "candidate made the failing tests pass (and kept the passing tests "
        "passing); 'failing' means it did not. Reflect across the whole "
        "candidate history and distill what makes a correct fix."
    )

    if passing:
        shown = passing[:MAX_PASSING_SHOWN]
        elided = len(passing) - len(shown)
        block = ["These candidates PASSED the test suite (verified correct):"]
        for i in shown:
            block.append(_format_candidate(i + 1, diffs[i]))
        if elided > 0:
            block.append(f"... {elided} more passing candidate(s) elided.\n")
        sections.append("\n".join(block))

    if failing:
        shown = failing[:MAX_FAILING_SHOWN]
        elided = len(failing) - len(shown)
        block = ["These candidates FAILED the test suite (verified incorrect):"]
        for i in shown:
            block.append(_format_candidate(i + 1, diffs[i]))
        if elided > 0:
            block.append(f"... {elided} more failing candidate(s) elided.\n")
        sections.append("\n".join(block))

    if not passing:
        sections.append(
            "No candidate passed. Identify the shared flaw across the failing "
            "attempts and what a correct fix would need to do differently."
        )
    else:
        sections.append(
            "Distill: which approach the passing candidates share, what the "
            "failing candidates get wrong, and the minimal robust change that "
            "keeps the verified-correct behavior while avoiding the failing "
            "approaches."
        )

    sections.append(
        "Write your distilled insight inside XML tags "
        "<insights>...</insights>. Keep it concise and actionable."
    )
    return "\n\n".join(sections)


def build_final_generation_prompt(
    instruction: str,
    diffs: list[str],
    eval_success: list[bool],
    insights: str,
) -> str:
    """Build a prompt for the final-round patch guided by distilled insights.

    The model is given the reflection's distilled ``insights`` plus the verified
    best anchor (the first passing candidate, when one exists) and asked to emit
    one refined unified diff inside a ``<refined_diff>`` tag.
    """
    passing = [i for i, ok in enumerate(eval_success) if ok]
    anchor: list[str] = []
    if passing:
        anchor_i = passing[0]
        anchor.append(
            "The strongest verified starting point is this PASSING candidate "
            f"(candidate {anchor_i + 1}, test-suite verified):"
        )
        anchor.append(_format_candidate(anchor_i + 1, diffs[anchor_i]))
        anchor.append(
            "Prefer a small change on top of it unless the insights call for a "
            "different approach."
        )
    else:
        anchor.append(
            "No candidate passed, so there is no verified anchor. Generate the "
            "fix from scratch guided by the insights."
        )

    return (
        f"I am a software engineer fixing a bug. Here is the task:\n\n"
        f"<instruction>\n{instruction}\n</instruction>\n\n"
        + "\n".join(anchor)
        + f"\n\nDistilled insight from reflecting across all candidate attempts:\n"
        f"<insights>\n{insights}\n</insights>\n\n"
        "Using that insight, produce ONE refined patch as a unified diff that "
        "solves the task. Emit the diff (and only the diff) inside XML tags "
        "<refined_diff>...</refined_diff>. Do not include any other patch."
    )
