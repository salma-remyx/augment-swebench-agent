"""Reviewer / solver prompts for coarse-to-fine solution refinement.

Used by ``utils.solution_refiner`` to run one round of the MAgICoRe refine
loop: a *reviewer* inspects the currently-selected candidate diff against the
task instruction and emits structured feedback, then a *solver* produces an
improved diff that acts on that feedback. Splitting the two roles into
separate calls (rather than asking one model to self-correct in a single shot)
is the paper's multi-agent refinement axis -- the reviewer's critique grounds
the solver's edit, and the reviewer's ``needs_refinement`` flag is the early
stop that prevents over-correction of an already-good solution.
"""

# Review angles the reviewer is steered toward. The paper supplies a general
# "find the flaw in this solution" reviewer; these three generic code-review
# foci make the reviewer zero-config while keeping its critique concrete.
DEFAULT_REVIEW_FOCUS = (
    "Does the change actually fix the behavior described in the instruction, "
    "or does it miss the real cause of the problem?",
    "Does it introduce regressions, wrong APIs, or broken control flow in "
    "code paths the instruction did not mention?",
    "Is anything left incomplete or only handling the happy path -- edge "
    "cases, error handling, or other parts of the instruction?",
)


def build_review_prompt(instruction: str, candidate_diff: str) -> str:
    """Prompt the reviewer to critique one candidate diff.

    The reviewer answers two structured questions: whether the candidate still
    needs work (``<needs_refinement>``), and concrete, actionable feedback
    (``<feedback>``). A ``false`` needs_refinement is the early stop signal --
    the candidate is already good and refining further would over-correct.
    """
    focus = "\n".join(f"- {f}" for f in DEFAULT_REVIEW_FOCUS)
    return f"""\
You are an expert code reviewer. A candidate solution to a software task is
shown below. Decide whether it still needs improvement, and if so give precise,
actionable feedback the solver can act on.

<instruction>
{instruction}
</instruction>

<candidate_diff>
{candidate_diff}
</candidate_diff>

Focus your critique on:
{focus}

First decide whether the candidate needs any change at all. Respond ONLY with
these two tags and nothing else:

<needs_refinement>true</needs_refinement>   (or false if it is already correct)
<feedback>
Concrete critique and the exact edit direction the solver should take. If
needs_refinement is false, leave this empty.
</feedback>
"""


def build_solve_prompt(instruction: str, candidate_diff: str, feedback: str) -> str:
    """Prompt the solver to produce an improved diff acting on the feedback.

    The solver returns a single unified diff in ``<refined_diff>`` that
    addresses the reviewer's feedback while staying a focused fix for the
    instruction.
    """
    return f"""\
You are an expert software engineer. Improve the candidate diff below so it
better solves the task, acting on the reviewer's feedback. Keep the change
focused on the instruction.

<instruction>
{instruction}
</instruction>

<candidate_diff>
{candidate_diff}
</candidate_diff>

<reviewer_feedback>
{feedback}
</reviewer_feedback>

Produce the full improved unified diff. Respond ONLY with:

<refined_diff>
...the complete improved unified diff here...
</refined_diff>
"""
