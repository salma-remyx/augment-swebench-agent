"""Line-anchored feedback formatting for failed file edits.

Adapted from *Line-Anchored Feedback Cuts Token Costs and Improves
Correctness in AI Code Editing* (arXiv:2607.12713). The paper shows that
delivering requested-change feedback anchored to specific line numbers
(with inline comments) cuts generated tokens and improves correctness for
Claude models, versus delivering the same request as holistic prose.

The repo's edit tool already numbers its *success* output (`cat -n`); this
module brings the same line anchoring to its *failure* path. When an
``old_str`` is not found verbatim, instead of echoing the unanchored
``old_str`` back at the model, we locate the file region that most closely
resembles it and render that region with line numbers plus an inline
marker pointing at the probable intended location.

Mode 2 (adapted port): the paper's FileMark alignment is a learned /
interactive inline-comment export over a live editor. We substitute that
auxiliary with a parameter-free stdlib proxy (``difflib``) that finds the
closest line window — same signal (anchor feedback to the file's own line
numbers, add an inline comment), target-native execution. The core
mechanism — line-anchored feedback on requested changes — is preserved at
full fidelity. Line numbering mirrors the edit tool's own ``cat -n``
convention (split on ``"\\n"``, 1-based) so the anchors agree with what
``view`` / a successful edit already show.
"""

import difflib

CONTEXT_LINES: int = 4
"""Lines of context shown on each side of an anchored region."""

MIN_RATIO: float = 0.3
"""Below this window similarity we report no useful closest match.

Empirically, near-copy lines (the common case: the model's ``old_str`` is a
slightly-off copy of real file lines) score ~0.8+, while unrelated short
lines score <0.25; 0.3 sits in that gap.
"""

MAX_RENDERED_OCCURRENCES: int = 5
"""Cap on duplicate occurrences rendered with full context.

Feedback tokens are the cost the paper optimizes; a pathological ``old_str``
(e.g. a single common token) can occur hundreds of times, so occurrences
beyond this cap are summarized as a line-number list instead of rendered.
"""


def _line_similarity(a: str, b: str) -> float:
    """Rough per-line similarity in [0, 1] (parameter-free difflib proxy)."""
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def find_closest_region(
    file_content: str,
    old_str: str,
    min_ratio: float = MIN_RATIO,
) -> tuple[int, int, float] | None:
    """Find the line window in ``file_content`` most similar to ``old_str``.

    Returns ``(start_line, width, ratio)`` where ``start_line`` is 1-based,
    ``width`` is the window size in lines, and ``ratio`` is the best mean
    per-line similarity; or ``None`` when no window clears ``min_ratio``
    (e.g. empty file, or the request is unrelated to the file contents).
    """
    file_lines = file_content.split("\n")
    old_lines = old_str.split("\n")
    if not old_lines or not old_str.strip() or not file_lines:
        return None

    width = len(old_lines)
    n = len(file_lines)
    if width > n:
        # Compare against the whole file as one window instead of skipping.
        width = n

    best: tuple[int, int, float] | None = None
    best_ratio = min_ratio
    # Bound cost on very large files by striding candidate start positions;
    # the failure path is not hot, but the tool may edit large files.
    stride = 1 if n <= 2000 else max(1, n // 2000)
    for start in range(0, max(1, n - width + 1), stride):
        window = file_lines[start : start + width]
        ratio = sum(_line_similarity(a, b) for a, b in zip(window, old_lines))
        ratio /= width
        if ratio > best_ratio:
            best_ratio = ratio
            best = (start + 1, width, ratio)
    return best


def render_numbered_region(
    file_content: str,
    start_line: int,
    width: int,
    context_lines: int = CONTEXT_LINES,
    marker: str = "<-- closest match to your old_str",
) -> str:
    """Render a file window (1-based ``start_line``, ``width`` lines) ``cat -n``-style.

    Includes ``context_lines`` of surrounding context and an inline marker
    comment anchored to the first line of the window — the line-anchored
    delivery from the paper.
    """
    file_lines = file_content.split("\n")
    n = len(file_lines)
    lo = max(0, start_line - 1 - context_lines)
    hi = min(n, start_line - 1 + width + context_lines)
    out: list[str] = []
    for idx in range(lo, hi):
        out.append(f"{idx + 1:6}\t{file_lines[idx]}")
        if idx == start_line - 1:
            out.append(f"{'':6}\t# {marker}")
    return "\n".join(out)


def build_no_match_feedback(
    file_content: str,
    old_str: str,
    context_lines: int = CONTEXT_LINES,
) -> str:
    """Build line-anchored feedback for an ``old_str`` not found verbatim.

    Returns anchored feedback text (possibly empty) for appending to an
    edit-tool error message. Preserves a ``str -> str`` contract: callers
    pass the file content and the unmatched ``old_str`` and get back text.
    """
    match = find_closest_region(file_content, old_str)
    if match is None:
        return ""
    start_line, width, _ratio = match
    rendered = render_numbered_region(
        file_content, start_line, width, context_lines=context_lines
    )
    return (
        "Line-anchored feedback: the requested `old_str` was not found "
        "verbatim. The file region that most closely resembles it is "
        f"anchored at line {start_line}:\n{rendered}\n"
        "Re-issue `str_replace` with `old_str` copied from the exact "
        "line(s) above."
    )


def build_duplicate_feedback(
    file_content: str,
    line_numbers: list[int],
    context_lines: int = CONTEXT_LINES,
) -> str:
    """Build line-anchored feedback for a non-unique (duplicate) ``old_str``.

    ``line_numbers`` are the 1-based lines where matches were found. Renders
    each occurrence (up to ``MAX_RENDERED_OCCURRENCES``) with surrounding
    context and an inline marker so the model can disambiguate which
    occurrence to target; further occurrences are summarized as a
    line-number list to keep the feedback bounded. Returns "" when empty.
    """
    if not line_numbers:
        return ""
    file_lines = file_content.split("\n")
    n = len(file_lines)
    unique_lines = sorted(set(line_numbers))
    out: list[str] = []
    for i, ln in enumerate(unique_lines[:MAX_RENDERED_OCCURRENCES]):
        if i > 0:
            out.append(f"{'':6}\t# ---")
        lo = max(0, ln - 1 - context_lines)
        hi = min(n, ln + context_lines)
        for idx in range(lo, hi):
            out.append(f"{idx + 1:6}\t{file_lines[idx]}")
            if idx == ln - 1:
                out.append(f"{'':6}\t# <-- duplicate occurrence of old_str")
    remaining = unique_lines[MAX_RENDERED_OCCURRENCES:]
    if remaining:
        out.append(
            f"{'':6}\t# ... and {len(remaining)} more occurrence(s) "
            f"at lines {remaining}"
        )
    body = "\n".join(out)
    return (
        "Line-anchored feedback: the occurrences of `old_str` are anchored "
        f"below with line numbers:\n{body}\n"
        "Include more surrounding lines in `old_str` so it matches exactly "
        "one location."
    )
