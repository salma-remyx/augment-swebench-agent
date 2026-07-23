"""Tests for line_anchored_feedback utility."""

from utils.line_anchored_feedback import (
    build_duplicate_feedback,
    build_no_match_feedback,
    find_closest_region,
    render_numbered_region,
)


def test_find_closest_region_locates_nearest_window():
    file_content = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    old_str = "def foo():\n    return 42"
    match = find_closest_region(file_content, old_str)
    assert match is not None
    start_line, width, ratio = match
    assert start_line == 1  # closest window starts at line 1
    assert width == 2
    assert ratio > 0.5


def test_find_closest_region_returns_none_for_unrelated_text():
    file_content = "alpha\nbeta\ngamma\n"
    match = find_closest_region(file_content, "zzz_unrelated_qqq")
    assert match is None


def test_find_closest_region_empty_inputs():
    assert find_closest_region("", "def foo(): pass") is None
    assert find_closest_region("def foo():\n    pass\n", "") is None


def test_render_numbered_region_uses_cat_n_format_and_marker():
    file_content = "a\nb\nc\nd\ne\n"
    rendered = render_numbered_region(file_content, start_line=3, width=1)
    # cat -n style: width-6 number + tab + line content
    assert "     3\tc" in rendered
    # context lines on either side are present
    assert "     1\ta" in rendered
    assert "     5\te" in rendered
    # inline marker anchored to the matched line
    assert "# <-- closest match to your old_str" in rendered


def test_build_no_match_feedback_returns_anchored_block():
    file_content = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    feedback = build_no_match_feedback(file_content, "def foo():\n    return 42")
    assert "Line-anchored feedback" in feedback
    assert "anchored at line 1" in feedback
    # contains a numbered line
    assert "     1\tdef foo():" in feedback


def test_build_no_match_feedback_empty_when_no_match():
    assert build_no_match_feedback("alpha\nbeta\n", "zzz_unrelated") == ""


def test_build_duplicate_feedback_anchors_each_occurrence():
    file_content = "line1\ntarget\nline3\ntarget\nline5"
    feedback = build_duplicate_feedback(file_content, [2, 4])
    assert "duplicate occurrence" in feedback
    # both occurrences rendered with their line numbers
    assert "     2\ttarget" in feedback
    assert "     4\ttarget" in feedback


def test_build_duplicate_feedback_empty_when_no_lines():
    assert build_duplicate_feedback("a\nb\n", []) == ""


def test_build_duplicate_feedback_caps_rendered_occurrences():
    # 8 occurrences: only the first 5 are rendered with context; the rest
    # are summarized so feedback stays bounded (the paper's token-cost lever).
    file_content = "\n".join(f"line{i}" if i % 3 else "target" for i in range(24))
    line_numbers = [i + 1 for i, line in enumerate(file_content.split("\n")) if line == "target"]
    assert len(line_numbers) == 8
    feedback = build_duplicate_feedback(file_content, line_numbers)
    assert feedback.count("duplicate occurrence") == 5
    assert "3 more occurrence(s) at lines [16, 19, 22]" in feedback
