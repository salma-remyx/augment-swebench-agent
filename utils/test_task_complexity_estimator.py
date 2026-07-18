"""Tests for the E3 task-complexity estimator (utils/task_complexity_estimator.py).

These exercise the parameter-free lexical-cue classifier and the single grep
probe that together produce a TaskOperatingPoint -- the Estimate step that
tools.agent.Agent.run_impl consumes as its first-phase turn budget.
"""

from utils.task_complexity_estimator import (
    DEFAULT_TURNS_BY_TIER,
    ComplexityTier,
    TaskComplexityEstimator,
    TaskOperatingPoint,
)


class TestTaskComplexityEstimator:
    """Tests for the lexical-cue classifier and tier mapping."""

    def setup_method(self):
        self.estimator = TaskComplexityEstimator()

    def test_simple_instruction_is_simple(self):
        op = self.estimator.estimate("Fix the typo in the README title.")
        assert op.tier is ComplexityTier.SIMPLE
        assert op.initial_turns == DEFAULT_TURNS_BY_TIER[ComplexityTier.SIMPLE]
        assert op.score < 0.35

    def test_simple_cue_rename(self):
        op = self.estimator.estimate("Rename the function foo to bar.")
        assert op.tier is ComplexityTier.SIMPLE

    def test_complex_instruction_is_complex(self):
        op = self.estimator.estimate(
            "Refactor the entire codebase to migrate from the old pipeline "
            "architecture to a new end-to-end workflow, including comprehensive "
            "integration tests."
        )
        assert op.tier is ComplexityTier.COMPLEX
        assert op.initial_turns == DEFAULT_TURNS_BY_TIER[ComplexityTier.COMPLEX]
        assert op.score >= 0.65

    def test_moderate_instruction_is_moderate(self):
        op = self.estimator.estimate(
            "Change the parser so that nested quotes are handled correctly "
            "across the three supported input formats."
        )
        assert op.tier is ComplexityTier.MODERATE
        assert op.initial_turns == DEFAULT_TURNS_BY_TIER[ComplexityTier.MODERATE]

    def test_simple_has_fewer_turns_than_complex(self):
        simple = self.estimator.estimate("Fix the typo in the README.")
        complex_ = self.estimator.estimate(
            "Refactor the entire codebase end-to-end with comprehensive tests."
        )
        assert simple.initial_turns < complex_.initial_turns

    def test_score_is_clamped_to_unit_interval(self):
        op = self.estimator.estimate(
            "Refactor rewrite migrate the entire codebase end-to-end into a new "
            "feature with comprehensive exhaustive tests across many files, "
            "redesigning the architecture pipeline and workflow from scratch, "
            "fixing concurrency race conditions and deadlocks and memory leaks."
        )
        assert 0.0 <= op.score <= 1.0
        assert op.score == 1.0  # many complex cues saturate the clamp

    def test_empty_instruction_defaults_to_moderate(self):
        op = self.estimator.estimate("")
        assert op.tier is ComplexityTier.MODERATE
        assert op.initial_turns == DEFAULT_TURNS_BY_TIER[ComplexityTier.MODERATE]

    def test_signals_are_recorded(self):
        op = self.estimator.estimate("Fix the typo in the README.")
        assert any("typo" in sig for sig in op.signals)

    def test_operating_point_is_frozen(self):
        op = self.estimator.estimate("Fix the typo in the README.")
        assert isinstance(op, TaskOperatingPoint)
        # frozen dataclass: signals is a tuple, not a mutable list
        assert isinstance(op.signals, tuple)

    def test_custom_turns_by_tier_override(self):
        custom = {
            ComplexityTier.SIMPLE: 1,
            ComplexityTier.MODERATE: 2,
            ComplexityTier.COMPLEX: 3,
        }
        estimator = TaskComplexityEstimator(turns_by_tier=custom)
        op = estimator.estimate("Fix the typo in the README.")
        assert op.tier is ComplexityTier.SIMPLE
        assert op.initial_turns == 1
        # overriding the instance budget must not mutate the class constant
        assert DEFAULT_TURNS_BY_TIER[ComplexityTier.SIMPLE] == 3


class TestGrepProbe:
    """Tests for the optional single grep probe."""

    def test_wide_surface_area_bumps_complexity(self, tmp_path):
        identifier = "compute_revenue_pretty"
        for i in range(10):
            (tmp_path / f"module_{i}.py").write_text(f"def {identifier}(): return 0\n")
        # Without the probe this is a SIMPLE task (single rename); with a wide
        # probe it should be bumped up because the identifier is everywhere.
        without_probe = TaskComplexityEstimator().estimate(
            f"Rename the {identifier} helper."
        )
        with_probe = TaskComplexityEstimator().estimate(
            f"Rename the {identifier} helper.", workspace_root=tmp_path
        )
        assert "grep-files=10" in with_probe.signals
        assert with_probe.score > without_probe.score

    def test_probe_is_optional(self, tmp_path):
        # Passing no workspace_root must still produce a valid operating point.
        op = TaskComplexityEstimator().estimate("Fix the typo in the README.")
        assert op.tier in (
            ComplexityTier.SIMPLE,
            ComplexityTier.MODERATE,
            ComplexityTier.COMPLEX,
        )
        assert not any(sig.startswith("grep-files") for sig in op.signals)

    def test_probe_returns_none_for_missing_root(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        op = TaskComplexityEstimator().estimate(
            "Fix the typo in compute_revenue_pretty.", workspace_root=missing
        )
        # No probe signal because the workspace does not exist.
        assert not any(sig.startswith("grep-files") for sig in op.signals)
        # ... and the simple cues still classify it as SIMPLE.
        assert op.tier is ComplexityTier.SIMPLE


class TestIdentifierExtraction:
    """Tests for the helper that picks grep-probe targets."""

    def test_extracts_camelcase_and_snake_case(self):
        ids = TaskComplexityEstimator._extract_identifiers(
            'Update UserProfile and rebuild_customer_index from "config.yaml".'
        )
        assert "UserProfile" in ids
        assert "rebuild_customer_index" in ids
        assert "config.yaml" in ids

    def test_stops_common_english_words(self):
        ids = TaskComplexityEstimator._extract_identifiers(
            "Please fix the bug in the file for the project."
        )
        # None of these generic words should survive as grep targets.
        for word in ("Please", "fix", "the", "bug", "file", "project"):
            assert word not in ids

    def test_caps_to_three_identifiers(self):
        ids = TaskComplexityEstimator._extract_identifiers(
            " ".join(f"symbol_{i}_name" for i in range(10))
        )
        assert len(ids) <= 3


def test_no_workspace_means_no_filesystem_probe():
    """With no workspace root the estimator must not touch the filesystem."""
    op = TaskComplexityEstimator().estimate(
        "Fix the typo in the README.", workspace_root=None
    )
    assert isinstance(op.initial_turns, int)
    assert op.initial_turns > 0
    assert not any(sig.startswith("grep-files") for sig in op.signals)
