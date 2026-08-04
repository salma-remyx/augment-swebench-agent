"""Tests for the PR-Issue alignment pre-filter and its call-site wiring.

These tests import the (non-new) call-site module ``run_agent_on_swebench_problem``
and the (non-new) ``utils.swebench_eval_utils`` to ground the check in the
existing SWE-bench eval path, then exercise ``utils.pr_issue_alignment`` on
synthetic instances that mirror the fields the instance-loading loop yields
(``problem_statement`` / ``patch`` / ``test_patch`` / ``FAIL_TO_PASS`` /
``PASS_TO_PASS``).
"""

import run_agent_on_swebench_problem as runner
from utils.pr_issue_alignment import (
    AlignmentReport,
    check_pr_issue_alignment,
    is_pr_issue_misaligned,
)
from utils.swebench_eval_utils import get_dataset_name

# A long, realistic issue: plenty of meaningful identifiers so the "stub" and
# "no shared identifiers" signals are about real sparsity, not a short string.
_ISSUE = (
    "The login flow crashes when a user session expires. After login the "
    "session token is not refreshed, so the login page loops. The user "
    "expects login to succeed and the session to persist across requests."
)

_PATCH_LOGIN = """diff --git a/app/auth/login.py b/app/auth/login.py
--- a/app/auth/login.py
+++ b/app/auth/login.py
@@ -1,3 +1,5 @@
 class LoginView:
     def login(self, request):
-        return self.render(request)
+        session = self.refresh_session(request)
+        token = self.issue_token(session)
+        return self.render(request, session)
"""

_TEST_LOGIN = """diff --git a/tests/auth/test_login.py b/tests/auth/test_login.py
+++ b/tests/auth/test_login.py
@@
+    def test_session_refreshed_on_login(self):
+        assert login(request).session.token
"""


def _instance(
    statement: str = _ISSUE,
    patch: str = _PATCH_LOGIN,
    test_patch: str = _TEST_LOGIN,
    fail_to_pass: str = '["tests.auth.test_login.test_session_refreshed_on_login"]',
    pass_to_pass: str = "[]",
    instance_id: str = "repo__proj-1",
) -> dict:
    return {
        "instance_id": instance_id,
        "problem_statement": statement,
        "patch": patch,
        "test_patch": test_patch,
        "FAIL_TO_PASS": fail_to_pass,
        "PASS_TO_PASS": pass_to_pass,
    }


class _SeriesRow:
    """Mimics the pandas.Series row the loop iterates: supports ``.get``."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def get(self, key: str, default: object = None) -> object:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> object:
        return self._data[key]


class TestCallSiteWiring:
    def test_call_site_module_imports_the_checker(self):
        # The instance-loading loop is wired to the checker via the module-level
        # import in run_agent_on_swebench_problem (the call-site edit).
        assert hasattr(runner, "is_pr_issue_misaligned")
        assert runner.is_pr_issue_misaligned is is_pr_issue_misaligned

    def test_filter_flag_is_opt_in_and_defaults_off(self):
        # --filter-misaligned is wired as store_true in main()'s argparse, so it
        # defaults to False (existing runs are unchanged when the flag is unset)
        # and the checker is only consulted when the flag is passed.
        import inspect

        src = inspect.getsource(runner)
        assert '"--filter-misaligned"' in src
        assert "store_true" in src
        assert "is_pr_issue_misaligned(problem)" in src

    def test_check_runs_against_verified_dataset_shape(self):
        # The check consumes exactly the row shape the eval path loads.
        assert get_dataset_name("verified") == "princeton-nlp/SWE-bench_Verified"
        report = check_pr_issue_alignment(_instance())
        assert isinstance(report, AlignmentReport)


class TestAlignedInstances:
    def test_well_aligned_instance_is_not_flagged(self):
        report = check_pr_issue_alignment(_instance())
        assert report.is_misaligned is False
        assert report.pattern is None
        # The issue, patch and tests all share identifiers with each other.
        assert report.signals["issue_patch_overlap"] >= 1
        assert report.signals["issue_test_overlap"] >= 1

    def test_misaligned_bool_matches_report(self):
        assert is_pr_issue_misaligned(_instance()) is False

    def test_accepts_series_like_row_from_the_loop(self):
        # The loop yields a pandas.Series (duck-typed via .get / __getitem__).
        report = check_pr_issue_alignment(_SeriesRow(_instance()))
        assert report.is_misaligned is False

    def test_missing_fields_are_tolerated(self):
        # An instance with only an id + statement must not raise.
        report = check_pr_issue_alignment(
            {"instance_id": "x", "problem_statement": _ISSUE}
        )
        assert report.is_misaligned is False


class TestMisalignedPatterns:
    def test_empty_or_stub_statement(self):
        report = check_pr_issue_alignment(_instance(statement="Fix it."))
        assert report.is_misaligned is True
        assert report.pattern == "empty_or_stub_statement"

    def test_chore_or_version_patch(self):
        chore_patch = (
            "diff --git a/requirements.txt b/requirements.txt\n"
            "--- a/requirements.txt\n"
            "+++ b/requirements.txt\n"
            "@@ -1 +1 @@\n"
            "-requests==2.31.0\n"
            "+requests==2.32.0\n"
        )
        report = check_pr_issue_alignment(_instance(patch=chore_patch))
        assert report.is_misaligned is True
        assert report.pattern == "chore_or_version_patch"

    def test_patch_unrelated_to_issue(self):
        cache_patch = """diff --git a/app/cache/prefetch.py b/app/cache/prefetch.py
--- a/app/cache/prefetch.py
+++ b/app/cache/prefetch.py
@@ -1,3 +1,5 @@
 class PrefetchBuffer:
-    def fetch(self):
-        return self.store.get()
+    def prefetch(self):
+        buffer = self.allocate_buffer()
+        return self.invalidate(buffer)
"""
        # Tests still share a token with the issue, so the *patch* signal is the
        # one that fires.
        report = check_pr_issue_alignment(_instance(patch=cache_patch))
        assert report.is_misaligned is True
        assert report.pattern == "patch_unrelated_to_issue"
        assert report.signals["issue_patch_overlap"] == 0

    def test_test_oracle_unrelated_to_issue(self):
        # Patch still shares tokens with the issue; only the test oracle is
        # about unrelated (timezone/glyph rendering) behaviour.
        unrelated_tests = """diff --git a/tests/render/test_glyph.py b/tests/render/test_glyph.py
+++ b/tests/render/test_glyph.py
@@
+    def test_timezone_glyph_render(self):
+        assert render_glyph(timezone)
"""
        report = check_pr_issue_alignment(
            _instance(
                test_patch=unrelated_tests,
                fail_to_pass='["tests.render.test_glyph.test_timezone_glyph_render"]',
            )
        )
        assert report.is_misaligned is True
        assert report.pattern == "test_oracle_unrelated_to_issue"
        assert report.signals["issue_test_overlap"] == 0

    def test_bool_agrees_for_each_pattern(self):
        cases = [
            (_instance(statement="Fix it."), True),
            (
                _instance(
                    patch="diff --git a/requirements.txt b/requirements.txt\n+++ b/requirements.txt\n+requests==2.32.0\n"
                ),
                True,
            ),
        ]
        for instance, expected in cases:
            assert is_pr_issue_misaligned(instance) is expected
