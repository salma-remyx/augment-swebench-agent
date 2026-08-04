"""PR-Issue alignment checker for SWE-bench-like instances.

Adapted (Mode 2) from PAIChecker (arXiv:2607.28587v1), "Uncovering and
Checking PR-Issue Misalignment in SWE-Bench-Like Benchmarks". PAIChecker
detects PR-Issue misalignment via a three-phase multi-agent pipeline:
(1) specific pattern identification, (2) cross-agent label synthesis, and
(3) code-level LLM validation, producing a binary misaligned/aligned label
plus a named pattern.

This module keeps PAIChecker's *core mechanism* -- pattern-based
identification producing a binary label and a named pattern -- and
substitutes only the auxiliary LLM components with parameter-free
lexical/structural proxies that run over the fields the repo already holds
on each SWE-bench row:

* cross-agent label synthesis -> a single deterministic scorer
* code-level LLM validation   -> identifier-overlap heuristics

The issue is ``problem_statement``; the PR/gold patch is ``patch``; the test
oracle is ``test_patch`` / ``FAIL_TO_PASS`` / ``PASS_TO_PASS``.

The heuristics are deliberately high-precision: they fire only on strong
evidence (e.g. zero shared identifiers between issue and patch) because the
intended use is an opt-in pre-filter that protects leaderboard numbers from
misaligned instances -- a false positive would silently drop a valid task.
PAIChecker's full taxonomy (5 patterns / 11 fine-grained scenarios, recall of
nuanced semantic mismatches) requires the LLM agents this adaptation replaces;
the four patterns here approximate the detectable sub-signals.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

# Shorter than this, a token is too generic to carry signal (articles, "id",
# "err", file extensions like "py").
_MIN_TOKEN_LEN = 4
# A real issue statement is at least a couple of sentences; below this many
# distinct identifiers it is a stub / placeholder.
_MIN_ISSUE_TOKENS = 5
# Need enough identifiers in the patch / tests before "shares none" is
# meaningful rather than just "patch is tiny".
_MIN_PATCH_TOKENS = 4
_MIN_TEST_TOKENS = 3

# Generic English filler + boilerplate code identifiers. Kept narrow so that
# domain words (model, test, data, value, login, ...) stay in the signal.
_STOPWORDS = frozenset(
    """
    the a an and or but if then else for to of in on at by with from into onto
    upon over under as is are was were be been being this that these those it
    its their his her your our we you they he she them us not no nor so too can
    will shall may might must should would could do does did has have had when
    while where which who whom what than there here how why about new add old
    self cls return import class pass raise none true false null none void
    """.split()
)

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_SPLIT_RE = re.compile(r"[_]+|(?=[A-Z])")
_FILE_HEADER_RE = re.compile(r"^\+\+\+ b/(.+?)\s*$", re.MULTILINE)
_DIFFGIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$", re.MULTILINE)
_LOCKFILE_RE = re.compile(
    r"(\.lock$|go\.sum$|package-lock\.json$|yarn\.lock$|pnpm-lock\.yaml$|"
    r"pipfile\.lock$|poetry\.lock$|requirements[\w.\-]*\.txt$|"
    r"constraints[\w.\-]*\.txt$)",
    re.IGNORECASE,
)


@dataclass
class AlignmentReport:
    """Result of checking one SWE-bench instance for PR-Issue misalignment."""

    is_misaligned: bool
    pattern: str | None
    reason: str
    signals: dict[str, float] = field(default_factory=dict)


def is_pr_issue_misaligned(instance: Any) -> bool:
    """Return True if the instance matches a PR-Issue misalignment pattern.

    Thin boolean wrapper over :func:`check_pr_issue_alignment`, intended as the
    call-site hook for the instance-loading loop.
    """
    return check_pr_issue_alignment(instance).is_misaligned


def check_pr_issue_alignment(instance: Any) -> AlignmentReport:
    """Check a SWE-bench-like instance for PR-Issue misalignment.

    ``instance`` may be a mapping or any object exposing ``.get(key, default)``
    (e.g. a ``pandas.Series`` as produced by the instance-loading loop). Missing
    fields are treated as empty.

    Patterns are evaluated in priority order; the first strong match wins.
    """
    statement = _field(instance, "problem_statement", "")
    patch = _field(instance, "patch", "")
    test_patch = _field(instance, "test_patch", "")
    fail_to_pass = _test_names(_field(instance, "FAIL_TO_PASS", ""))
    pass_to_pass = _test_names(_field(instance, "PASS_TO_PASS", ""))

    issue_idents = _identifiers(statement)
    patch_idents = _identifiers(_added_text(patch)) | _identifiers(
        " ".join(_changed_file_paths(patch))
    )
    test_idents = _identifiers(_added_text(test_patch)) | _identifiers(
        " ".join(fail_to_pass + pass_to_pass)
    )

    signals = {
        "issue_token_count": float(len(issue_idents)),
        "patch_token_count": float(len(patch_idents)),
        "test_token_count": float(len(test_idents)),
        "issue_patch_overlap": float(len(issue_idents & patch_idents)),
        "issue_test_overlap": float(len(issue_idents & test_idents)),
    }

    for detector in _DETECTORS:
        hit = detector(statement, patch, issue_idents, patch_idents, test_idents)
        if hit is not None:
            pattern, reason = hit
            return AlignmentReport(True, pattern, reason, signals)

    return AlignmentReport(False, None, "no misalignment pattern matched", signals)


# --- detectors ---------------------------------------------------------------
# Each returns (pattern, reason) when a strong misalignment signal fires, else
# None. Evaluated in order; the first non-None result wins.


def _empty_statement(
    statement: str,
    patch: str,
    issue_idents: set[str],
    patch_idents: set[str],
    test_idents: set[str],
) -> tuple[str, str] | None:
    if len(issue_idents) < _MIN_ISSUE_TOKENS:
        return (
            "empty_or_stub_statement",
            f"problem_statement yields only {len(issue_idents)} meaningful "
            f"identifier(s); too sparse to specify a real issue",
        )
    return None


def _chore_patch(
    statement: str,
    patch: str,
    issue_idents: set[str],
    patch_idents: set[str],
    test_idents: set[str],
) -> tuple[str, str] | None:
    paths = _changed_file_paths(patch)
    if paths and all(_is_lockfile_path(p) for p in paths):
        return (
            "chore_or_version_patch",
            "patch only touches lockfile / dependency files -- a maintenance "
            "chore rather than an issue fix",
        )
    return None


def _patch_unrelated_to_issue(
    statement: str,
    patch: str,
    issue_idents: set[str],
    patch_idents: set[str],
    test_idents: set[str],
) -> tuple[str, str] | None:
    if len(issue_idents) < _MIN_ISSUE_TOKENS or len(patch_idents) < _MIN_PATCH_TOKENS:
        return None
    if not (issue_idents & patch_idents):
        return (
            "patch_unrelated_to_issue",
            "patch touches no identifier mentioned in the problem statement",
        )
    return None


def _test_oracle_unrelated_to_issue(
    statement: str,
    patch: str,
    issue_idents: set[str],
    patch_idents: set[str],
    test_idents: set[str],
) -> tuple[str, str] | None:
    if len(issue_idents) < _MIN_ISSUE_TOKENS or len(test_idents) < _MIN_TEST_TOKENS:
        return None
    if not (issue_idents & test_idents):
        return (
            "test_oracle_unrelated_to_issue",
            "FAIL_TO_PASS / PASS_TO_PASS tests share no identifier with the "
            "issue -- the oracle tests behaviour other than what was reported",
        )
    return None


_DETECTORS: tuple[Callable[..., tuple[str, str] | None], ...] = (
    _empty_statement,
    _chore_patch,
    _patch_unrelated_to_issue,
    _test_oracle_unrelated_to_issue,
)


# --- field extraction + tokenization -----------------------------------------


def _field(instance: Any, key: str, default: Any) -> Any:
    """Read ``key`` from a mapping or a ``.get``-supporting object (pandas row).

    pandas' missing values arrive as ``float`` NaN -- coerce those to ``default``
    so downstream string handling never sees a NaN.
    """
    if isinstance(instance, Mapping):
        value = instance.get(key, default)
    else:
        value = instance.get(key, default)
    if isinstance(value, float):
        return default
    return value


def _test_names(raw: Any) -> list[str]:
    """Normalise SWE-bench FAIL_TO_PASS / PASS_TO_PASS into a list of names.

    In the HuggingFace dump these are JSON-encoded strings of test node ids;
    they may also arrive as a list. Anything unparseable is treated as one name.
    """
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(name) for name in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return [raw]
        if isinstance(parsed, (list, tuple)):
            return [str(name) for name in parsed]
        return [str(parsed)]
    return [str(raw)]


def _identifiers(text: str) -> set[str]:
    """Lowercased, de-camel/snake-cased identifier tokens with filler removed."""
    tokens: set[str] = set()
    for raw in _IDENT_RE.findall(text or ""):
        for chunk in _CAMEL_SPLIT_RE.split(raw):
            token = chunk.lower()
            if (
                len(token) >= _MIN_TOKEN_LEN
                and token not in _STOPWORDS
                and not token.isdigit()
            ):
                tokens.add(token)
    return tokens


def _added_text(patch: str) -> str:
    """Concatenated added lines of a unified diff (excludes +++ headers)."""
    if not patch:
        return ""
    lines = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    return "\n".join(lines)


def _changed_file_paths(patch: str) -> list[str]:
    """Destination file paths touched by a unified diff."""
    if not patch:
        return []
    paths = _FILE_HEADER_RE.findall(patch)
    if paths:
        return paths
    return [b for _a, b in _DIFFGIT_RE.findall(patch)]


def _is_lockfile_path(path: str) -> bool:
    return bool(_LOCKFILE_RE.search(path))
