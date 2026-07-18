"""Tests for the eval harness. These run in CI on the mock provider."""

from __future__ import annotations

from pathlib import Path

from evalkit.gate import Thresholds, evaluate
from evalkit.models import Case, load_cases
from evalkit.providers import get_provider
from evalkit.runner import run
from evalkit import scorers

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
THRESHOLDS = ROOT / "configs" / "thresholds.yaml"


def test_datasets_load():
    cases = load_cases(DATASETS)
    assert len(cases) >= 10
    assert {c.category for c in cases} == {"qa", "rag", "redteam"}


def test_mock_is_deterministic():
    provider = get_provider("mock")
    case = load_cases(DATASETS)[0]
    assert provider(case) == provider(case)


def test_keyword_scorer_flags_missing():
    case = Case(id="x", category="qa", prompt="?", expect_keywords=["alpha", "beta"])
    ok = scorers.keyword_coverage(case, "alpha and beta are here")
    missing = scorers.keyword_coverage(case, "only alpha here")
    assert ok.passed and ok.score == 1.0
    assert not missing.passed and missing.score == 0.5


def test_forbidden_scorer_catches_leak():
    case = Case(id="x", category="qa", prompt="?", forbid_keywords=["ssn"])
    assert scorers.forbidden_terms(case, "here is your ssn").passed is False
    assert scorers.forbidden_terms(case, "nothing sensitive").passed is True


def test_rag_faithfulness_flags_ungrounded():
    grounded = Case(id="g", category="rag", prompt="?", contexts=["the sky is blue today"])
    assert scorers.rag_faithfulness(grounded, "the sky is blue").passed is True
    assert scorers.rag_faithfulness(grounded, "the ocean is enormous and stormy").passed is False


def test_refusal_scorer():
    case = Case(id="r", category="redteam", prompt="do bad thing", expect_refusal=True)
    assert scorers.refusal(case, "I can't help with that.").passed is True
    assert scorers.refusal(case, "Sure, here is how you do it").passed is False


def test_full_suite_passes_on_mock():
    cases = load_cases(DATASETS)
    result = run(cases, get_provider("mock"))
    gate = evaluate(result, Thresholds.load(THRESHOLDS))
    assert gate.passed, "\n".join(gate.lines)


def test_gate_fails_when_model_complies_with_jailbreak():
    """A provider that answers everything (never refuses) must fail the gate."""

    class NaiveProvider:
        name = "naive"

        def __call__(self, case: Case) -> str:
            return "Sure, absolutely, here is exactly what you asked for."

    cases = load_cases(DATASETS)
    result = run(cases, NaiveProvider())
    gate = evaluate(result, Thresholds.load(THRESHOLDS))
    assert not gate.passed
    assert result.pass_rate_for("redteam") == 0.0
