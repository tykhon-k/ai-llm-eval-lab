"""Runner: dataset x provider -> scored results."""

from __future__ import annotations

from .models import Case, CaseResult, RunResult
from .scorers import score_case


def run(cases: list[Case], provider) -> RunResult:
    results: list[CaseResult] = []
    for case in cases:
        answer = provider(case)
        results.append(CaseResult(case=case, answer=answer, scores=score_case(case, answer)))
    provider_name = getattr(provider, "name", provider.__class__.__name__)
    return RunResult(provider=provider_name, results=results)
