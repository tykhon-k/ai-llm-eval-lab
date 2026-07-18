"""Core data types shared across the harness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Case:
    """One evaluation case loaded from a dataset file."""

    id: str
    category: str  # "qa" | "rag" | "redteam"
    prompt: str
    system: str = ""
    contexts: list[str] = field(default_factory=list)
    expect_keywords: list[str] = field(default_factory=list)
    forbid_keywords: list[str] = field(default_factory=list)
    expect_refusal: bool = False

    @staticmethod
    def from_dict(d: dict) -> "Case":
        return Case(
            id=d["id"],
            category=d["category"],
            prompt=d["prompt"],
            system=d.get("system", ""),
            contexts=d.get("contexts", []),
            expect_keywords=d.get("expect_keywords", []),
            forbid_keywords=d.get("forbid_keywords", []),
            expect_refusal=d.get("expect_refusal", False),
        )


def load_cases(path: str | Path) -> list[Case]:
    """Load every *.jsonl file under a directory (or a single file)."""
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    cases: list[Case] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("//"):
                cases.append(Case.from_dict(json.loads(line)))
    return cases


@dataclass
class Score:
    """The output of a single scorer for a single case."""

    scorer: str
    score: float  # 0.0 - 1.0
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case: Case
    answer: str
    scores: list[Score]

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.scores)


@dataclass
class RunResult:
    provider: str
    results: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)

    def pass_rate_for(self, category: str) -> float:
        subset = [r for r in self.results if r.case.category == category]
        if not subset:
            return 1.0  # nothing to fail
        return sum(r.passed for r in subset) / len(subset)
