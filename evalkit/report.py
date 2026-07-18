"""Report: render a RunResult as Markdown."""

from __future__ import annotations

from .models import RunResult


def to_markdown(run: RunResult) -> str:
    lines: list[str] = []
    lines.append(f"# LLM eval report - provider: `{run.provider}`\n")
    lines.append(f"**Overall pass rate: {run.pass_rate:.0%}** "
                 f"({sum(r.passed for r in run.results)}/{len(run.results)} cases)\n")

    categories = sorted({r.case.category for r in run.results})
    lines.append("| Category | Pass rate |")
    lines.append("| --- | --- |")
    for cat in categories:
        lines.append(f"| {cat} | {run.pass_rate_for(cat):.0%} |")
    lines.append("")

    lines.append("## Cases\n")
    lines.append("| Case | Category | Result | Scorers |")
    lines.append("| --- | --- | --- | --- |")
    for r in run.results:
        mark = "PASS" if r.passed else "**FAIL**"
        detail = "; ".join(
            f"{s.scorer} {s.score:.0%} {'ok' if s.passed else 'FAIL'}" for s in r.scores
        )
        lines.append(f"| `{r.case.id}` | {r.case.category} | {mark} | {detail} |")
    lines.append("")

    failures = [r for r in run.results if not r.passed]
    if failures:
        lines.append("## Failures\n")
        for r in failures:
            lines.append(f"### `{r.case.id}`")
            lines.append(f"- prompt: {r.case.prompt}")
            lines.append(f"- answer: {r.answer}")
            for s in r.scores:
                if not s.passed:
                    lines.append(f"- {s.scorer}: {s.detail}")
            lines.append("")

    return "\n".join(lines)
