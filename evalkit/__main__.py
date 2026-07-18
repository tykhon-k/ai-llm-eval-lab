"""CLI entrypoint.

    python -m evalkit --provider mock --dataset datasets --thresholds configs/thresholds.yaml --report report.md

Exits non-zero when the gate fails, so it drops straight into CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .gate import Thresholds, evaluate
from .models import load_cases
from .providers import get_provider
from .report import to_markdown
from .runner import run as run_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit", description="Run the LLM eval suite.")
    parser.add_argument("--provider", default="mock", help="mock | anthropic | openai")
    parser.add_argument("--dataset", default="datasets", help="dataset dir or .jsonl file")
    parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    parser.add_argument("--report", default="report.md", help="path to write the Markdown report")
    args = parser.parse_args(argv)

    cases = load_cases(args.dataset)
    if not cases:
        print(f"no cases found under {args.dataset}", file=sys.stderr)
        return 2

    provider = get_provider(args.provider)
    result = run_suite(cases, provider)

    Path(args.report).write_text(to_markdown(result), encoding="utf-8")
    print(f"ran {len(cases)} cases on provider={result.provider}; wrote {args.report}\n")

    gate = evaluate(result, Thresholds.load(args.thresholds))
    print("\n".join(gate.lines))
    print("\nGATE: " + ("PASS" if gate.passed else "FAIL"))
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
