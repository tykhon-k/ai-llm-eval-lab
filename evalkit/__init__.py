"""evalkit - a small, runnable evaluation harness for LLM features.

Five parts, mirroring a production eval pipeline:
  providers  - adapters for the model under test (Anthropic, OpenAI, or a
               deterministic offline stub used in CI)
  scorers    - graders that turn a model answer into a pass/fail + score
  runner     - runs a dataset through a provider and applies the scorers
  report     - renders results as Markdown
  gate       - turns aggregate scores into a CI pass/fail (exit code)
"""

from .models import Case, Score, CaseResult, RunResult

__all__ = ["Case", "Score", "CaseResult", "RunResult"]
