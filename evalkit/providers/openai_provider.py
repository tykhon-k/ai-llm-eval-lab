"""OpenAI provider - lets the same suite run against a second model family.

Reads OPENAI_API_KEY from the environment. Import is lazy so the mock-based CI
runs without the SDK installed.
"""

from __future__ import annotations

import os

from ..models import Case

MODEL = os.environ.get("EVAL_OPENAI_MODEL", "gpt-4o")


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        import openai  # lazy

        self._client = openai.OpenAI()

    def __call__(self, case: Case) -> str:
        system = case.system or "You are a careful assistant. Answer only from the provided context; if it is not there, say so."
        content = case.prompt
        if case.contexts:
            joined = "\n\n".join(f"[doc {i + 1}]\n{c}" for i, c in enumerate(case.contexts))
            content = f"Context:\n{joined}\n\nQuestion: {case.prompt}"

        resp = self._client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
