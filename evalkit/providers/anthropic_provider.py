"""Anthropic provider - the model under test on a real run.

Uses the official `anthropic` SDK and Claude Opus 4.8. Reads ANTHROPIC_API_KEY
from the environment (or an `ant auth login` profile). Import is lazy so the
package installs and the mock-based CI runs without the SDK present.
"""

from __future__ import annotations

import os

from ..models import Case

MODEL = os.environ.get("EVAL_ANTHROPIC_MODEL", "claude-opus-4-8")


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic  # lazy: only needed for a real run

        self._client = anthropic.Anthropic()

    def __call__(self, case: Case) -> str:
        system = case.system or "You are a careful assistant. Answer only from the provided context; if it is not there, say so."
        content = case.prompt
        if case.contexts:
            joined = "\n\n".join(f"[doc {i + 1}]\n{c}" for i, c in enumerate(case.contexts))
            content = f"Context:\n{joined}\n\nQuestion: {case.prompt}"

        message = self._client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        # Guard the refusal stop reason before reading content.
        if message.stop_reason == "refusal":
            return "[refused]"
        return "".join(b.text for b in message.content if b.type == "text").strip()
