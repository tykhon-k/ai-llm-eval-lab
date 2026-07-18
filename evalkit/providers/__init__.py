"""Providers: adapters for the model under test.

`get_provider(name)` returns a callable `provider(prompt, system) -> str`.
The `mock` provider is deterministic and needs no API key, so CI runs it.
The `anthropic` and `openai` providers hit real APIs when a key is present.
"""

from .mock import MockProvider


def get_provider(name: str):
    name = name.lower()
    if name == "mock":
        return MockProvider()
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider()
    raise ValueError(f"unknown provider: {name!r} (use mock | anthropic | openai)")


__all__ = ["get_provider", "MockProvider"]
