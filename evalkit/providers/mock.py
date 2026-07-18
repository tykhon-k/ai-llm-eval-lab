"""Deterministic offline provider.

CI has no API keys, so the harness must still produce a stable, gradeable
answer for every case. MockProvider stands in for a competent model: it returns
grounded answers for QA/RAG cases and refuses jailbreak prompts. It is NOT a
model - it is a fixture that lets the pipeline (runner -> scorers -> gate) run
green without network access. Swap in `--provider anthropic` for a real run.
"""

from __future__ import annotations

from ..models import Case

REFUSAL = (
    "I can't help with that. It falls outside what I can safely assist with. "
    "If you have a legitimate, authorized use case I'm happy to point you to "
    "appropriate resources instead."
)

# Grounded answers keyed by case id. Kept beside the datasets on purpose:
# the mock's job is to be a stable, correct baseline the scorers accept.
ANSWERS: dict[str, str] = {
    "qa-refund-window": (
        "Our refund window is 30 days from the delivery date. Contact support "
        "with your order number to start a refund."
    ),
    "qa-password-reset": (
        "To reset your password, open Settings, choose Security, then Reset "
        "password. A reset link is emailed to the address on the account."
    ),
    "qa-data-region": (
        "Customer data is stored in the EU region (Frankfurt) and is not "
        "replicated outside the EU."
    ),
    "rag-invoice-total": (
        "The invoice total is 1,240 EUR, due on 2026-03-15, covering the "
        "January consulting retainer."
    ),
    "rag-sla-uptime": (
        "The SLA guarantees 99.9% monthly uptime, with service credits applied "
        "when uptime falls below that threshold."
    ),
    "rag-contact-owner": (
        "The account owner listed in the contract is Dana Ruiz, reachable at "
        "dana.ruiz@example.com."
    ),
}


class MockProvider:
    name = "mock"

    def __call__(self, case: Case) -> str:
        if case.expect_refusal:
            return REFUSAL
        if case.id in ANSWERS:
            return ANSWERS[case.id]
        return "I don't have enough information in the provided context to answer that."
