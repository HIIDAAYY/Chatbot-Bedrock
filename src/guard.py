"""Safety guardrail utilities for model outputs."""

from __future__ import annotations

from typing import Iterable, List, Tuple

LOW_CONFIDENCE_THRESHOLD = 0.45
DENYLIST_RESPONSE = (
    "Maaf, saya tidak dapat membagikan informasi tersebut. "
    "Tim kami siap membantu secara langsung bila diperlukan."
)
LOW_CONFIDENCE_RESPONSE = (
    "Maaf, saya belum yakin dapat menjawab pertanyaan Anda dengan tepat. "
    "Saya akan meneruskan ini ke tim layanan pelanggan kami."
)


def _contains_denylisted_term(text: str, denylist: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in denylist)


def apply(answer_text: str, confidence: float, denylist: Iterable[str] | None = None) -> dict:
    """
    Apply safety filters to model responses.

    Returns a dictionary containing the final text and whether escalation is required.
    """
    final_text = answer_text.strip()
    escalate = False

    if not final_text:
        escalate = True
        final_text = LOW_CONFIDENCE_RESPONSE
        return {"final_text": final_text, "escalate": escalate}

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        escalate = True
        final_text = LOW_CONFIDENCE_RESPONSE

    if denylist and _contains_denylisted_term(final_text, denylist):
        escalate = True
        final_text = DENYLIST_RESPONSE

    return {"final_text": final_text, "escalate": escalate}

