"""Lightweight intent classification utilities."""

from __future__ import annotations

import re
from typing import Dict, Tuple

FAQ_KEYWORDS = ("jam", "harga", "layanan", "informasi", "alamat", "promo")
ORDER_STATUS_KEYWORDS = ("status", "nomor pesanan", "order", "resi", "pengiriman")


def classify(text: str) -> Dict[str, float | str]:
    """Classify the user text into a minimal set of intents."""
    normalized = text.lower()

    if any(keyword in normalized for keyword in ORDER_STATUS_KEYWORDS):
        return {"intent": "order_status", "confidence": 0.85}

    if any(keyword in normalized for keyword in FAQ_KEYWORDS):
        return {"intent": "faq", "confidence": 0.7}

    return {"intent": "out_of_scope", "confidence": 0.3}


def check_order_status(_: str | None = None) -> str:
    """Placeholder order status handler."""
    return (
        "Fitur pengecekan status pesanan akan segera hadir. "
        "Tim kami sudah menerima permintaan Anda."
    )

