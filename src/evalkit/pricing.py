"""Token cost from a user-supplied price table.

evalkit ships no authoritative prices: the table lives in the suite (or a file it
points to) and must be checked against the provider's current pricing page.
Cached prompt tokens use their own price when given, else the input price.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from evalkit.config import Price
from evalkit.providers.base import TokenUsage

_PER_TOKEN = 1_000_000.0


def cost_usd(usage: Iterable[TokenUsage], prices: Mapping[str, Price]) -> float | None:
    """Total cost, or None if any billed model has no price (unknown, not zero)."""
    total = 0.0
    for u in usage:
        price = prices.get(u.model)
        if price is None:
            return None
        cache_read = price.cache_read_per_mtok
        cache_write = price.cache_write_per_mtok
        total += (
            u.input_tokens * price.input_per_mtok
            + u.output_tokens * price.output_per_mtok
            + u.cache_read_input_tokens
            * (price.input_per_mtok if cache_read is None else cache_read)
            + u.cache_creation_input_tokens
            * (price.input_per_mtok if cache_write is None else cache_write)
        ) / _PER_TOKEN
    return total
