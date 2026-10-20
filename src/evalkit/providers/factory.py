from __future__ import annotations

from evalkit.config import (
    AnthropicProviderConfig,
    FakeProviderConfig,
    OpenAIProviderConfig,
    ProviderConfig,
)
from evalkit.providers.base import Provider


def build_provider(cfg: ProviderConfig) -> Provider:
    """Construct the provider for a validated config. SDK imports stay lazy."""
    if isinstance(cfg, FakeProviderConfig):
        from evalkit.providers.fake import FakeProvider

        return FakeProvider(cfg.model, cfg.script)
    if isinstance(cfg, AnthropicProviderConfig):
        from evalkit.providers.anthropic import AnthropicProvider

        return AnthropicProvider(cfg)
    if isinstance(cfg, OpenAIProviderConfig):
        from evalkit.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(cfg)
    raise TypeError(f"unknown provider config {cfg!r}")  # pragma: no cover
