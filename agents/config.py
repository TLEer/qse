"""LLM provider selection for the agent loop.

Two providers, both spoken to through the official Anthropic SDK:

  anthropic  — api.anthropic.com, claude-opus-4-8, native structured outputs
               (client.messages.parse). Requires ANTHROPIC_API_KEY (or an
               `ant auth login` profile).
  deepseek   — DeepSeek's Anthropic-compatible endpoint
               (https://api.deepseek.com/anthropic). Same Messages API wire
               shape, but no server-side structured outputs, so the pipeline
               falls back to JSON-prompted output validated with Pydantic.
               Requires DEEPSEEK_API_KEY.

Select with LLM_PROVIDER=anthropic|deepseek (default anthropic); override
the model with LLM_MODEL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    base_url: str | None
    api_key_env: str
    native_structured_outputs: bool


_PROVIDERS = {
    "anthropic": ProviderConfig(
        name="anthropic",
        model="claude-opus-4-8",
        base_url=None,
        api_key_env="ANTHROPIC_API_KEY",
        native_structured_outputs=True,
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/anthropic",
        api_key_env="DEEPSEEK_API_KEY",
        native_structured_outputs=False,
    ),
}


def load_provider() -> ProviderConfig:
    name = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if name not in _PROVIDERS:
        raise SystemExit(f"LLM_PROVIDER must be one of {sorted(_PROVIDERS)}, got '{name}'")
    cfg = _PROVIDERS[name]
    if model := os.environ.get("LLM_MODEL"):
        cfg = ProviderConfig(cfg.name, model, cfg.base_url, cfg.api_key_env,
                             cfg.native_structured_outputs)
    return cfg


def make_client(cfg: ProviderConfig) -> anthropic.Anthropic:
    kwargs: dict = {}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise SystemExit(f"{cfg.api_key_env} is required for provider '{cfg.name}'")
        kwargs["api_key"] = key
    # For anthropic, let the SDK resolve credentials itself
    # (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile).
    return anthropic.Anthropic(**kwargs)
