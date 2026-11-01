from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelPricing:
    model_id: str
    label: str
    input_price_per_1m: float | None
    output_price_per_1m: float | None
    source: str
    status: str


@dataclass(frozen=True)
class CostEstimate:
    model_id: str
    pricing_status: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float | None
    output_cost_usd: float | None
    cost_per_call_usd: float | None


class BedrockPricingCatalog:
    def __init__(self, version: str, currency: str, models: dict[str, ModelPricing]) -> None:
        self.version = version
        self.currency = currency
        self.models = models

    def get(self, model_id: str) -> ModelPricing | None:
        return self.models.get(model_id)

    def estimate_call(
        self,
        model_id: str,
        prompt: str,
        context_texts: list[str],
        history: list[str],
        output_text: str,
    ) -> CostEstimate:
        pricing = self.get(model_id)
        input_tokens = _estimate_tokens(prompt) + sum(
            _estimate_tokens(text) for text in context_texts
        ) + sum(_estimate_tokens(line) for line in history)
        output_tokens = _estimate_tokens(output_text)

        if (
            pricing is None
            or pricing.input_price_per_1m is None
            or pricing.output_price_per_1m is None
        ):
            return CostEstimate(
                model_id=model_id,
                pricing_status="missing",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost_usd=None,
                output_cost_usd=None,
                cost_per_call_usd=None,
            )

        input_cost = (input_tokens / 1_000_000) * pricing.input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_1m
        cost = input_cost + output_cost
        return CostEstimate(
            model_id=model_id,
            pricing_status=pricing.status,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=round(input_cost, 8),
            output_cost_usd=round(output_cost, 8),
            cost_per_call_usd=round(cost, 8),
        )


def load_bedrock_pricing_catalog(path: Path | None = None) -> BedrockPricingCatalog:
    pricing_path = path or _default_pricing_path()
    payload = json.loads(pricing_path.read_text(encoding="utf-8"))
    models = {
        model_id: ModelPricing(
            model_id=model_id,
            label=str(model_payload.get("label", model_id)),
            input_price_per_1m=_optional_float(model_payload.get("input_price_per_1m")),
            output_price_per_1m=_optional_float(model_payload.get("output_price_per_1m")),
            source=str(model_payload.get("source", "")),
            status=str(model_payload.get("status", "unknown")),
        )
        for model_id, model_payload in payload.get("models", {}).items()
    }
    return BedrockPricingCatalog(
        version=str(payload.get("version", "unknown")),
        currency=str(payload.get("currency", "USD")),
        models=models,
    )


def _default_pricing_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "bedrock_pricing.json"


def _estimate_tokens(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, math.ceil(len(normalized) / 4))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
