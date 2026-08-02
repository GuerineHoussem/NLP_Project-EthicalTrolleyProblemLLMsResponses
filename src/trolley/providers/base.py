"""Interfaccia comune a tutti i provider (modelli locali HuggingFace e Gemini), cosi'
la pipeline puo' trattarli in modo uniforme pur avendo capacita' diverse (solo i modelli
locali espongono l'attenzione interna)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Capabilities:
    attention: bool = False
    logprobs: bool = False


@dataclass
class ProviderResponse:
    text: str
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    raw: Any = None


class Provider(ABC):
    label: str
    model_id: str
    capabilities: Capabilities

    @abstractmethod
    def generate(
        self,
        messages: list,
        *,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> ProviderResponse:
        """Genera una risposta a partire da una conversazione (lista di {role, content})."""
        raise NotImplementedError
