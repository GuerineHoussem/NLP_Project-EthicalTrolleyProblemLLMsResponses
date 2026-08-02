"""Provider per Gemini via SDK `google-genai`. Nessun accesso all'attenzione interna
(modello black-box via API): capabilities.attention resta sempre False.

Il tier gratuito ha un limite di richieste/minuto molto basso (15 per
gemini-3.1-flash-lite): senza rate-limiting lato client, un run di centinaia di
chiamate esaurisce la quota in pochi secondi e fallisce quasi tutto (visto in pratica:
242/250 fallite con 429 RESOURCE_EXHAUSTED). Questo provider quindi limita attivamente
il ritmo delle chiamate e ritenta con backoff sui 429."""
from __future__ import annotations

import time
from collections import deque
from typing import Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from .base import Capabilities, Provider, ProviderResponse

_ROLE_MAP = {"user": "user", "assistant": "model"}

# I modelli Gemini con "thinking" consumano parte di max_output_tokens per ragionamento
# interno non visibile (fino all'80-90% del budget anche con thinking_level="low"), quindi
# il tetto richiesto all'API deve avere un margine ampio oltre alla lunghezza di testo
# visibile desiderata, altrimenti la risposta viene troncata a zero token visibili.
_THINKING_HEADROOM_TOKENS = 1000

_MAX_CALLS_PER_MINUTE = 10  # sotto il limite free-tier di 15/min, con margine di sicurezza
_MAX_RETRIES = 6
_RETRY_BACKOFF_SECONDS = 20


class GeminiProvider(Provider):
    def __init__(self, model_id: str, label: str, api_key: str):
        self.model_id = model_id
        self.label = label
        self.capabilities = Capabilities(attention=False, logprobs=False)
        self._client = genai.Client(api_key=api_key)
        self._call_timestamps: deque = deque()

    def _wait_for_rate_limit(self) -> None:
        now = time.monotonic()
        while self._call_timestamps and now - self._call_timestamps[0] > 60:
            self._call_timestamps.popleft()
        if len(self._call_timestamps) >= _MAX_CALLS_PER_MINUTE:
            sleep_for = 60 - (now - self._call_timestamps[0]) + 0.5
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._call_timestamps.append(time.monotonic())

    def generate(
        self,
        messages: list,
        *,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> ProviderResponse:
        system_content = next((m["content"] for m in messages if m["role"] == "system"), None)
        history = [m for m in messages if m["role"] != "system"]

        contents = [
            genai_types.Content(role=_ROLE_MAP[m["role"]], parts=[genai_types.Part(text=m["content"])])
            for m in history
        ]

        config = genai_types.GenerateContentConfig(
            system_instruction=system_content,
            max_output_tokens=max_new_tokens + _THINKING_HEADROOM_TOKENS,
            temperature=temperature if do_sample else 0.0,
            thinking_config=genai_types.ThinkingConfig(thinking_level="low"),
        )

        start = time.perf_counter()
        response = self._call_with_retry(contents, config)
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage_metadata
        return ProviderResponse(
            text=(response.text or "").strip(),
            finish_reason=_first_finish_reason(response),
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            completion_tokens=getattr(usage, "candidates_token_count", None),
            latency_ms=latency_ms,
            raw=None,
        )

    def _call_with_retry(self, contents, config):
        for attempt in range(_MAX_RETRIES):
            self._wait_for_rate_limit()
            try:
                return self._client.models.generate_content(model=self.model_id, contents=contents, config=config)
            except genai_errors.ClientError as exc:
                if exc.code != 429 or attempt == _MAX_RETRIES - 1:
                    raise
                wait = _RETRY_BACKOFF_SECONDS * (attempt + 1)
                print(f"   [gemini] 429 rate limit, retry {attempt + 1}/{_MAX_RETRIES} tra {wait}s")
                time.sleep(wait)
        raise RuntimeError("unreachable")  # pragma: no cover


def _first_finish_reason(response) -> Optional[str]:
    candidates = response.candidates or []
    if candidates:
        reason = candidates[0].finish_reason
        return str(reason) if reason is not None else None
    return None
