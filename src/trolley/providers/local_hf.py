"""Provider per modelli locali via HuggingFace transformers, con supporto reale
all'estrazione dell'attenzione.

Nota tecnica importante (bug del vecchio analyzer.ipynb): `output_attentions=True`
richiede l'implementazione di attenzione "eager" — con quella di default (SDPA) le
attenzioni non vengono restituite. Ma tenere il modello in eager PERMANENTEMENTE rende
la generazione autoregressiva ~50x piu' lenta (misurato: ~5s/token invece di
~0.1-0.2s/token), perche' eager non usa i kernel fusi di SDPA. Soluzione: il modello
resta in SDPA per `generate()`, e passa a eager solo per la durata del singolo forward
pass extra usato per estrarre l'attenzione (`model.set_attn_implementation(...)`,
disponibile da transformers>=4.48), poi torna subito a SDPA.

Nota su bfloat16: con attenzione eager + quantizzazione 4-bit, il compute dtype
float16 produce NaN/inf nei punteggi di attenzione durante il campionamento
(`torch.multinomial` fallisce con "probability tensor contains inf, nan").
bfloat16 ha un range dinamico maggiore ed evita l'overflow.
"""
from __future__ import annotations

import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base import Capabilities, Provider, ProviderResponse


class LocalHFProvider(Provider):
    def __init__(self, model_id: str, label: str, hf_token: Optional[str] = None):
        self.model_id = model_id
        self.label = label
        self.capabilities = Capabilities(attention=True, logprobs=False)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=bnb_config,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _render_prompt(self, messages: list) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Alcuni template (es. Mistral-Instruct-v0.1) non supportano il ruolo "system":
            # uniamo il contenuto di sistema nel primo turno utente.
            merged = []
            pending_system = None
            for m in messages:
                if m["role"] == "system":
                    pending_system = m["content"]
                    continue
                if m["role"] == "user" and pending_system is not None:
                    merged.append({"role": "user", "content": f"{pending_system}\n\n{m['content']}"})
                    pending_system = None
                else:
                    merged.append(m)
            return self.tokenizer.apply_chat_template(
                merged, tokenize=False, add_generation_prompt=True
            )

    def generate(
        self,
        messages: list,
        *,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> ProviderResponse:
        prompt_text = self._render_prompt(messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
        if do_sample:
            gen_kwargs["temperature"] = temperature
        if self.tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        completion_ids = output_ids[:, input_len:]
        text = self.tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()

        return ProviderResponse(
            text=text,
            finish_reason=None,
            prompt_tokens=input_len,
            completion_tokens=completion_ids.shape[1],
            latency_ms=latency_ms,
            raw={"prompt_text": prompt_text},
        )

    def get_attentions(self, messages: list, response_text: str):
        """Singolo forward pass (teacher-forcing, nessun campionamento) su prompt+risposta
        gia' generata: molto piu' economico che estrarre l'attenzione durante la
        generazione autoregressiva token per token. Passa a eager solo per la durata di
        questo forward pass, poi torna a SDPA (necessario per non rallentare le
        `generate()` successive - vedi nota nel docstring del modulo).

        Ritorna (attentions, offsets, prompt_len, full_text).
        """
        prompt_text = self._render_prompt(messages)
        full_text = prompt_text + response_text

        encoding = self.tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
        offsets = encoding.pop("offset_mapping")[0].tolist()
        encoding = {k: v.to(self.model.device) for k, v in encoding.items()}
        prompt_len = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

        self.model.set_attn_implementation("eager")
        try:
            with torch.no_grad():
                outputs = self.model(**encoding, output_attentions=True)
        finally:
            self.model.set_attn_implementation("sdpa")

        return outputs.attentions, offsets, prompt_len, full_text

    def unload(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
