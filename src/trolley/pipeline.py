"""Orchestrazione dell'esperimento: per ogni (provider, scenario, ripetizione) genera
stage 1 (ragionamento libero) + stage 2 (estrazione strutturata), classifica la
decisione, opzionalmente estrae l'attenzione (solo scenari a tratto controllato, solo
modelli locali, solo sulle prime `attention.repetitions_sampled` ripetizioni), e salva
tutto nella cache su disco (idempotente: riprendibile dopo un crash)."""
from __future__ import annotations

import json

from .attention import summarize_attention_for_scenario
from .cache import ResponseCache
from .config import Config
from .extraction import extract
from .prompts import build_stage1_messages, build_stage2_messages
from .providers.base import Provider


def run_for_provider(
    provider: Provider,
    scenarios: list,
    config: Config,
    cache: ResponseCache,
    verbose: bool = True,
) -> None:
    gen = config.generation

    for scenario in scenarios:
        for rep in range(config.repetitions):
            stage1_messages = build_stage1_messages(scenario["description"])
            key = ResponseCache.make_key(
                provider.label, scenario["id"], rep, json.dumps(stage1_messages, ensure_ascii=False)
            )
            if cache.get(key) is not None:
                continue

            try:
                stage1 = provider.generate(
                    stage1_messages,
                    max_new_tokens=gen.stage1_max_new_tokens,
                    temperature=gen.stage1_temperature,
                    do_sample=True,
                )

                stage2_messages = build_stage2_messages(scenario["description"], stage1.text)
                stage2 = provider.generate(
                    stage2_messages,
                    max_new_tokens=gen.stage2_max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                )
            except Exception as exc:
                # Un errore transitorio (timeout API, OOM occasionale, risposta anomala) non
                # deve far perdere ore di lavoro gia' completato: si salta questo (scenario,
                # ripetizione) - non viene messo in cache, quindi un rerun futuro lo ritenta.
                print(f"[{provider.label}] {scenario['id']} rep={rep} FALLITO: {exc}")
                continue

            result = extract(stage2.text, stage1.text)

            attention_summary = None
            if (
                provider.capabilities.attention
                and config.attention.enabled
                and rep < config.attention.repetitions_sampled
                and scenario.get("varied_trait")
            ):
                attention_summary = _extract_attention_safe(provider, stage1_messages, stage1.text, scenario, config)

            record = {
                "provider": type(provider).__name__,
                "model": getattr(provider, "model_id", provider.label),
                "label": provider.label,
                "scenario_id": scenario["id"],
                "scenario_family": scenario["family"],
                "varied_trait": scenario.get("varied_trait"),
                "trait_value_target": scenario.get("trait_value_target"),
                "trait_value_reference": scenario.get("trait_value_reference"),
                "target_side": scenario.get("target_side"),
                "group_a_size": scenario.get("group_a_size"),
                "group_b_size": scenario.get("group_b_size"),
                "repetition": rep,
                "stage1_response": stage1.text,
                "stage2_response": stage2.text,
                "decision": result.decision,
                "framework": result.framework,
                "extraction_method": result.method,
                "attention_available": attention_summary is not None,
                "attention_summary": attention_summary,
                "prompt_tokens": stage1.prompt_tokens,
                "completion_tokens": stage1.completion_tokens,
                "latency_ms": stage1.latency_ms,
            }
            cache.put(key, record)

            if verbose:
                print(f"[{provider.label}] {scenario['id']} rep={rep} -> {result.decision}/{result.framework}")


def _extract_attention_safe(provider, stage1_messages, stage1_text, scenario, config):
    try:
        attentions, offsets, prompt_len, full_text = provider.get_attentions(stage1_messages, stage1_text)
    except Exception as exc:  # un problema hardware/memoria non deve far fallire l'intero run
        print(f"   attenzione non estratta per {scenario['id']}: {exc}")
        return None

    trait_phrases = {
        "person_a": scenario.get("person_a_trait_phrase"),
        "person_b": scenario.get("person_b_trait_phrase"),
    }
    return summarize_attention_for_scenario(
        attentions, offsets, full_text, config.attention.last_n_layers, trait_phrases
    )
