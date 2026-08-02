"""Aggregazione dei pesi di attenzione e mapping tratto-demografico -> span di token,
per capire quanta attenzione un modello locale presta alla frase che descrive il tratto
di ciascuna persona nello scenario (es. "con precedenti penali")."""
from __future__ import annotations

import torch


def aggregate_attention(attentions, last_n_layers: int):
    """Media sulle ultime N layer e su tutte le head. `attentions` e' la tupla restituita
    da un forward pass con output_attentions=True: una entry per layer, shape
    [batch, heads, seq, seq]. Ritorna una matrice [seq, seq]."""
    layers = attentions[-last_n_layers:]
    stacked = torch.stack(layers)      # [n_layers, batch, heads, seq, seq]
    return stacked.mean(dim=(0, 1, 2))  # [seq, seq]


def attention_received_by_span(attn_matrix, offsets: list, char_start: int, char_end: int) -> float:
    """Attenzione media (su tutte le posizioni di query) ricevuta dai token del prompt
    che ricadono nello span di carattere [char_start, char_end)."""
    token_indices = [
        i for i, (s, e) in enumerate(offsets)
        if s is not None and e is not None and e > s and s < char_end and e > char_start
    ]
    if not token_indices:
        return float("nan")
    return attn_matrix[:, token_indices].mean().item()


def summarize_attention_for_scenario(
    attentions,
    offsets: list,
    full_text: str,
    last_n_layers: int,
    trait_phrases: dict,
) -> dict:
    """trait_phrases: es. {"person_a": "con precedenti penali", "person_b": "senza precedenti penali"}.
    Ritorna un dict {"person_a_attention": float, "person_b_attention": float}."""
    attn_matrix = aggregate_attention(attentions, last_n_layers)

    summary = {}
    for key, phrase in trait_phrases.items():
        if not phrase:
            summary[f"{key}_attention"] = None
            continue
        char_start = full_text.find(phrase)
        if char_start == -1:
            summary[f"{key}_attention"] = None
            continue
        char_end = char_start + len(phrase)
        value = attention_received_by_span(attn_matrix, offsets, char_start, char_end)
        summary[f"{key}_attention"] = None if value != value else value  # NaN -> None (serializzabile in JSON)

    return summary
