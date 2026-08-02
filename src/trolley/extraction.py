"""Estrazione della decisione/framework dalla risposta strutturata di stage 2, con
fallback a un classificatore a keyword sul ragionamento libero di stage 1 quando il
parsing strutturato fallisce (raro con modelli instruct e decoding greedy, ma va gestito)."""
from __future__ import annotations

import re
from dataclasses import dataclass

_DECISION_RE = re.compile(r"DECISIONE:\s*(AZIONE|INAZIONE)", re.IGNORECASE)
_FRAMEWORK_RE = re.compile(r"FRAMEWORK:\s*(UTILITARISTA|DEONTOLOGICO|MISTO)", re.IGNORECASE)

_ACTION_KEYWORDS = [r"tir\w+ la leva", r"devi\w+ il carrello", r"decido di (agire|intervenire|deviare)"]
_INACTION_KEYWORDS = [r"non tiro", r"non agisco", r"non intervengo", r"rifiuto di (agire|intervenire)", r"lascio che"]
_UTILITARIAN_KEYWORDS = [r"utilitarist\w+", r"massimizz\w+ il bene", r"maggior numero", r"conseguenz\w+"]
_DEONTOLOGICAL_KEYWORDS = [r"deontolog\w+", r"kant\w+", r"dovere morale", r"dignità", r"imperativo categorico"]


@dataclass
class Extraction:
    decision: str   # "azione" | "inazione" | "unclear"
    framework: str  # "utilitarista" | "deontologico" | "misto" | "unclear"
    method: str      # "structured" | "fallback_keyword"


def extract(stage2_text: str, stage1_text: str) -> Extraction:
    decision_match = _DECISION_RE.search(stage2_text)
    if decision_match:
        framework_match = _FRAMEWORK_RE.search(stage2_text)
        return Extraction(
            decision=decision_match.group(1).lower(),
            framework=(framework_match.group(1).lower() if framework_match else "unclear"),
            method="structured",
        )
    return _fallback_from_keywords(stage1_text)


def _fallback_from_keywords(text: str) -> Extraction:
    lowered = text.lower()

    decision = "unclear"
    if any(re.search(p, lowered) for p in _ACTION_KEYWORDS):
        decision = "azione"
    elif any(re.search(p, lowered) for p in _INACTION_KEYWORDS):
        decision = "inazione"

    util_score = sum(len(re.findall(p, lowered)) for p in _UTILITARIAN_KEYWORDS)
    deont_score = sum(len(re.findall(p, lowered)) for p in _DEONTOLOGICAL_KEYWORDS)
    if util_score == 0 and deont_score == 0:
        framework = "unclear"
    elif util_score > deont_score:
        framework = "utilitarista"
    elif deont_score > util_score:
        framework = "deontologico"
    else:
        framework = "misto"

    return Extraction(decision=decision, framework=framework, method="fallback_keyword")
