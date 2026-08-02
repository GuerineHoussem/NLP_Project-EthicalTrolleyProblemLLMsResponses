"""Caricamento della configurazione dell'esperimento (config/config.yaml) e delle
variabili d'ambiente (.env). Nessuna chiave API va mai scritta qui o in config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GenerationConfig:
    stage1_max_new_tokens: int = 400
    stage1_temperature: float = 0.7
    stage2_max_new_tokens: int = 30


@dataclass
class AttentionConfig:
    enabled: bool = True
    repetitions_sampled: int = 1
    last_n_layers: int = 8


@dataclass
class LocalModelConfig:
    id: str
    label: str


@dataclass
class Paths:
    cache_dir: str
    responses_flat_csv: str
    bias_regression_csv: str
    figures_dir: str
    report_path: str


@dataclass
class Config:
    dataset_path: str
    scenarios_path: str
    scenario_seed: int
    traits_of_interest: list
    n_scenarios_per_trait: int
    n_control_scenarios: int
    repetitions: int
    local_models: list
    gemini_model: str
    generation: GenerationConfig
    attention: AttentionConfig
    paths: Paths

    @property
    def hf_token(self) -> Optional[str]:
        return os.environ.get("HF_TOKEN") or None

    @property
    def gemini_api_key(self) -> Optional[str]:
        return os.environ.get("GEMINI_API_KEY") or None


def load_config(path: str = "config/config.yaml") -> Config:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    return Config(
        dataset_path=raw["dataset_path"],
        scenarios_path=raw["scenarios_path"],
        scenario_seed=raw["scenario_seed"],
        traits_of_interest=raw["traits_of_interest"],
        n_scenarios_per_trait=raw["n_scenarios_per_trait"],
        n_control_scenarios=raw["n_control_scenarios"],
        repetitions=raw["repetitions"],
        local_models=[LocalModelConfig(**m) for m in raw["local_models"]],
        gemini_model=raw["gemini_model"],
        generation=GenerationConfig(**raw["generation"]),
        attention=AttentionConfig(**raw["attention"]),
        paths=Paths(**raw["paths"]),
    )
