"""Generazione delle figure finali in results/figures/."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _savefig(fig, out_dir: str, name: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decision_distribution(df: pd.DataFrame, out_dir: str) -> None:
    if df.empty:
        return
    counts = df.groupby(["label", "decision"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black")
    ax.set_title("Distribuzione delle decisioni per modello")
    ax.set_xlabel("")
    ax.set_ylabel("Conteggio risposte")
    ax.tick_params(axis="x", rotation=20)
    _savefig(fig, out_dir, "decision_distribution.png")


def plot_framework_distribution(df: pd.DataFrame, out_dir: str) -> None:
    if df.empty:
        return
    counts = df.groupby(["label", "framework"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax, colormap="Set1", edgecolor="black")
    ax.set_title("Framework etico dominante per modello")
    ax.set_xlabel("")
    ax.set_ylabel("Conteggio risposte")
    ax.tick_params(axis="x", rotation=20)
    _savefig(fig, out_dir, "framework_distribution.png")


def plot_bias_regression(bias_df: pd.DataFrame, out_dir: str) -> None:
    plot_df = bias_df.dropna(subset=["sacrifice_rate"]) if not bias_df.empty else bias_df
    if plot_df is None or plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="trait", y="sacrifice_rate", hue="model", ax=ax)
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.6, label="atteso per caso (50%)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Tasso di sacrificio della persona 'target'")
    ax.set_xlabel("Tratto variato")
    ax.set_title("Bias per tratto e per modello")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    _savefig(fig, out_dir, "bias_by_trait.png")


def plot_attention_vs_bias(df_with_attention: pd.DataFrame, out_dir: str) -> None:
    if "target_attention" not in df_with_attention.columns:
        return
    plot_df = df_with_attention.dropna(subset=["target_attention", "sacrifice_target"])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, sub in plot_df.groupby("label"):
        ax.scatter(sub["target_attention"], sub["sacrifice_target"].astype(float), label=model, alpha=0.6)
    ax.set_xlabel("Attenzione media ricevuta dalla frase del tratto 'target'")
    ax.set_ylabel("Persona 'target' sacrificata (1=si', 0=no)")
    ax.set_title("Attenzione al tratto vs. decisione di sacrificio")
    ax.legend()
    _savefig(fig, out_dir, "attention_vs_bias.png")
