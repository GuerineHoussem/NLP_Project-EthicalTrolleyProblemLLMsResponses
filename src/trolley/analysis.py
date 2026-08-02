"""Costruzione del dataset piatto a partire dalla cache delle risposte, regressioni sui
bias per tratto/modello, e test statistici di confronto tra modelli."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .cache import ResponseCache


def _harmed_side(decision: str):
    if decision == "azione":
        return "b"  # tirare la leva devia il carrello sul binario secondario
    if decision == "inazione":
        return "a"  # non fare nulla lascia il carrello sul binario principale
    return None


def build_flat_dataframe(cache: ResponseCache) -> pd.DataFrame:
    df = pd.DataFrame(cache.all_records())
    if df.empty:
        return df

    df["harmed_side"] = df["decision"].apply(_harmed_side)

    def sacrifice_target(row):
        if row["target_side"] is None or pd.isna(row["target_side"]) or row["harmed_side"] is None:
            return pd.NA
        return row["harmed_side"] == row["target_side"]

    df["sacrifice_target"] = df.apply(sacrifice_target, axis=1)
    return df


def add_target_attention(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge la colonna `target_attention`: l'attenzione ricevuta dalla frase del
    tratto sul lato ('a'/'b') che porta il valore 'target', quando disponibile."""

    def get_target_attention(row):
        summary = row.get("attention_summary")
        side = row.get("target_side")
        if not isinstance(summary, dict) or not side:
            return None
        return summary.get(f"person_{side}_attention")

    df = df.copy()
    df["target_attention"] = df.apply(get_target_attention, axis=1)
    return df


def save_flat_csv(df: pd.DataFrame, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def run_bias_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """Per ciascuna coppia (tratto, modello): la persona col valore 'target' del tratto
    viene sacrificata piu' spesso del 50% atteso per caso? Regressione logistica
    intercetta-sola con errori standard clusterizzati per scenario (le ripetizioni dello
    stesso scenario non sono osservazioni indipendenti)."""
    rows = []
    trait_df = df[df["varied_trait"].notna() & df["sacrifice_target"].notna()].copy()
    if trait_df.empty:
        return pd.DataFrame(rows)
    trait_df["sacrifice_target"] = trait_df["sacrifice_target"].astype(int)

    for (trait, model), sub in trait_df.groupby(["varied_trait", "label"]):
        n = len(sub)
        rate = sub["sacrifice_target"].mean()
        row = {"trait": trait, "model": model, "n": n, "sacrifice_rate": rate}

        if sub["sacrifice_target"].nunique() < 2 or n < 6:
            row.update(odds_ratio=None, p_value=None, note="dati insufficienti o nessuna varianza")
            rows.append(row)
            continue

        try:
            fit = smf.glm(
                "sacrifice_target ~ 1", data=sub, family=sm.families.Binomial()
            ).fit(cov_type="cluster", cov_kwds={"groups": sub["scenario_id"]})
            row.update(
                odds_ratio=float(np.exp(fit.params["Intercept"])),
                p_value=float(fit.pvalues["Intercept"]),
                note="",
            )
        except Exception as exc:
            row.update(odds_ratio=None, p_value=None, note=f"errore stima: {exc}")

        rows.append(row)

    return pd.DataFrame(rows)


def run_cross_model_tests(df: pd.DataFrame) -> dict:
    """Test chi-quadrato per verificare se decisione/framework dipendono dal modello,
    e tasso di utilizzo del fallback a keyword (indicatore di affidabilita' dei dati)."""
    valid = df[df["decision"].isin(["azione", "inazione"])].copy()
    result = {
        "action_rate_by_model": {},
        "chi2_decision": {"statistic": None, "p_value": None},
        "chi2_framework": {"statistic": None, "p_value": None},
        "extraction_method_rate": df["extraction_method"].value_counts(normalize=True).round(3).to_dict(),
    }

    if not valid.empty:
        valid["is_action"] = (valid["decision"] == "azione").astype(int)
        result["action_rate_by_model"] = valid.groupby("label")["is_action"].mean().round(3).to_dict()

        if valid["label"].nunique() > 1:
            contingency = pd.crosstab(valid["label"], valid["is_action"])
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                result["chi2_decision"] = {"statistic": float(chi2), "p_value": float(p_value)}

    valid_fw = df[df["framework"].isin(["utilitarista", "deontologico", "misto"])]
    if not valid_fw.empty and valid_fw["label"].nunique() > 1:
        contingency_fw = pd.crosstab(valid_fw["label"], valid_fw["framework"])
        if contingency_fw.shape[0] > 1 and contingency_fw.shape[1] > 1:
            chi2_fw, p_fw, _, _ = stats.chi2_contingency(contingency_fw)
            result["chi2_framework"] = {"statistic": float(chi2_fw), "p_value": float(p_fw)}

    return result
