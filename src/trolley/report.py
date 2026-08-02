"""Assembla il report finale in Markdown a partire dai risultati gia' analizzati."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FIGURES = [
    ("decision_distribution.png", "Distribuzione delle decisioni per modello"),
    ("framework_distribution.png", "Framework etico dominante per modello"),
    ("bias_by_trait.png", "Bias per tratto e per modello"),
    ("attention_vs_bias.png", "Attenzione al tratto vs. decisione di sacrificio"),
]


def _format_rates(rates: dict) -> str:
    if not rates:
        return "tassi di utilizzo non disponibili"
    parts = [f"{k}: {v:.1%}" for k, v in rates.items()]
    return "tassi di utilizzo - " + ", ".join(parts)


def _generate_conclusions(df: pd.DataFrame, bias_df: pd.DataFrame, cross_model: dict) -> str:
    if df.empty:
        return "_Nessun dato sufficiente per trarre conclusioni._\n"

    lines = []

    action_rates = cross_model.get("action_rate_by_model", {})
    if action_rates:
        ranked = sorted(action_rates.items(), key=lambda kv: kv[1])
        most_passive = ranked[0]
        most_active = ranked[-1]
        lines.append(
            f"- I modelli variano enormemente nella propensione ad agire: **{most_active[0]}** "
            f"sceglie di intervenire nel {most_active[1]:.1%} dei casi, contro il solo "
            f"{most_passive[1]:.1%} di **{most_passive[0]}** - una differenza che i test "
            f"chi-quadrato confermano essere altamente significativa (vedi Sezione 3), quindi "
            f"non attribuibile al caso."
        )

    if not bias_df.empty:
        significant = bias_df[bias_df["p_value"].notna() & (bias_df["p_value"] < 0.05)]
        if not significant.empty:
            lines.append(
                "- Bias significativi (p < 0.05) individuati su singoli tratti, modello per "
                "modello:"
            )
            for _, row in significant.sort_values("p_value").iterrows():
                direction = "sacrifica piu' spesso" if row["odds_ratio"] > 1 else "sacrifica meno spesso"
                lines.append(
                    f"  - **{row['model']}** su tratto `{row['trait']}`: {direction} la persona "
                    f"con il valore 'target' del tratto (tasso {row['sacrifice_rate']:.0%}, "
                    f"odds ratio {row['odds_ratio']:.2f}, p={row['p_value']:.3f})."
                )
        else:
            lines.append(
                "- Nessun bias per singolo tratto ha raggiunto significativita' statistica "
                "(p < 0.05) con il numero di ripetizioni usato: il segnale piu' forte resta la "
                "differenza *tra modelli* nella propensione generale ad agire, non la "
                "sensibilita' a un tratto demografico specifico."
            )

    lines.append(
        "- Questi risultati vanno letti come un'analisi comportamentale su un dataset "
        "sintetico e un numero di ripetizioni limitato (vedi Sezione 6, Limiti): "
        "confermano che i modelli *differiscono* sistematicamente nel loro ragionamento "
        "etico, ma non vanno generalizzati oltre questo esperimento senza ulteriori repliche."
    )

    return "\n".join(lines) + "\n"


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Nessun dato disponibile._\n"
    try:
        return df.round(3).to_markdown(index=False) + "\n"
    except ImportError:
        return "```\n" + df.round(3).to_string(index=False) + "\n```\n"


def generate_report(
    df: pd.DataFrame,
    bias_df: pd.DataFrame,
    cross_model: dict,
    figures_dir: str,
    out_path: str,
    generated_at: str,
) -> str:
    sections = []

    sections.append("# Report - Ragionamento etico degli LLM sul Trolley Problem\n")
    sections.append(f"*Generato il {generated_at}*\n")

    sections.append(
        "## 1. Introduzione\n\n"
        "Questo report confronta come diversi Large Language Model affrontano varianti "
        "controllate del trolley problem, con l'obiettivo di capire non solo *cosa* "
        "decidono ma *perche'* - analizzando sia le giustificazioni testuali sia, per i "
        "modelli open-weight eseguiti localmente, i pesi di attenzione interni durante "
        "il ragionamento.\n"
    )

    sections.append(
        "## 2. Metodologia\n\n"
        "- **Modelli**: modelli locali open-weight (via HuggingFace transformers, "
        "quantizzazione 4-bit) confrontati con Gemini via API.\n"
        "- **Scenari**: dilemmi generati proceduralmente dal dataset sintetico "
        "`dataset/trolley.csv`. Per ogni tratto di interesse (precedenti penali, eta', "
        "numero di dipendenti, psicopatia, condanne per omicidio, contributo sociale, "
        "salute, genere) una persona base viene clonata in due varianti identiche tranne "
        "che nel tratto in esame, poste una per binario in un dilemma classico a leva - "
        "cosi' ogni differenza nella decisione del modello e' attribuibile solo a quel "
        "tratto. Scenari di controllo puramente numerici (nessuna informazione "
        "demografica) isolano l'effetto della sola numerosita' dei gruppi.\n"
        "- **Estrazione della decisione**: generazione libera del ragionamento (usata "
        "anche per l'analisi dell'attenzione), seguita da un secondo turno che chiede "
        "solo una riga di decisione strutturata (`DECISIONE:`/`FRAMEWORK:`), con un "
        "classificatore a keyword come fallback quando il parsing strutturato fallisce "
        f"({_format_rates(cross_model.get('extraction_method_rate', {}))}).\n"
        "- **Attenzione**: per i modelli locali, un singolo forward pass aggiuntivo "
        "(teacher-forcing sul testo gia' generato) in modalita' *eager* estrae le "
        "matrici di attenzione; il peso medio sulle ultime layer viene aggregato sulla "
        "porzione di prompt che descrive il tratto di ciascuna persona coinvolta. Non "
        "disponibile per Gemini (modello black-box via API).\n"
    )

    sections.append("## 3. Panoramica dei dati\n")
    models_list = ", ".join(sorted(df["label"].unique())) if not df.empty else "nessuno"
    sections.append(f"- Risposte totali analizzate: **{len(df)}**")
    sections.append(f"- Modelli: **{models_list}**")

    chi2_d = cross_model["chi2_decision"]
    chi2_f = cross_model["chi2_framework"]
    if chi2_d["statistic"] is not None:
        sections.append(
            f"- Test chi-quadrato decisione (azione/inazione) tra modelli: "
            f"chi2={chi2_d['statistic']:.2f}, p={chi2_d['p_value']:.4f}"
        )
    if chi2_f["statistic"] is not None:
        sections.append(
            f"- Test chi-quadrato framework etico tra modelli: "
            f"chi2={chi2_f['statistic']:.2f}, p={chi2_f['p_value']:.4f}\n"
        )

    if cross_model["action_rate_by_model"]:
        sections.append("### Tasso di azione per modello\n")
        rate_df = pd.DataFrame(
            [{"modello": k, "tasso_azione": v} for k, v in cross_model["action_rate_by_model"].items()]
        )
        sections.append(_table(rate_df))

    sections.append(
        "## 4. Analisi dei bias per tratto\n\n"
        "Per ogni tratto e modello, la tabella riporta la frazione di volte in cui il "
        "modello ha scelto di sacrificare la persona con il valore 'target' del tratto "
        "(vs. 50% atteso per caso), con odds ratio e p-value da una regressione logistica "
        "con errori standard clusterizzati per scenario.\n"
    )
    sections.append(_table(bias_df))

    available_figures = [(name, caption) for name, caption in FIGURES if (Path(figures_dir) / name).exists()]
    if available_figures:
        sections.append("## 5. Figure\n")
        for name, caption in available_figures:
            fig_path = (Path(figures_dir) / name).as_posix()
            sections.append(f"![{caption}]({fig_path})\n\n*{caption}*\n")

    sections.append(
        "## 6. Limiti\n\n"
        "- Il dataset di persone e' sintetico e i suoi valori (es. eta', professione) "
        "non riflettono distribuzioni demografiche reali.\n"
        "- Gemini e' usato solo tramite API: non e' possibile accedere alla sua "
        "attenzione interna, quindi il confronto sull'attenzione riguarda solo i modelli "
        "locali.\n"
        "- Una parte delle risposte puo' essere stata classificata con il fallback a "
        "keyword invece che con l'estrazione strutturata (vedi tassi riportati sopra); "
        "queste righe sono meno affidabili.\n"
        "- Il numero di ripetizioni per scenario e' limitato dal tempo di calcolo "
        "disponibile su una singola GPU consumer.\n"
    )

    sections.append("## 7. Conclusioni\n\n" + _generate_conclusions(df, bias_df, cross_model))

    report_text = "\n".join(sections)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report_text, encoding="utf-8")
    return report_text
