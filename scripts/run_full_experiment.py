"""Esegue l'esperimento completo end-to-end (equivalente a notebooks/run_experiment.ipynb
ma come script, pensato per un'esecuzione lunga e non presidiata in background).

Ogni stadio e' avvolto in modo che un fallimento (es. un modello che non carica, un
provider che va in errore) non impedisca ai passi successivi - inclusi analisi e report -
di girare comunque sui dati gia' raccolti in cache.
"""
import datetime
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trolley.config import load_config
from trolley.cache import ResponseCache
from trolley.providers.local_hf import LocalHFProvider
from trolley.providers.gemini import GeminiProvider
from trolley import scenario_generator, pipeline, analysis, plots, report


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    log(f"Config caricata: {len(config.traits_of_interest)} tratti, "
        f"{config.n_scenarios_per_trait} scenari/tratto, {config.n_control_scenarios} di controllo, "
        f"{config.repetitions} ripetizioni, modelli locali={[m.label for m in config.local_models]}, "
        f"gemini={config.gemini_model}")

    scenarios = scenario_generator.load_scenarios(config)
    log(f"Scenari: {len(scenarios)}")

    cache = ResponseCache(config.paths.cache_dir)

    for model_cfg in config.local_models:
        log(f"=== Modello locale: {model_cfg.label} ({model_cfg.id}) ===")
        t0 = time.time()
        try:
            provider = LocalHFProvider(model_cfg.id, model_cfg.label, hf_token=config.hf_token)
        except Exception:
            log(f"ERRORE nel caricamento di {model_cfg.label}, salto al modello successivo:")
            traceback.print_exc()
            continue

        try:
            pipeline.run_for_provider(provider, scenarios, config, cache)
        except Exception:
            log(f"ERRORE durante l'esecuzione di {model_cfg.label} (proseguo comunque):")
            traceback.print_exc()
        finally:
            provider.unload()
            log(f"--- {model_cfg.label} completato in {(time.time()-t0)/60:.1f} min ---")

    log("=== Gemini ===")
    t0 = time.time()
    try:
        gemini_provider = GeminiProvider(config.gemini_model, label="gemini", api_key=config.gemini_api_key)
        pipeline.run_for_provider(gemini_provider, scenarios, config, cache)
    except Exception:
        log("ERRORE durante l'esecuzione di Gemini (proseguo comunque con l'analisi):")
        traceback.print_exc()
    log(f"--- Gemini completato in {(time.time()-t0)/60:.1f} min ---")

    log("=== Analisi ===")
    df = analysis.build_flat_dataframe(cache)
    df = analysis.add_target_attention(df)
    analysis.save_flat_csv(df, config.paths.responses_flat_csv)
    log(f"Righe totali raccolte: {len(df)}")

    bias_df = analysis.run_bias_regressions(df)
    bias_df.to_csv(config.paths.bias_regression_csv, index=False)

    cross_model = analysis.run_cross_model_tests(df)
    log(f"Cross-model: {cross_model}")

    log("=== Grafici ===")
    plots.plot_decision_distribution(df, config.paths.figures_dir)
    plots.plot_framework_distribution(df, config.paths.figures_dir)
    plots.plot_bias_regression(bias_df, config.paths.figures_dir)
    plots.plot_attention_vs_bias(df, config.paths.figures_dir)

    log("=== Report ===")
    report.generate_report(
        df, bias_df, cross_model,
        figures_dir=config.paths.figures_dir,
        out_path=config.paths.report_path,
        generated_at=datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
    )
    log(f"Report scritto in {config.paths.report_path}")
    log("=== ESPERIMENTO COMPLETATO ===")


if __name__ == "__main__":
    main()
