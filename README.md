# Trolley Problem — Bias e Attenzione negli LLM

Progetto di NLP: **come e perché** modelli linguistici diversi
ragionano diversamente sul *trolley problem*, confrontando modelli locali open-weight
(con vera analisi dell'attenzione interna) e Gemini via API, su scenari generati in modo
controllato per isolare l'effetto di singoli tratti demografici (età, precedenti penali,
numero di dipendenti, ecc.) sulle decisioni del modello.

## Struttura

```
config/config.yaml           configurazione centrale (modelli, n. scenari, ripetizioni)
dataset/trolley.csv           popolazione sintetica usata per generare le persone negli scenari
scenarios/generated_scenarios.json   scenari generati (riproducibili, seed fisso)
src/trolley/                  package Python con tutta la logica (vedi sotto)
notebooks/run_experiment.ipynb   notebook sottile: orchestra i moduli, nessuna logica pesante
results/                      cache grezza, dataset piatto CSV, regressioni, figure
report/report.md              report dati generato automaticamente dalla pipeline
report/paper.md                paper accademico completo (introduzione, metodologia, risultati, discussione, limiti, conclusioni, riferimenti)
```

Moduli in `src/trolley/`:

| Modulo | Responsabilità |
|---|---|
| `config.py` | carica `config.yaml` + variabili d'ambiente (`.env`) |
| `people.py` | popolazione sintetica, cloni di persone a tratto controllato |
| `scenario_generator.py` | genera scenari a tratto controllato + scenari di controllo numerico |
| `prompts.py` | prompt di ragionamento (stage 1) ed estrazione strutturata (stage 2) |
| `providers/local_hf.py` | modelli locali via `transformers` (4-bit, `attn_implementation="eager"`) |
| `providers/gemini.py` | Gemini via SDK `google-genai` |
| `cache.py` | cache JSONL su disco (esperimento riprendibile dopo un crash) |
| `extraction.py` | parsing della decisione strutturata + fallback a keyword |
| `attention.py` | aggregazione attenzione e mapping tratto→span di testo |
| `pipeline.py` | orchestrazione: per ogni (provider, scenario, ripetizione) |
| `analysis.py` | dataset piatto, regressioni sui bias, test cross-modello |
| `plots.py` | figure in `results/figures/` |
| `report.py` | assembla `report/report.md` |

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

copy .env.example .env         # poi inserisci HF_TOKEN e GEMINI_API_KEY reali
```

`HF_TOKEN`: necessario per Llama-3.1 (gated) — https://huggingface.co/settings/tokens
`GEMINI_API_KEY`: https://aistudio.google.com/apikey


