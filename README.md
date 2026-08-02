# Trolley Problem — Bias e Attenzione negli LLM

Progetto universitario di NLP: studia **come e perché** modelli linguistici diversi
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

**Non committare mai `.env`** (è già in `.gitignore`).

## Come eseguire

Apri `notebooks/run_experiment.ipynb` ed esegui le celle in ordine. Prima di lanciare il
run completo, è fortemente consigliato un test rapido su piccola scala: in
`config/config.yaml` riduci temporaneamente `repetitions: 1`, `n_scenarios_per_trait: 1`,
`n_control_scenarios: 2` e lascia un solo modello in `local_models`, per validare l'intera
pipeline (generazione scenari → provider → estrazione → cache → analisi → report) in
pochi minuti.

### Tempi stimati (configurazione di default: 50+ scenari, 5 ripetizioni, 3 modelli locali + Gemini)

- Modelli locali: ~5 ore di generazione + 10-15 min per l'estrazione dell'attenzione (su
  una sola ripetizione per scenario, per contenere tempo/VRAM).
- Gemini: 45-90 minuti circa (dipende dal rate limit del tuo tier).
- **Totale: 6-8 ore**, ma interrompibile e riprendibile in qualsiasi momento grazie alla
  cache in `results/cache/` — pensato per essere eseguito in sessioni separate (un
  modello locale alla volta), non in un'unica sessione continua. Per ridurre i tempi, la
  prima leva da abbassare è `repetitions` in `config.yaml`.

### Esportare il report in PDF (opzionale)

```bash
pandoc report/report.md -o report/report.pdf
```

## Note metodologiche

- **Attenzione**: disponibile solo per i modelli locali (l'attenzione interna di Gemini
  non è accessibile via API). Estratta con un singolo forward pass aggiuntivo in modalità
  eager, non durante la generazione autoregressiva (troppo costoso in tempo/memoria).
- **Scenari a tratto controllato**: ogni tratto viene isolato clonando una persona base in
  due varianti identiche tranne che nel tratto in esame, per evitare che più tratti
  confondano insieme l'effetto misurato.
- **Dataset sintetico**: `dataset/trolley.csv` è dati sintetici generati per esercizio, non
  riflette distribuzioni demografiche reali — vedi la sezione "Limiti" del report finale.
