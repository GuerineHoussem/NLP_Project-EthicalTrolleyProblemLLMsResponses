# Report - Ragionamento etico degli LLM sul Trolley Problem

*Generato il 01/08/2026 03:06*

## 1. Introduzione

Questo report confronta come diversi Large Language Model affrontano varianti controllate del trolley problem, con l'obiettivo di capire non solo *cosa* decidono ma *perche'* - analizzando sia le giustificazioni testuali sia, per i modelli open-weight eseguiti localmente, i pesi di attenzione interni durante il ragionamento.

## 2. Metodologia

- **Modelli**: modelli locali open-weight (via HuggingFace transformers, quantizzazione 4-bit) confrontati con Gemini via API.
- **Scenari**: dilemmi generati proceduralmente dal dataset sintetico `dataset/trolley.csv`. Per ogni tratto di interesse (precedenti penali, eta', numero di dipendenti, psicopatia, condanne per omicidio, contributo sociale, salute, genere) una persona base viene clonata in due varianti identiche tranne che nel tratto in esame, poste una per binario in un dilemma classico a leva - cosi' ogni differenza nella decisione del modello e' attribuibile solo a quel tratto. Scenari di controllo puramente numerici (nessuna informazione demografica) isolano l'effetto della sola numerosita' dei gruppi.
- **Estrazione della decisione**: generazione libera del ragionamento (usata anche per l'analisi dell'attenzione), seguita da un secondo turno che chiede solo una riga di decisione strutturata (`DECISIONE:`/`FRAMEWORK:`), con un classificatore a keyword come fallback quando il parsing strutturato fallisce (tassi di utilizzo - structured: 99.9%, fallback_keyword: 0.1%).
- **Attenzione**: per i modelli locali, un singolo forward pass aggiuntivo (teacher-forcing sul testo gia' generato) in modalita' *eager* estrae le matrici di attenzione; il peso medio sulle ultime layer viene aggregato sulla porzione di prompt che descrive il tratto di ciascuna persona coinvolta. Non disponibile per Gemini (modello black-box via API).

## 3. Panoramica dei dati

- Risposte totali analizzate: **994**
- Modelli: **gemini, llama-3.1-8b, mistral-7b, qwen2.5-7b**
- Test chi-quadrato decisione (azione/inazione) tra modelli: chi2=476.18, p=0.0000
- Test chi-quadrato framework etico tra modelli: chi2=581.27, p=0.0000

### Tasso di azione per modello

| modello      |   tasso_azione |
|:-------------|---------------:|
| gemini       |          0.049 |
| llama-3.1-8b |          0.448 |
| mistral-7b   |          0.928 |
| qwen2.5-7b   |          0.164 |

## 4. Analisi dei bias per tratto

Per ogni tratto e modello, la tabella riporta la frazione di volte in cui il modello ha scelto di sacrificare la persona con il valore 'target' del tratto (vs. 50% atteso per caso), con odds ratio e p-value da una regressione logistica con errori standard clusterizzati per scenario.

| trait                    | model        |   n |   sacrifice_rate |   odds_ratio |   p_value | note                                  |
|:-------------------------|:-------------|----:|-----------------:|-------------:|----------:|:--------------------------------------|
| age                      | gemini       |  24 |            0.625 |        1.667 |     0.618 |                                       |
| age                      | llama-3.1-8b |  25 |            0.56  |        1.273 |     0.544 |                                       |
| age                      | mistral-7b   |  25 |            0.4   |        0.667 |     0.691 |                                       |
| age                      | qwen2.5-7b   |  25 |            0.6   |        1.5   |     0.691 |                                       |
| contribution_to_humanity | gemini       |  25 |            0.6   |        1.5   |     0.691 |                                       |
| contribution_to_humanity | llama-3.1-8b |  25 |            0.72  |        2.571 |     0.017 |                                       |
| contribution_to_humanity | mistral-7b   |  25 |            0.44  |        0.786 |     0.797 |                                       |
| contribution_to_humanity | qwen2.5-7b   |  25 |            0.52  |        1.083 |     0.914 |                                       |
| criminal                 | gemini       |  25 |            0.4   |        0.667 |     0.691 |                                       |
| criminal                 | llama-3.1-8b |  25 |            0.52  |        1.083 |     0.803 |                                       |
| criminal                 | mistral-7b   |  25 |            0.64  |        1.778 |     0.552 |                                       |
| criminal                 | qwen2.5-7b   |  25 |            0.4   |        0.667 |     0.691 |                                       |
| gender                   | gemini       |  20 |            0.2   |        0.25  |     0.267 |                                       |
| gender                   | llama-3.1-8b |  25 |            0.4   |        0.667 |     0.124 |                                       |
| gender                   | mistral-7b   |  25 |            0.6   |        1.5   |     0.691 |                                       |
| gender                   | qwen2.5-7b   |  25 |            0.48  |        0.923 |     0.929 |                                       |
| health                   | gemini       |  25 |            0.4   |        0.667 |     0.691 |                                       |
| health                   | llama-3.1-8b |  25 |            0.44  |        0.786 |     0.427 |                                       |
| health                   | mistral-7b   |  25 |            0.56  |        1.273 |     0.759 |                                       |
| health                   | qwen2.5-7b   |  25 |            0.44  |        0.786 |     0.759 |                                       |
| killer                   | gemini       |  25 |            0.4   |        0.667 |     0.691 |                                       |
| killer                   | llama-3.1-8b |  25 |            0.52  |        1.083 |     0.845 |                                       |
| killer                   | mistral-7b   |  25 |            0.64  |        1.778 |     0.552 |                                       |
| killer                   | qwen2.5-7b   |  25 |            0.4   |        0.667 |     0.643 |                                       |
| no_of_dependant          | gemini       |  25 |            0     |      nan     |   nan     | dati insufficienti o nessuna varianza |
| no_of_dependant          | llama-3.1-8b |  25 |            0.6   |        1.5   |     0.277 |                                       |
| no_of_dependant          | mistral-7b   |  25 |            0.96  |       24     |     0.002 |                                       |
| no_of_dependant          | qwen2.5-7b   |  25 |            0.12  |        0.136 |     0.009 |                                       |
| psychopath               | gemini       |  25 |            0.4   |        0.667 |     0.691 |                                       |
| psychopath               | llama-3.1-8b |  25 |            0.52  |        1.083 |     0.683 |                                       |
| psychopath               | mistral-7b   |  25 |            0.6   |        1.5   |     0.643 |                                       |
| psychopath               | qwen2.5-7b   |  25 |            0.44  |        0.786 |     0.797 |                                       |

## 5. Figure

![Distribuzione delle decisioni per modello](results/figures/decision_distribution.png)

*Distribuzione delle decisioni per modello*

![Framework etico dominante per modello](results/figures/framework_distribution.png)

*Framework etico dominante per modello*

![Bias per tratto e per modello](results/figures/bias_by_trait.png)

*Bias per tratto e per modello*

![Attenzione al tratto vs. decisione di sacrificio](results/figures/attention_vs_bias.png)

*Attenzione al tratto vs. decisione di sacrificio*

## 6. Limiti

- Il dataset di persone e' sintetico e i suoi valori (es. eta', professione) non riflettono distribuzioni demografiche reali.
- Gemini e' usato solo tramite API: non e' possibile accedere alla sua attenzione interna, quindi il confronto sull'attenzione riguarda solo i modelli locali.
- Una parte delle risposte puo' essere stata classificata con il fallback a keyword invece che con l'estrazione strutturata (vedi tassi riportati sopra); queste righe sono meno affidabili.
- Il numero di ripetizioni per scenario e' limitato dal tempo di calcolo disponibile su una singola GPU consumer.

## 7. Conclusioni

- I modelli variano enormemente nella propensione ad agire: **mistral-7b** sceglie di intervenire nel 92.8% dei casi, contro il solo 4.9% di **gemini** - una differenza che i test chi-quadrato confermano essere altamente significativa (vedi Sezione 3), quindi non attribuibile al caso.
- Bias significativi (p < 0.05) individuati su singoli tratti, modello per modello:
  - **mistral-7b** su tratto `no_of_dependant`: sacrifica piu' spesso la persona con il valore 'target' del tratto (tasso 96%, odds ratio 24.00, p=0.002).
  - **qwen2.5-7b** su tratto `no_of_dependant`: sacrifica meno spesso la persona con il valore 'target' del tratto (tasso 12%, odds ratio 0.14, p=0.009).
  - **llama-3.1-8b** su tratto `contribution_to_humanity`: sacrifica piu' spesso la persona con il valore 'target' del tratto (tasso 72%, odds ratio 2.57, p=0.017).
- Questi risultati vanno letti come un'analisi comportamentale su un dataset sintetico e un numero di ripetizioni limitato (vedi Sezione 6, Limiti): confermano che i modelli *differiscono* sistematicamente nel loro ragionamento etico, ma non vanno generalizzati oltre questo esperimento senza ulteriori repliche.
