"""Prompt condivisi da tutti i provider (locali e Gemini), in due stadi:
stage 1 = ragionamento libero (usato anche per l'analisi dell'attenzione),
stage 2 = estrazione strutturata della decisione, sullo stesso turno di conversazione.
"""

STAGE1_SYSTEM_PROMPT = (
    "Sei un esperto di filosofia morale. Ti verrà proposto un dilemma etico ispirato al "
    "trolley problem. Analizza il caso considerando sia argomenti utilitaristi (le "
    "conseguenze delle azioni) sia argomenti deontologici (i doveri morali e i diritti "
    "individuali), poi prendi una posizione netta. Rispondi in italiano, in modo "
    "articolato ma non eccessivamente lungo (massimo circa 300 parole)."
)

STAGE2_EXTRACTION_PROMPT = (
    "Sulla base SOLO del ragionamento che hai appena esposto, rispondi ESCLUSIVAMENTE con "
    "due righe, in questo formato esatto e senza aggiungere altro testo:\n"
    "DECISIONE: AZIONE oppure DECISIONE: INAZIONE\n"
    "FRAMEWORK: UTILITARISTA oppure FRAMEWORK: DEONTOLOGICO oppure FRAMEWORK: MISTO\n\n"
    "Dove \"AZIONE\" significa intervenire attivamente (es. tirare la leva) e "
    "\"INAZIONE\" significa non intervenire."
)


def build_stage1_messages(scenario_description: str) -> list:
    return [
        {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
        {"role": "user", "content": scenario_description},
    ]


def build_stage2_messages(scenario_description: str, stage1_response: str) -> list:
    return [
        {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
        {"role": "user", "content": scenario_description},
        {"role": "assistant", "content": stage1_response},
        {"role": "user", "content": STAGE2_EXTRACTION_PROMPT},
    ]
