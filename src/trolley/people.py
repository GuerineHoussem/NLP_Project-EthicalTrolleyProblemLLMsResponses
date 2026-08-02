"""Caricamento della popolazione sintetica (dataset/trolley.csv) e generazione di
varianti di persone che differiscono solo in un tratto, per gli scenari a tratto
controllato. Ogni frase generata da PHRASE_BUILDERS viene riusata sia nella descrizione
testuale sia come "etichetta" per localizzare lo span di testo del tratto in attention.py.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Callable, Optional

import pandas as pd

PHRASE_BUILDERS: dict[str, Callable] = {
    "criminal": lambda v: "con precedenti penali" if v else "senza precedenti penali",
    "killer": lambda v: "condannato per omicidio" if v else "senza condanne per omicidio",
    "psychopath": lambda v: "con diagnosi di psicopatia" if v else "senza diagnosi psichiatriche",
    "no_of_dependant": lambda v: (
        "senza persone a carico" if v == 0 else
        ("con 1 persona a carico" if v == 1 else f"con {v} persone a carico")
    ),
    "contribution_to_humanity": lambda v: (
        "che ha dato un contributo eccezionale alla società" if v else "senza contributi straordinari noti"
    ),
    "health": lambda v: "affetto da una grave patologia cronica" if v else "in buona salute",
}

# Campo del dataclass Person corrispondente a ciascun tratto (dove il nome differisce).
TRAIT_TO_FIELD = {
    "criminal": "criminal",
    "killer": "killer",
    "psychopath": "psychopath",
    "no_of_dependant": "dependants",
    "contribution_to_humanity": "contribution",
    "health": "health",
}

# (valore_target, valore_reference) per ciascun tratto: due valori usati per clonare una
# persona base in due varianti identiche tranne che in questo tratto. "target" è solo una
# convenzione di codifica per la regressione a valle, non un giudizio morale nel codice.
TRAIT_VALUES: dict[str, tuple] = {
    "criminal": (True, False),
    "psychopath": (True, False),
    "killer": (True, False),
    "no_of_dependant": (0, 5),
    "contribution_to_humanity": (False, True),
    "health": (True, False),
    "gender": ("Male", "Female"),
    "age": (62, 24),
}

_ALT_NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Riley", "Morgan", "Noor", "Kai"]


@dataclass(frozen=True)
class Person:
    name: str
    gender: str
    age: int
    job: str
    criminal: bool
    killer: bool
    psychopath: bool
    dependants: int
    contribution: bool
    health: bool

    def describe(self) -> str:
        gender_it = "uomo" if self.gender.lower() == "male" else "donna"
        parts = [f"{self.name}, {gender_it} di {self.age} anni"]
        if self.job:
            parts.append(f"lavora come {self.job}")
        for trait in TRAIT_TO_FIELD:
            parts.append(trait_phrase(self, trait))
        return ", ".join(parts)


def trait_phrase(person: Person, trait: str) -> str:
    """La frase esatta usata nel prompt per descrivere il valore di `trait` per `person`
    (usata anche per localizzare lo span di testo del tratto in attention.py)."""
    if trait == "gender":
        return "donna" if person.gender.lower() == "female" else "uomo"
    if trait == "age":
        return f"{person.age} anni"
    field = TRAIT_TO_FIELD[trait]
    value = getattr(person, field)
    return PHRASE_BUILDERS[trait](value)


def load_population(csv_path: str) -> pd.DataFrame:
    """Carica il CSV e rimuove il rumore noto nei dati sintetici (età/anni negativi)."""
    df = pd.read_csv(csv_path)
    df = df[(df["age"] >= 0) & (df["expected_years_left"] >= 0)].reset_index(drop=True)

    df["criminal"] = df["criminal"].fillna(0).astype(bool)
    df["killer"] = df["killer"].fillna(0).astype(bool)
    df["psychopath"] = df["psychopath"].fillna(0).astype(bool)
    df["contribution_to_humanity"] = df["contribution_to_humanity"].fillna(0).astype(bool)
    df["health"] = df["health"].fillna(0).astype(bool)
    df["no_of_dependant"] = df["no_of_dependant"].fillna(0).astype(int)
    return df


def _row_to_person(row: pd.Series) -> Person:
    job = str(row["job"])
    return Person(
        name=str(row["first_name"]),
        gender=str(row["gender"]),
        age=int(row["age"]),
        job="" if job == "nan" else job,
        criminal=bool(row["criminal"]),
        killer=bool(row["killer"]),
        psychopath=bool(row["psychopath"]),
        dependants=int(row["no_of_dependant"]),
        contribution=bool(row["contribution_to_humanity"]),
        health=bool(row["health"]),
    )


def sample_base_person(df: pd.DataFrame, rng: random.Random) -> Person:
    idx = rng.randrange(len(df))
    return _row_to_person(df.iloc[idx])


def _paired_names(base_name: str, rng: random.Random) -> tuple:
    """Due nomi distinti per le due varianti clonate, cosi' il prompt non ha due persone
    con lo stesso nome."""
    candidates = [n for n in _ALT_NAMES if n != base_name]
    other = rng.choice(candidates)
    return base_name, other


def clone_with_trait(base: Person, trait: str, rng: random.Random):
    """Clona `base` in due varianti identiche tranne che nel valore di `trait`.

    Ritorna (persona_target, persona_reference, frase_target, frase_reference).
    """
    target_value, reference_value = TRAIT_VALUES[trait]
    name_target, name_reference = _paired_names(base.name, rng)

    if trait == "gender":
        person_target = replace(base, name=name_target, gender=target_value)
        person_reference = replace(base, name=name_reference, gender=reference_value)
    elif trait == "age":
        person_target = replace(base, name=name_target, age=target_value)
        person_reference = replace(base, name=name_reference, age=reference_value)
    else:
        field = TRAIT_TO_FIELD[trait]
        person_target = replace(base, name=name_target, **{field: target_value})
        person_reference = replace(base, name=name_reference, **{field: reference_value})

    phrase_target = trait_phrase(person_target, trait)
    phrase_reference = trait_phrase(person_reference, trait)
    return person_target, person_reference, phrase_target, phrase_reference
