"""Cache su disco (JSONL append-only) per rendere l'esperimento ripetibile e
interrompibile: se il run si interrompe o si aggiunge un modello, il lavoro gia'
completato non viene rifatto."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class ResponseCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / "responses.jsonl"
        self._entries: dict = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._entries[record["key"]] = record

    @staticmethod
    def make_key(model_label: str, scenario_id: str, repetition: int, rendered_prompt: str) -> str:
        payload = f"{model_label}|{scenario_id}|{repetition}|{rendered_prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        return self._entries.get(key)

    def put(self, key: str, record: dict) -> None:
        full_record = {"key": key, **record}
        self._entries[key] = full_record
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(full_record, ensure_ascii=False) + "\n")

    def all_records(self) -> list:
        return list(self._entries.values())
