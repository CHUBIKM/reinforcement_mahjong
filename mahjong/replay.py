from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Replay:
    events: List[Dict]

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.events, ensure_ascii=False, indent=indent)

    def save(self, path: str) -> None:
        Path(path).write_text(self.to_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "Replay":
        raw = Path(path).read_text(encoding="utf-8")
        return cls(events=json.loads(raw))
