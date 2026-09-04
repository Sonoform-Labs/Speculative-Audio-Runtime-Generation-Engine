from __future__ import annotations

import json
import os

from .types import GameEvent


def load_events(path: str | os.PathLike) -> list[GameEvent]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [GameEvent.from_dict(d) for d in data["events"]]


def save_events(path: str | os.PathLike, events: list[GameEvent], meta: dict | None = None) -> None:
    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    os.makedirs(parent, exist_ok=True)
    payload = {"meta": meta or {}, "events": [e.to_dict() for e in events]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
