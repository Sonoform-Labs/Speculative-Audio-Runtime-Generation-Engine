from __future__ import annotations

from .affordance import AffordanceEngine, GameContext
from .affordance_learn import RuntimeAffordanceLearner
from .generators import MockGenerator
from .router import EventRouter
from .types import CacheKey

GAMES = {
    "fps": {
        "surfaces": ["concrete", "metal", "grass"],
        "events": [("footstep", "surface"), ("land", "surface"),
                   ("shoot", "metal"), ("reload", "metal"), ("explosion", "metal")],
    },
    "platformer": {
        "surfaces": ["grass", "wood", "stone", "ice"],
        "events": [("footstep", "surface"), ("jump", "surface"), ("land", "surface"),
                   ("coin", "metal"), ("hit", "wood")],
    },
    "racing": {
        "surfaces": ["asphalt", "dirt", "gravel"],
        "events": [("skid", "surface"), ("engine", "metal"),
                   ("crash", "metal"), ("horn", "metal")],
    },
    "explore": {
        "surfaces": ["sand", "snow", "water"],
        "events": [("footstep", "surface"), ("rustle", "surface"),
                   ("splash", "water"), ("pickup", "metal")],
    },
}


def _session(game, rounds):
    seq = []
    surfaces = game["surfaces"]
    for r in range(rounds):
        surf = surfaces[r % len(surfaces)]
        seq.append(("ctx", GameContext(float(r), None, [surf], "world")))
        for et, mode in game["events"]:
            mat = surf if mode == "surface" else mode
            seq.append(("ev", CacheKey(et, mat, "normal")))
    return seq


def run_game(game, use_learner, rounds=8):
    if use_learner:
        eng = AffordanceEngine(MockGenerator(), learner=RuntimeAffordanceLearner(),
                               router=EventRouter(), variants_per_key=2,
                               vram_budget_kb=500_000, gpu_ms_budget=1e9)
    else:
        eng = AffordanceEngine(MockGenerator(), variants_per_key=2,
                               vram_budget_kb=500_000, gpu_ms_budget=1e9)
    seq = _session(game, rounds)
    hits = total = warm_hits = warm_total = 0
    warm_start = len(game["surfaces"])  # first full surface cycle = warmup
    round_idx = -1
    for kind, payload in seq:
        if kind == "ctx":
            round_idx += 1
            eng.on_context_change(payload, now=payload.time_s)
        else:
            served = eng.on_event(payload)
            hit = served == "cached"
            hits += hit
            total += 1
            if round_idx >= warm_start:
                warm_hits += hit
                warm_total += 1
    return {
        "hit_rate": hits / total if total else 0.0,
        "warm_hit_rate": warm_hits / warm_total if warm_total else 0.0,
        "events": total,
    }


def compare(rounds=8):
    out = {}
    for name, game in GAMES.items():
        out[name] = {
            "learned": run_game(game, True, rounds),
            "authored": run_game(game, False, rounds),
        }
    return out
