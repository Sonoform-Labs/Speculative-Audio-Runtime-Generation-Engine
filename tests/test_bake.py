import numpy as np
import pytest

from predictive_sfx.affordance import AffordanceEngine, GameContext
from predictive_sfx.generators import MockGenerator
from predictive_sfx.generators.base import Generator
from predictive_sfx.generators.cached import CachedGenerator
from predictive_sfx.sidecar import serve, SfxClient
from predictive_sfx.types import AudioClip, CacheKey


class _PromptStub(Generator):
    name = "promptstub"

    def __init__(self):
        self.prompts = {}
        self.sample_rate = 48000
        self.latency_ms = 0.0
        self.seen = []

    def generate(self, key, *, variant=0):
        self.seen.append(self.prompts.get(str(key), "default"))
        return AudioClip(np.zeros(100, dtype=np.float32), 48000, variant)


def test_generate_batch_returns_all():
    g = MockGenerator()
    items = [(CacheKey("footstep", "grass", "normal"), v) for v in range(4)]
    clips = g.generate_batch(items)
    assert len(clips) == 4
    assert all(isinstance(c, AudioClip) for c in clips)


def test_audioclip_wav_roundtrip(tmp_path):
    a = AudioClip(np.linspace(-1.0, 1.0, 2000, dtype=np.float32), 44100)
    p = tmp_path / "c.wav"
    a.to_wav(p)
    b = AudioClip.from_wav(p)
    assert b.sample_rate == 44100
    assert b.samples.shape == a.samples.shape
    assert np.max(np.abs(b.samples - a.samples)) <= 1.0 / 32767 + 1e-6


def test_cached_generator_persists_and_loads(tmp_path):
    cg = CachedGenerator(MockGenerator(), str(tmp_path))
    k = CacheKey("footstep", "wood", "normal")
    c1 = cg.generate(k, variant=0)
    assert len(list(tmp_path.glob("*.wav"))) == 1
    c2 = cg.generate(k, variant=0)
    assert c2.variant_id == 0
    assert np.max(np.abs(c1.samples - c2.samples)) <= 1.0 / 32767 + 1e-6


def test_cached_batch_mixes_hits_and_misses(tmp_path):
    cg = CachedGenerator(MockGenerator(), str(tmp_path))
    k = CacheKey("footstep", "stone", "normal")
    cg.generate(k, variant=0)
    clips = cg.generate_batch([(k, 0), (k, 1), (k, 2)])
    assert len(clips) == 3
    assert len(list(tmp_path.glob("*.wav"))) == 3


def test_prewarm_pins_and_fills():
    e = AffordanceEngine(MockGenerator(), variants_per_key=2, target_variants=6)
    keys = [CacheKey("footstep", "grass", "normal"), CacheKey("footstep", "metal", "normal")]
    e.prewarm(keys)
    for k in keys:
        assert k in e.pinned
        assert len(e.cache[k]) == 2
        assert e.want[k] == 6


def test_deepen_step_fills_to_target_then_stops():
    e = AffordanceEngine(MockGenerator(), variants_per_key=2, target_variants=5)
    k = CacheKey("footstep", "grass", "normal")
    e.prewarm([k])
    assert len(e.cache[k]) == 2
    guard = 0
    while e.deepen_step():
        guard += 1
        assert guard < 50
    assert len(e.cache[k]) == 5
    assert e.deepen_step() is False


def test_context_change_warms_all_topk_surfaces():
    e = AffordanceEngine(MockGenerator(), variants_per_key=2, target_variants=6, continuous_topk=4)
    e.on_context_change(GameContext(0.0, None, ["grass", "metal", "wood"], "arena"), gpu_pressure=0.3)
    for mat in ("grass", "metal", "wood"):
        assert len(e.cache[CacheKey("footstep", mat, "normal")]) == 2


def test_prewarm_prompt_override_reaches_generator():
    g = _PromptStub()
    e = AffordanceEngine(g, variants_per_key=1, target_variants=1)
    k = CacheKey("footstep", "stone", "normal")
    e.prewarm([k], prompts={str(k): "knight armor on cold stone, metallic clank"})
    assert g.prompts[str(k)].startswith("knight armor")
    assert any("knight armor" in p for p in g.seen)


def test_cached_path_differs_by_prompt(tmp_path):
    inner = _PromptStub()
    cg = CachedGenerator(inner, str(tmp_path))
    k = CacheKey("footstep", "stone", "normal")
    p0 = cg._path(k, 0)
    inner.prompts[str(k)] = "knight armor"
    assert cg._path(k, 0) != p0


def test_deepen_terminates_under_vram_pressure():
    e = AffordanceEngine(MockGenerator(), vram_budget_kb=110, variants_per_key=1, target_variants=6)
    e.prewarm([CacheKey("footstep", "grass", "normal"), CacheKey("footstep", "metal", "normal")])
    calls = 0
    while e.deepen_step():
        calls += 1
        assert calls < 100
    assert e.deepen_step() is False


def test_prewarmed_key_survives_context_change():
    e = AffordanceEngine(MockGenerator(), variants_per_key=2, target_variants=4)
    k = CacheKey("shoot", "metal", "normal")
    e.prewarm([k])
    assert k in e.cache
    e.on_context_change(GameContext(0.0, None, ["grass"], "field"))
    assert k in e.cache
    assert k in e.pinned
    assert e.on_event(k) == "cached"


def test_sidecar_prewarm_manifest_caches():
    srv = serve(generator="mock", host="127.0.0.1", port=0,
                vram_budget_kb=50_000, gpu_ms_budget=10_000_000, variants_per_key=2, target_variants=4)
    try:
        with SfxClient(*srv.server_address) as c:
            ack = c.prewarm([{"event_type": "footstep", "material": "slime"},
                             {"event_type": "footstep", "material": "snow"}])
            assert ack["cached_keys"] >= 2
            meta, _ = c.event("footstep", "slime", "normal")
            assert meta["served"] == "cached"
    finally:
        srv.shutdown()
        srv.server_close()
