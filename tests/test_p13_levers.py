import json
import os
import sys
import threading
import types

import numpy as np
import pytest

from sarge import prompts, verify
from sarge.affordance import AffordanceEngine
from sarge.generators.remote import RemoteGenerator
from sarge.generators.routed import RoutedGenerator, UpgradeFailed, side_for
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.sidecar import server as sv
from sarge.types import AudioClip, CacheKey

from test_generic_levers import (BENCH, CAP, FakeGen, FakeLLM, FakeVerifier, SCRIPTS, _canon_raw, _rest_cmd)

CLEAN = CAP + "+clean"
GUN = CacheKey("shoot", "pistol", "normal")
STEP = CacheKey("footstep", "gravel", "normal")
ODD = CacheKey("Zorbo_Krell_v2", "", "normal")


@pytest.fixture
def ws(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import woosh_server
    return woosh_server


@pytest.fixture
def gate(monkeypatch):
    if BENCH not in sys.path:
        monkeypatch.syspath_prepend(BENCH)
    import fill_gate
    import upgrade_gate
    return upgrade_gate, fill_gate


def test_drop_mod_removes_one_modifier_and_keeps_the_rest():
    assert prompts.drop_mod(CLEAN, prompts.CLEAN_MOD) == CAP
    assert prompts.drop_mod(CAP + "+loop+clean", prompts.CLEAN_MOD) == CAP + "+loop"
    assert prompts.drop_mod(CAP + "+clean+loop", prompts.CLEAN_MOD) == CAP + "+loop"
    assert prompts.drop_mod(CAP, prompts.CLEAN_MOD) == CAP and prompts.drop_mod("tuned", prompts.CLEAN_MOD) == "tuned"
    assert prompts.template_mods(prompts.drop_mod(CAP + "+loop+clean", prompts.LOOP_MOD)) == {"clean"}


def test_sa3_raw_words_keep_the_typed_name_on_sa3_only():
    raw = StableAudio3Generator(prompt_style=CLEAN, raw_words=True)
    cleaned = StableAudio3Generator(prompt_style=CLEAN)
    assert raw._prompt(ODD) == prompts.template(CAP, ODD.event_type, "", model="sa3")
    assert raw._prompt(ODD) == "TrackType: SFX, A single zorbo krell v 2 sound, recorded up close."
    assert cleaned._prompt(ODD) == "TrackType: SFX, A single zorbo krell sound, recorded up close."
    assert raw.words_style == CAP and cleaned.words_style == CLEAN and raw.prompt_style == CLEAN
    raw.prompts[str(ODD)] = "a game prompt"
    assert raw._prompt(ODD) == "a game prompt"
    plain = StableAudio3Generator(prompt_style=CAP, raw_words=True)
    assert plain._prompt(ODD) == StableAudio3Generator(prompt_style=CAP)._prompt(ODD)


class G:
    def __init__(self, style, raw=False):
        self.sample_rate = 44100
        self.prompts = {}
        self.prompt_style = style
        self.raw_words = raw
        self.words_style = prompts.drop_mod(style, prompts.CLEAN_MOD) if raw else style
        self.calls = []

    def generate(self, key, *, variant=0, prompt=None):
        self.calls.append((str(key), variant, prompt))
        return AudioClip(np.full(8, 0.1 + 0.2 * (variant >= verify.CAND_STRIDE), np.float32), 44100,
                         variant_id=variant)

    def generate_batch(self, items):
        return [self.generate(k, variant=v) for k, v in items]


def test_raw_words_tag_the_disk_cache_only_when_on():
    off = RoutedGenerator(guns=G(CLEAN), rest=RemoteGenerator())
    on = RoutedGenerator(guns=G(CLEAN, True), rest=RemoteGenerator())
    assert "sa3:raw" not in off.post_tag and on.post_tag == off.post_tag + "|sa3:raw"
    assert off.sa3_best == 1 and "sa3_upgrade_candidates" not in off.fast_stats()
    assert on.sa3_style() == CAP and off.sa3_style() == CLEAN


def test_sidecar_raw_words_flag_reaches_sa3_and_leaves_the_worker_on_clean(monkeypatch):
    off, cap = _rest_cmd(monkeypatch, ["--prompts", CLEAN])
    on, cap2 = _rest_cmd(monkeypatch, ["--prompts", CLEAN, "--sa3-raw-words"])
    assert on == off and on[-2:] == ["--prompts", CLEAN]
    assert "raw_words" not in cap["gen_kwargs"] and cap2["gen_kwargs"]["raw_words"] is True
    assert cap2["gen_kwargs"]["prompt_style"] == CLEAN
    import run_sidecar as rs
    with pytest.raises(SystemExit):
        rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", sys.executable, "--prompts", CAP,
                 "--sa3-raw-words"])


def test_floor_drops_a_pool_prompt_far_below_the_keys_own_prompt():
    assert verify.floor_drops([0.3, 0.1], 0.05) == [1] and verify.floor_drops([0.3, 0.26], 0.05) == []
    assert verify.choose_floored([0.2, 0.5], [0.3, 0.1], 0.05) == 0
    assert verify.choose_floored([0.2, 0.5], [0.3, 0.1], 0.25) == 1
    assert verify.choose_floored([0.2, 0.5], [0.3, 0.25], 0.05) == 1
    assert verify.choose_floored([0.5, 0.5], [0.3, 0.3], 0.0) == 0
    assert verify.choose_floored([0.2, 0.5, 0.4], [0.3, 0.1, 0.29], 0.05) == 2
    assert verify.choose_floored([0.7], [0.1], 0.0) == 0
    for m in ([0.1, 0.2], [0.3, 0.3, 0.1], [0.5]):
        assert verify.choose_floored(m, [0.0] * len(m), 0.0) == verify.choose(m)


class FloorVerifier(FakeVerifier):
    def __init__(self, score, low):
        super().__init__(score)
        self.low = low
        self.pairs = []

    def scores(self, y, sr, tembs):
        i = len(self.pairs)
        self.pairs.append(tembs)
        return [self.fn(i), self.low(i)]


def _floor_up(ws, v, floor):
    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: None, commit=lambda: None,
                    best=2, verifier=v, floor=floor)
    up._load_verifier()
    up.gen = FakeGen()
    return up


def test_pool_floor_scores_the_cleaned_name_too_and_drops_only_far_lower_prompts(ws):
    k = CacheKey("SFX_Zorbo_Krell_01", "", "normal")
    pool = ["A single zz hum sound, recorded up close."]
    texts = {"request": "zz hum sound", "event": "zz hum sound"}
    p = [0.2, 0.2, 0.5, 0.5]
    low = [0.3, 0.3, 0.1, 0.1]
    base = ws.prompt_for(k, CLEAN)
    for floor, want in ((0.05, 0), (0.25, 1)):
        v = FloorVerifier(lambda i: p[i] if i < 4 else 0.0, lambda i: low[i] if i < 4 else 0.0)
        up = _floor_up(ws, v, floor)
        up.render(k, 0, CLEAN, None, None, False, "peak", pool=pool, texts=texts)
        assert up.choice[(str(k), (base, pool[0]))] == want and not v.scored
        assert v.pairs[0] == ("emb:zz hum sound", "emb:zorbo krell sound")
        assert set(v.texts) == {"zz hum sound", "zorbo krell sound"}
        info = up.info()
        assert info["pool_floor"] == floor and info["pool_floor_drops"] == (1 if want == 0 else 0)
    plain = _floor_up(ws, FakeVerifier(lambda i: p[i] if i < 4 else 0.0), None)
    plain.render(k, 0, CLEAN, None, None, False, "peak", pool=pool, texts=texts)
    assert plain.choice[(str(k), (base, pool[0]))] == 1 and "pool_floor" not in plain.info()


def test_verifier_scores_share_one_audio_embedding_and_equal_score(ws):
    import torch
    v = ws.Verifier()
    v.sr = 32000
    e = torch.tensor([0.6, 0.8])
    calls = []

    class Tower:
        def get_audio_embedding_from_data(self, m, device=None, use_tensor=True):
            calls.append(1)
            return e.unsqueeze(0)

    v.tower = Tower()
    y = np.zeros((4800, 2), np.float32)
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    got = v.scores(y, 48000, (a, (a, b), b))
    assert len(calls) == 1 and got == [v.score(y, 48000, a), v.score(y, 48000, (a, b)), v.score(y, 48000, b)]
    assert got == pytest.approx([0.6, 0.6, 0.8])


def test_floor_flags_check_their_needs_and_reach_the_worker_only_when_set(ws, monkeypatch, tmp_path):
    for argv in (["--upgrade", "flow", "--upgrade-pool-floor", "0.05"],
                 ["--upgrade", "flow", "--upgrade-best", "4", "--upgrade-pool-floor", "-0.1"],
                 ["--upgrade", "flow", "--upgrade-best", "4", "--upgrade-pool-guard", "--upgrade-pool-floor", "0"]):
        monkeypatch.setattr(sys, "argv", ["woosh_server.py"] + argv)
        with pytest.raises(SystemExit):
            ws.main()
    from sarge import writer as wr

    class Llama(FakeLLM):
        def __init__(self, *a, **kw):
            super().__init__(lambda m: _canon_raw("zz hum"))

    monkeypatch.setattr(wr, "LlamaBackend", Llama)
    base = ["--prompts", CLEAN, "--woosh-upgrade", "flow", "--woosh-upgrade-best", "4", "--woosh-upgrade-pool",
            "canon", "--writer-store", str(tmp_path)]
    off, _ = _rest_cmd(monkeypatch, base)
    assert "--upgrade-pool-floor" not in off
    on, _ = _rest_cmd(monkeypatch, base + ["--woosh-upgrade-pool-floor", "0.05"])
    assert on[len(off):] == ["--upgrade-pool-floor", "0.05"] and on[:len(off)] == off
    import run_sidecar as rs
    for bad in (base[:-4] + ["--woosh-upgrade-pool-floor", "0.05"], base + ["--woosh-upgrade-pool-floor", "-1"],
                base + ["--woosh-upgrade-pool-guard", "--woosh-upgrade-pool-floor", "0.05"]):
        with pytest.raises(SystemExit):
            rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", sys.executable] + bad)


def test_worker_scores_sa3_takes_against_the_request_words_of_the_style_it_is_given(ws):
    v = FakeVerifier(lambda i: 0.1 * i)
    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: None, commit=lambda: None,
                    best=2, verifier=v)
    with pytest.raises(RuntimeError):
        up.score_takes(ODD, [np.zeros(8, np.float32)], 44100, CAP)
    up._load_verifier()
    with pytest.raises(RuntimeError):
        up.score_takes(ODD, [np.zeros(8, np.float32)], 44100, CAP)
    up.gen = FakeGen()
    got = up.score_takes(ODD, [np.zeros(8, np.float32)] * 3, 44100, CAP)
    assert got == pytest.approx([0.0, 0.1, 0.2]) and v.texts == [verify.key_text(ODD)]
    up.score_takes(ODD, [np.zeros(8, np.float32)], 44100, CLEAN)
    assert v.texts[-1] == verify.key_text(ODD, clean=True) == "zorbo krell sound"
    assert up.info()["verified_takes"] == 4 and up.gen.calls == []
    off = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: None, commit=lambda: None)
    with pytest.raises(RuntimeError):
        off.score_takes(ODD, [np.zeros(8, np.float32)], 44100, CAP)


def test_remote_verify_frames_the_takes_and_the_worker_reads_them_back(ws, monkeypatch):
    r = RemoteGenerator()
    sent = []
    reply = {"type": "scores", "scores": [0.5, 0.25]}
    monkeypatch.setattr(r, "_ensure", lambda: None)
    monkeypatch.setattr(r, "_roundtrip", lambda req, pcm=b"": sent.append((req, pcm)) or (reply, b""))
    a = AudioClip(np.arange(6, dtype=np.float32), 44100, variant_id=0)
    b = AudioClip(np.arange(8, dtype=np.float32).reshape(4, 2), 44100, variant_id=4096)
    assert r.verify(GUN, [a, b], CAP) == [0.5, 0.25]
    req, pcm = sent[-1]
    assert req == {"type": "verify", "event_type": "shoot", "material": "pistol", "intensity": "normal",
                   "context": "default", "sample_rate": 44100, "frames": [6, 4], "channels": [1, 2], "style": CAP}
    ys = ws.verify_clips(req, pcm)
    assert np.array_equal(ys[0], a.samples) and np.array_equal(ys[1], b.samples)
    reply = {"type": "error", "message": "the verifier is failed"}
    with pytest.raises(RuntimeError):
        r.verify(GUN, [a], CAP)
    assert ws.verify_reply(None, req, pcm)["type"] == "error"
    bad = ws.verify_reply(types.SimpleNamespace(score_takes=lambda *x: [1.0, 2.0]), dict(req, frames=[6, 5]), pcm)
    assert bad["type"] == "error"
    good = ws.verify_reply(types.SimpleNamespace(score_takes=lambda key, ys, sr, style: [len(ys), sr]), req, pcm)
    assert good["type"] == "scores" and good["scores"] == [2, 44100]


class FakeRemote:
    def __init__(self, log, score=None):
        self.log, self.prompts = log, {}
        self.score = score or (lambda c: float(c.samples[0]))

    def generate(self, key, *, variant=0, prompt=None, engine=None, **kw):
        self.log.append(("gen", str(key), variant, prompt, engine))
        return AudioClip(np.full(8, 0.7, dtype=np.float32), 48000, variant_id=variant)

    def control(self, obj, reply=True):
        self.log.append(("control", obj["type"], reply))
        return None if not reply else {"type": "upgrade", "state": "ready"}

    def verify(self, key, clips, style=None):
        self.log.append(("verify", str(key), [c.variant_id for c in clips], style))
        return [self.score(c) for c in clips]


def test_routed_renders_sa3_candidates_on_the_gun_model_and_asks_the_worker_to_score(monkeypatch):
    log = []
    g = RoutedGenerator(guns=G(CLEAN, True), rest=RemoteGenerator(), upgrade="flow", sa3_best=4)
    g._remote = lambda: FakeRemote(log)
    c = g.sa3_candidate(GUN, 4096)
    assert c.variant_id == 4096 and g.guns.calls == [(str(GUN), 4096, None)]
    assert g.sa3_verify(GUN, [c, c]) == pytest.approx([0.3, 0.3]) and log[-1] == ("verify", str(GUN), [4096, 4096],
                                                                                    CAP)
    st = g.fast_stats()
    assert st["sa3_upgrade_candidates"] == 1 and st["sa3_upgrade_scored"] == 2

    class Down(FakeRemote):
        def verify(self, *a, **kw):
            raise ConnectionError("gone")

    g._up = Down(log)
    with pytest.raises(UpgradeFailed) as e:
        g.sa3_verify(GUN, [c])
    assert e.value.unreachable and g._up is None


class FakeRouted:
    def __init__(self, best=4, score=None):
        self.prompts = {}
        self.calls = []
        self.state = "ready"
        self.sa3_best = best
        self.score = score or (lambda c: float(c.samples[0]))
        self.verify_error = None
        self.cand_error = None

    def route(self, key):
        return "guns" if key.event_type == "shoot" else "rest"

    def upgrade_control(self, kind):
        self.calls.append(("control", kind))
        return {"state": self.state}

    def upgrade_take(self, key, variant, prompt=None, **kw):
        self.calls.append(("take", str(key), variant, prompt))
        return AudioClip(np.full(8, 0.9, dtype=np.float32), 48000, variant_id=variant)

    def sa3_candidate(self, key, variant):
        self.calls.append(("cand", str(key), variant))
        if self.cand_error:
            raise self.cand_error
        return AudioClip(np.full(8, 0.1 * (variant // verify.CAND_STRIDE), dtype=np.float32), 44100,
                         variant_id=variant)

    def sa3_verify(self, key, clips):
        self.calls.append(("verify", str(key), [c.variant_id for c in clips]))
        if self.verify_error:
            raise self.verify_error
        return [self.score(c) for c in clips]


def _server(monkeypatch, g, guns=(0, 1), steps=(0,)):
    monkeypatch.setattr(sv, "UPGRADE_DELAY_S", 0.0)
    monkeypatch.setattr(sv, "UPGRADE_IDLE_S", 0.0)
    monkeypatch.setattr(sv, "UPGRADE_POLL_S", 0.0)
    eng = AffordanceEngine(g)
    eng.cache[GUN] = [AudioClip(np.zeros(8, np.float32), 44100, variant_id=v) for v in guns]
    if steps:
        eng.cache[STEP] = [AudioClip(np.zeros(8, np.float32), 48000, variant_id=v) for v in steps]
    eng.fired.add(GUN)
    s = types.SimpleNamespace(engine=eng, upgrade="flow", upgrade_state="ready", upgrade_t0=0.0, upgrade_polled=0.0,
                              upgrade_block=0.0, upgrade_done={}, upgrade_stage={}, prompt_rev={}, last_gen_t=0.0,
                              event_log=None)
    sv._upgrade_init(s)
    sv._sa3_init(s)
    return s, eng


def _kinds(g):
    return [c[0] for c in g.calls if c[0] != "control"]


def test_sa3_lane_renders_n_minus_one_candidates_a_take_scores_them_and_swaps_the_key_at_once(monkeypatch):
    g = FakeRouted(best=4)
    s, eng = _server(monkeypatch, g, steps=())
    old = list(eng.cache[GUN])
    for _ in range(4):
        assert sv._sa3_step(s)
    assert _kinds(g) == ["cand", "cand", "cand", "verify"] and [c[2] for c in g.calls if c[0] == "cand"] == [
        4096, 8192, 12288]
    assert g.calls[-1] == ("verify", str(GUN), [0, 4096, 8192, 12288])
    assert all(c is o for c, o in zip(eng.cache[GUN], old)) and GUN not in s.prompt_rev
    for _ in range(4):
        assert sv._sa3_step(s)
    new = eng.cache[GUN]
    assert [c.variant_id for c in new] == [0, 1] and all(float(c.samples[0]) == pytest.approx(0.3) for c in new)
    assert s.prompt_rev[GUN] == 1 and s.sa3_rev[GUN] == 1
    assert sorted(s.sa3_done[GUN]) == [0, 1] and all(s.sa3_done[GUN][c.variant_id] is c for c in new)
    assert s.sa3_count["swaps"] == 1 and s.sa3_count["swapped"] == 2 and s.sa3_count["candidates"] == 6
    assert not sv._sa3_step(s) and len(g.calls) == 8 + sum(1 for c in g.calls if c[0] == "control")
    summ = sv._upgrade_summary(s)["sa3"]
    assert summ["swapped_takes"] == 2 and summ["queued_takes"] == 0 and summ["best"] == 4
    assert "sa3 best of 4" in sv._upgrade_line(sv._upgrade_summary(s))
    assert sv._sa3_served(s, GUN, 0, new[0]) and not sv._sa3_served(s, GUN, 0, old[0])


def test_sa3_lane_keeps_a_served_take_the_verifier_prefers_without_a_new_revision(monkeypatch):
    g = FakeRouted(best=2, score=lambda c: 1.0 if c.variant_id < verify.CAND_STRIDE else 0.0)
    s, eng = _server(monkeypatch, g, guns=(0,), steps=())
    old = eng.cache[GUN][0]
    assert sv._sa3_step(s) and sv._sa3_step(s)
    assert eng.cache[GUN][0] is old and GUN not in s.prompt_rev and list(s.sa3_done[GUN]) == [0]
    assert s.sa3_done[GUN][0] is old
    assert s.sa3_count["kept"] == 1 and s.sa3_count["swaps"] == 0 and not sv._sa3_step(s)


def test_sa3_lane_waits_for_idle_and_the_worker_and_drops_work_on_a_prompt_change(monkeypatch):
    g = FakeRouted(best=3)
    s, eng = _server(monkeypatch, g, guns=(0,), steps=())
    eng.claims.add(GUN)
    assert not sv._sa3_step(s) and not _kinds(g)
    eng.claims.clear()
    s.upgrade_state = "idle"
    g.state = "loading"
    assert not sv._sa3_step(s) and not _kinds(g)
    s.upgrade_state = "ready"
    assert sv._sa3_step(s) and _kinds(g) == ["cand"]
    g.prompts[str(GUN)] = "a written prompt"
    assert sv._sa3_step(s) and s.sa3_stage[GUN]["prompt"] == "a written prompt"
    assert s.sa3_stage[GUN]["cands"][0] and len(s.sa3_stage[GUN]["cands"][0]) == 1
    sv._sa3_drop(s, GUN)
    assert GUN not in s.sa3_stage


def test_sa3_lane_backs_off_a_failing_key_and_stops_when_the_worker_does_not_score(monkeypatch, capsys):
    g = FakeRouted(best=2)
    s, eng = _server(monkeypatch, g, guns=(0,), steps=())
    g.cand_error = UpgradeFailed("RuntimeError: cuda says no")
    for _ in range(sv.UPGRADE_RETRIES + 1):
        s.sa3_fail.get(GUN, {}).update(until=0.0)
        assert sv._sa3_step(s)
    assert s.sa3_fail[GUN]["dead"] and s.sa3_count["failed_keys"] == 1 and not sv._sa3_step(s)
    g2 = FakeRouted(best=2)
    s2, eng2 = _server(monkeypatch, g2, guns=(0,), steps=())
    g2.verify_error = UpgradeFailed("RuntimeError: model server: RuntimeError: the verifier is failed")
    assert sv._sa3_step(s2) and sv._sa3_step(s2)
    assert s2.sa3_state == "failed" and "does not score" in capsys.readouterr().out and not sv._sa3_step(s2)
    g3 = FakeRouted(best=2)
    s3, _ = _server(monkeypatch, g3, guns=(0,), steps=())
    g3.verify_error = UpgradeFailed("ConnectionError: gone", unreachable=True)
    assert sv._sa3_step(s3) and not sv._sa3_step(s3) and s3.upgrade_block > 0 and s3.sa3_state == "on"


def test_lanes_alternate_and_the_woosh_lane_runs_alone_with_the_sa3_lane_off(monkeypatch):
    g = FakeRouted(best=2)
    s, eng = _server(monkeypatch, g, guns=(0,), steps=(0,))
    order = []
    for _ in range(3):
        assert sv._upgrade_any(s)
        order.append(_kinds(g)[-1])
    assert order == ["take", "cand", "verify"] and not sv._upgrade_any(s)
    calls = []
    t = types.SimpleNamespace(engine=eng, upgrade="flow")
    monkeypatch.setattr(sv, "_upgrade_step", lambda srv: calls.append(srv) or False)
    monkeypatch.setattr(sv, "_sa3_step", lambda srv: pytest.fail("the sa3 lane ran with the flag off"))
    assert not sv._upgrade_any(t) and calls == [t]
    assert not sv._sa3_on(t) and sv._sa3_drop(t, GUN) is None and not sv._sa3_served(t, GUN, 0, None)
    sv._sa3_sync(t, GUN)


def test_serve_starts_the_sa3_lane_only_with_best_of_two_or_more(monkeypatch):
    for best, want in ((1, False), (4, True)):
        gen = RoutedGenerator(guns=G(CAP), rest=G(CAP), upgrade="flow", warm_guns=False, sa3_best=best)
        srv = sv.serve(engine=AffordanceEngine(gen), port=0, warmup=False, upgrade="flow")
        try:
            assert sv._sa3_on(srv) is want and getattr(srv, "sa3_done", None) == ({} if want else None)
        finally:
            srv.shutdown()
            srv.server_close()
    gen = RoutedGenerator(guns=G(CAP), rest=G(CAP), warm_guns=False, sa3_best=4)
    srv = sv.serve(engine=AffordanceEngine(gen), port=0, warmup=False)
    try:
        assert not sv._sa3_on(srv)
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_client_gets_the_verified_sa3_take_after_the_swap(monkeypatch, tmp_path):
    from sarge.sidecar.client import connect
    monkeypatch.setattr(sv, "UPGRADE_DELAY_S", 0.0)
    monkeypatch.setattr(sv, "UPGRADE_IDLE_S", 0.0)
    monkeypatch.setattr(sv, "UPGRADE_POLL_S", 0.0)
    log = []
    gen = RoutedGenerator(guns=G(CAP), rest=G(CAP), upgrade="flow", warm_guns=False, sa3_best=2)
    gen._remote = lambda: FakeRemote(log)
    eng = AffordanceEngine(gen, realtime=True, vram_budget_kb=200_000, authored=set())
    path = tmp_path / "ev.jsonl"
    srv = sv.serve(engine=eng, host="127.0.0.1", port=0, warmup=False, upgrade="flow", event_log=str(path))
    try:
        c = connect(port=srv.server_address[1])
        try:
            m0, s0 = c.event("shoot", "pistol")
            assert m0["served"] == "realtime" and s0[0] == pytest.approx(0.1)
            assert sv._sa3_step(srv) and sv._sa3_step(srv) and srv.prompt_rev[GUN] == 1
            m1, s1 = c.event("shoot", "pistol", have=[0])
            assert m1["prompt_rev"] == 1 and s1[0] == pytest.approx(0.3) and m1["variant_id"] == 0
            st = c.stats()["upgrade"]
        finally:
            c.close()
    finally:
        srv.shutdown()
        srv.server_close()
    assert ("verify", str(GUN), [0, 4096], CAP) in log and st["sa3"]["swapped_takes"] == 1
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["upgraded"] for r in recs if r["type"] == "event"] == [False, True]
    swap = [r for r in recs if r["type"] == "sa3_swap"][0]
    assert swap["picks"] == {"0": 1} and swap["takes"] == 1 and swap["first_swap"]


def test_sidecar_sa3_best_flag_needs_the_flow_upgrade_and_its_verifier(monkeypatch):
    base = ["--prompts", CAP, "--woosh-upgrade", "flow", "--woosh-upgrade-best", "4"]
    off, cap = _rest_cmd(monkeypatch, base)
    on, cap2 = _rest_cmd(monkeypatch, base + ["--sa3-upgrade-best", "4"])
    assert on == off and "sa3_best" not in cap["gen_kwargs"] and cap2["gen_kwargs"]["sa3_best"] == 4
    import run_sidecar as rs
    for bad in (["--sa3-upgrade-best", "4"], ["--woosh-upgrade", "flow", "--sa3-upgrade-best", "4"],
                base + ["--sa3-upgrade-best", "0"]):
        with pytest.raises(SystemExit):
            rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", sys.executable] + bad)


def test_wide_fill_holds_twelve_gun_and_twenty_four_woosh_first_hits_on_their_sides(gate):
    ug, _fg = gate
    base = ug.fill_plan("base")
    assert base == [("rest", e, m, "normal") for e, m in ug.FILL_REST] + [
        ("guns", e, m, "normal") for e, m in ug.FILL_GUNS] + [("rest", e, m, "soft") for e, m in ug.FILL_REST[:5]]
    wide = ug.fill_plan("wide")
    guns = [x for x in wide if x[0] == "guns"]
    rest = [x for x in wide if x[0] == "rest"]
    assert len(guns) == 12 and len(rest) == 24 and len(set(wide)) == 36
    for side, e, m, i in wide:
        assert side_for(CacheKey(e, m, i), "families") == side, (e, m)
    assert [x[0] for x in wide[:6]] == ["rest", "rest", "guns", "rest", "rest", "guns"]


def _gate_json(out, tag, ms):
    rows = [{"pass": "fill", "side": side, "key": k, "ms": v, "served": "realtime"} for side, d in ms.items()
            for k, v in d.items()]
    with open(os.path.join(out, "gate_%s.json" % tag), "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh)


def test_fill_gate_interleaves_runs_and_fails_only_when_the_lower_bound_passes_the_bound(gate, tmp_path):
    _ug, fg = gate
    assert fg.schedule(["a", "b"], 4) == [(0, "a"), (0, "b"), (1, "b"), (1, "a"), (2, "a"), (2, "b"), (3, "b"),
                                         (3, "a")]
    rng = np.random.default_rng(0)
    guns = ["g%d" % i for i in range(12)]
    rest = ["w%d" % i for i in range(24)]
    for shift, verdict in ((3.0, "PASS"), (12.0, "FAIL")):
        out = tmp_path / ("s%d" % int(shift))
        out.mkdir()
        plan = {"name": "t", "arms": [["base", []], ["pick", ["--x"]]], "reps": 4, "base": "base", "schedule": []}
        for r, arm in fg.schedule(["base", "pick"], 4):
            tag = "t_%s_r%d" % (arm, r)
            add = shift if arm == "pick" else 0.0
            _gate_json(str(out), tag, {"guns": {k: 150.0 + add + rng.normal(0, 2) for k in guns},
                                       "rest": {k: 30.0 + rng.normal(0, 1) for k in rest}})
            plan["schedule"].append({"rep": r, "arm": arm, "tag": tag, "rc": 0})
        with open(out / "plan_t.json", "w", encoding="utf-8") as fh:
            json.dump(plan, fh)
        fg.main(["judge", "--name", "t", "--out", str(out)])
        res = json.loads((out / "fill_gate_t.json").read_text(encoding="utf-8"))
        g = res["sides"]["guns"]["arms"]["pick"]
        assert g["verdict"] == verdict and res["verdict"]["pick"] == verdict and res["enough_keys"]
        assert g["minus_base_ci"][0] <= g["minus_base_ms"] <= g["minus_base_ci"][1]
        assert res["sides"]["rest"]["arms"]["pick"]["verdict"] == "PASS" and res["sides"]["guns"]["n_keys"] == 12
    x = np.zeros((2, 2, 12))
    x[1] += 5.0
    edge = fg.judge_side(x, 0, 5.0)[1]
    assert edge["minus_base_ci"] == [5.0, 5.0] and edge["verdict"] == "PASS"
    x[1] += 0.01
    assert fg.judge_side(x, 0, 5.0)[1]["verdict"] == "FAIL"


def test_sa3_candidates_pause_the_cpu_writer_like_served_renders(monkeypatch):
    from sarge import writer as wr

    class Backend:
        def __init__(self):
            self.log = []

        def pause(self):
            self.log.append("pause")

        def resume(self):
            self.log.append("resume")

    g = FakeRouted(best=2)
    s, eng = _server(monkeypatch, g, guns=(0,), steps=())
    a, b = Backend(), Backend()
    eng.gen = wr.Yielding(types.SimpleNamespace(inner=wr.Yielding(g, b)), a)
    assert [x.backend for x in sv._yield_gates(s)] == [a, b]
    assert sv._sa3_step(s) and a.log == ["pause", "resume"] and b.log == ["pause", "resume"]
    assert sv._sa3_step(s) and a.log == ["pause", "resume"] and s.sa3_count["kept"] + s.sa3_count["swapped"] == 1
    eng.gen = g
    assert sv._yield_gates(s) == []
