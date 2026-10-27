import os
import sys
import time

import numpy as np
import pytest

from sarge.affordance import AffordanceEngine
from sarge.generators import remote as rg
from sarge.generators.mock import MockGenerator
from sarge.generators.routed import RoutedGenerator
from sarge.sidecar import SfxClient, serve
from sarge.sidecar import server as sv
from sarge.types import AudioClip, CacheKey

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
FIELDS = {"path", "route", "gen", "prompt", "take", "phys", "scene", "predicted"}
BALL = {"shape": "sphere", "ext": [0.6, 0.6, 0.6], "m": 1.0, "dyn": 1, "pm": "sarge_bouncy", "bnc": 1.0,
        "name": "AutoCaptureBall_0"}
METAL = {"shape": "box", "ext": [10, 0.2, 10], "dyn": 0, "tag": "metal", "name": "Surface_metal"}
PHYS = {"a": BALL, "b": METAL, "vn": 12.0, "vt": 0.3, "n": [0, 1, 0], "j": 12.0, "dist": 6.5}
SCENE = {"src": "npc", "who": "Orc_Brute", "obj": "Club_Wooden", "act": "use", "room": "open"}
STEP = CacheKey("footstep", "gravel", "normal")


def _mock(**kw):
    eng = AffordanceEngine(MockGenerator(), realtime=True, vram_budget_kb=200_000, authored=set())
    return serve(engine=eng, host="127.0.0.1", port=0, warmup=False, **kw)


def _stop(srv):
    srv.shutdown()
    srv.server_close()


class Fake:
    def __init__(self, level):
        self.level = level
        self.prompts = {}
        self.sample_rate = 48000

    def generate(self, key, *, variant=0, prompt=None):
        return AudioClip(np.full(8, self.level, dtype=np.float32), 48000, variant_id=variant)

    def generate_batch(self, items):
        return [self.generate(k, variant=v) for k, v in items]


class FakeRemote:
    def generate(self, key, *, variant=0, prompt=None, engine=None):
        return AudioClip(np.full(8, 0.7, dtype=np.float32), 48000, variant_id=variant)

    def control(self, obj, reply=True):
        return None if not reply else {"type": "upgrade", "state": "ready"}


def _routed(monkeypatch, upgrade="flow"):
    monkeypatch.setattr(sv, "UPGRADE_DELAY_S", 0.0)
    monkeypatch.setattr(sv, "UPGRADE_IDLE_S", 0.0)
    monkeypatch.setattr(sv, "UPGRADE_POLL_S", 0.0)
    gen = RoutedGenerator(guns=Fake(0.1), rest=Fake(0.2), upgrade=upgrade)
    gen._remote = lambda: FakeRemote()
    eng = AffordanceEngine(gen, realtime=True, vram_budget_kb=200_000, authored=set())
    srv = serve(engine=eng, host="127.0.0.1", port=0, warmup=False, upgrade=upgrade)
    return srv, gen, eng


def test_a_first_hit_renders_a_repeat_comes_from_the_cache_and_a_held_take_is_meta_only():
    srv = _mock()
    try:
        with SfxClient(*srv.server_address, trace=True) as c:
            m0, s0 = c.event("footstep", "wood")
            m1, s1 = c.event("footstep", "wood")
            m2, s2 = c.event("footstep", "wood", have=[m1["variant_id"]])
    finally:
        _stop(srv)
    assert set(m0["trace"]) == FIELDS and m0["served"] == "realtime"
    assert m0["trace"]["path"] == "render" and m1["trace"]["path"] == "cache" and m2["trace"]["path"] == "meta"
    t = m0["trace"]
    assert t["route"] == "" and t["gen"] == "mock" and t["prompt"] == "" and t["take"] == ""
    assert t["phys"] is False and t["scene"] is False and t["predicted"] is False
    assert np.array_equal(s0, s1) and len(s2) == 0


def _session(srv, trace):
    with SfxClient(*srv.server_address, trace=trace) as c:
        rows = [c.event("footstep", "wood"), c.event("footstep", "wood"), c.event("shoot", "metal", "hard"),
                c.event("footstep", "wood", have=[0])]
        c.predict([{"source": "motion", "event_type": "impact", "material": "stone", "eta_s": 5.0,
                    "confidence": 1.0}])
        rows.append(c.event("impact", "stone"))
        return c.welcome, rows, c.stats(), c.stats(feed=0)


def test_only_a_client_that_asks_gets_the_trace_and_every_sample_stays_the_same():
    servers = {"asks": _mock(), "plain": _mock(), "off": _mock(trace=False)}
    try:
        got = {name: _session(srv, name != "plain") for name, srv in servers.items()}
    finally:
        for srv in servers.values():
            _stop(srv)
    (w_on, r_on, st_on, fd_on) = got["asks"]
    assert w_on["trace"] is True and "trace" not in got["plain"][0] and "trace" not in got["off"][0]
    assert {k: v for k, v in w_on.items() if k != "trace"} == got["plain"][0] == got["off"][0]
    for name in ("plain", "off"):
        _w, rows, st, fd = got[name]
        for (m_on, s_on), (m, s) in zip(r_on, rows):
            assert "trace" in m_on and "trace" not in m
            assert {k: v for k, v in m_on.items() if k != "trace"} == m
            assert s_on.dtype == s.dtype and np.array_equal(s_on, s)
        assert "feed" not in st and st == st_on
    assert "feed" not in got["off"][3] and got["plain"][3]["feed"] == [] and fd_on["feed_seq"] == 0
    assert {k: v for k, v in fd_on.items() if k not in ("feed", "feed_seq")} == got["off"][3]


def test_the_trace_is_the_same_on_every_run():
    runs = []
    for _ in range(2):
        srv = _mock()
        try:
            with SfxClient(*srv.server_address, trace=True) as c:
                c.predict([{"source": "motion", "event_type": "footstep", "material": "wood", "eta_s": 5.0,
                            "confidence": 1.0}])
                runs.append([c.event("footstep", "wood")[0], c.event("footstep", "wood")[0],
                             c.event("footstep", "wood", have=[0])[0]])
        finally:
            _stop(srv)
    assert runs[0] == runs[1] and [m["trace"]["path"] for m in runs[0]] == ["render", "cache", "meta"]


def test_predicted_follows_a_registered_prediction_for_that_exact_key():
    srv = _mock()
    try:
        with SfxClient(*srv.server_address, trace=True) as c:
            m0, _ = c.event("footstep", "wood")
            ack = c.predict([{"source": "motion", "event_type": "impact", "material": "stone", "eta_s": 5.0,
                              "confidence": 1.0}])
            m1, _ = c.event("impact", "stone")
            m2, _ = c.event("splash", "water")
    finally:
        _stop(srv)
    assert ack["queued"] == 1
    assert m0["trace"]["predicted"] is False and m1["trace"]["predicted"] is True
    assert m2["trace"]["predicted"] is False


def test_world_and_scene_records_show_in_the_trace():
    srv = _mock(world=True, scene=True)
    try:
        with SfxClient(*srv.server_address, trace=True) as c:
            m0, _ = c.event("impact", "metal", "hard", phys=PHYS, scene=SCENE)
            m1, _ = c.event("impact", "metal", "hard")
    finally:
        _stop(srv)
    assert m0["trace"]["phys"] is True and m0["trace"]["scene"] is True and m0["phys_key"] and m0["scene_key"]
    assert m1["trace"]["phys"] is False and m1["trace"]["scene"] is False


def test_routed_trace_names_the_route_the_model_and_the_prompt_source(monkeypatch):
    srv, gen, eng = _routed(monkeypatch, upgrade=None)
    gen.rest.model = "woosh-dflow"
    try:
        with SfxClient(*srv.server_address, trace=True) as c:
            gun, _ = c.event("shoot", "pistol")
            step, _ = c.event("footstep", "gravel")
            told, _ = c.event("door", "wood", prompt="an old door creaks open")
            srv.writer_owned = {str(CacheKey("splash", "water", "normal"))}
            gen.prompts[str(CacheKey("splash", "water", "normal"))] = "a written splash"
            wrote, _ = c.event("splash", "water")
    finally:
        _stop(srv)
    assert (gun["trace"]["route"], gun["trace"]["gen"]) == ("guns", "sa3")
    assert (step["trace"]["route"], step["trace"]["gen"]) == ("rest", "woosh_dflow")
    assert step["trace"]["prompt"] == "template" and told["trace"]["prompt"] == "game"
    assert wrote["trace"]["prompt"] == "writer"


def test_a_flow_upgrade_marks_the_served_take_and_lands_in_the_feed(monkeypatch):
    srv, gen, eng = _routed(monkeypatch)
    try:
        with SfxClient(*srv.server_address, trace=True) as c:
            first = c.stats(feed=0)
            m0, s0 = c.event("footstep", "gravel")
            assert sv._upgrade_step(srv)
            m1, s1 = c.event("footstep", "gravel")
            feed = c.stats(feed=0)
            again = c.stats(feed=feed["feed_seq"])
            replay = c.stats(feed=feed["feed_seq"] + 50)
    finally:
        _stop(srv)
    assert first["feed"] == [] and first["feed_seq"] == 0
    assert m0["trace"]["path"] == "render" and m0["trace"]["take"] == "" and m0["trace"]["gen"] == "woosh"
    assert m1["trace"]["path"] == "cache" and m1["trace"]["take"] == "flow" and m1["trace"]["gen"] == "woosh_flow"
    assert s1[0] == pytest.approx(0.7) and s0[0] == pytest.approx(0.2)
    kinds = [(f["kind"], f["key"], f["gen"]) for f in feed["feed"]]
    assert kinds == [("flow_take", "footstep|gravel", "woosh_flow"), ("flow_swap", "footstep|gravel", "woosh_flow")]
    assert feed["feed"][1]["takes"] == 1 and [f["seq"] for f in feed["feed"]] == [1, 2]
    assert again["feed"] == [] and again["feed_seq"] == 2 and len(replay["feed"]) == 2


def test_background_renders_reach_the_feed():
    eng = AffordanceEngine(MockGenerator(), realtime=True, vram_budget_kb=200_000, authored=set())
    srv = serve(engine=eng, host="127.0.0.1", port=0, warmup=False, deepen=True)
    try:
        with SfxClient(*srv.server_address, trace=True) as c:
            c.predict([{"source": "motion", "event_type": "footstep", "material": "stone", "eta_s": 5.0,
                        "confidence": 1.0}])
            deadline = time.monotonic() + 10.0
            feed = c.stats(feed=0)
            while not feed["feed"] and time.monotonic() < deadline:
                time.sleep(0.05)
                feed = c.stats(feed=0)
            m, _ = c.event("footstep", "stone")
    finally:
        srv.deepening = False
        _stop(srv)
    rec = feed["feed"][0]
    assert rec["kind"] == "render" and rec["key"] == "footstep|stone" and rec["why"] == "demand_gen"
    assert rec["gen"] == "mock" and rec["variant"] == 0 and rec["phys"] is False
    assert m["trace"]["path"] == "cache" and m["trace"]["predicted"] is True


def test_a_written_prompt_and_a_meaning_flip_reach_the_feed(monkeypatch):
    srv, gen, eng = _routed(monkeypatch, upgrade=None)
    srv.writer_owned = set()
    srv.prompt_rev = {}
    try:
        sv._writer_written(srv, STEP, {"woosh": "gravel crunching underfoot", "sa3": "x", "ms": 812.34})
        gun = CacheKey("rack", "shotgun", "normal")
        gen.route_mode = "meaning"
        gen.made[gun] = "rest"
        gen.meaning = lambda key: True
        assert sv._route_step(srv)
        with SfxClient(*srv.server_address, trace=True) as c:
            feed = c.stats(feed=0)["feed"]
    finally:
        _stop(srv)
    assert [(f["kind"], f["key"]) for f in feed] == [("writer", "footstep|gravel"), ("route_flip", "rack|shotgun")]
    assert feed[0]["prompt"] == "writer" and feed[0]["ms"] == pytest.approx(812.3) and feed[0]["route"] == "rest"
    assert feed[1]["side"] == "guns" and feed[1]["route"] == "guns" and feed[1]["gen"] == "sa3"


def test_a_bad_feed_value_is_a_non_fatal_bad_fields():
    from sarge.sidecar import ProtocolError
    from sarge.sidecar.wire import MsgType
    srv = _mock()
    try:
        with SfxClient(*srv.server_address, trace=True) as c:
            with pytest.raises(ProtocolError) as e:
                c._round_trip(MsgType.STATS, {"type": "stats", "feed": "soon"})
            assert e.value.code == "bad_fields" and not e.value.fatal
            assert c.ping(1.0)["t"] == 1.0
    finally:
        _stop(srv)


def test_the_trace_never_breaks_a_reply(monkeypatch):
    srv = _mock()
    monkeypatch.setattr(sv, "_trace_key", lambda *a, **k: 1 / 0)
    try:
        with SfxClient(*srv.server_address, trace=True) as c:
            m, s = c.event("footstep", "wood")
    finally:
        _stop(srv)
    assert m["trace"]["path"] == "render" and m["trace"]["gen"] == "" and len(s) > 0


def test_the_woosh_hello_names_the_model(monkeypatch):
    class Sock:
        def setsockopt(self, *a):
            pass

    r = rg.RemoteGenerator(port=1)
    assert r.model == ""
    monkeypatch.setattr(rg.socket, "create_connection", lambda *a, **k: Sock())
    monkeypatch.setattr(r, "_roundtrip", lambda obj, pcm=b"": ({"type": "hello", "sample_rate": 48000,
                                                                "model": "woosh-dflow"}, b""))
    r._connect()
    assert r.model == "woosh-dflow" and r.sample_rate == 48000


@pytest.fixture
def rs(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import run_sidecar
    return run_sidecar


def test_the_sidecar_traces_by_default_and_no_trace_turns_it_off(rs, monkeypatch):
    seen = []

    class Stop(Exception):
        pass

    def fake_serve(**kw):
        seen.append(kw["trace"])
        raise Stop()

    monkeypatch.setattr(rs, "serve", fake_serve)
    for extra in ([], ["--no-trace"]):
        with pytest.raises(Stop):
            rs.main(["--generator", "mock", "--no-cache"] + extra)
    assert seen == [True, False]
