import os
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import woosh_server as ws
from sarge.affordance import AffordanceEngine
from sarge.generators import remote as rg
from sarge.generators.routed import RoutedGenerator
from sarge.sidecar import server as sv
from sarge.types import AudioClip, CacheKey

STEP = CacheKey("footstep", "gravel", "normal")
GUN = CacheKey("shoot", "pistol", "normal")


class Fake:
    def __init__(self, tag):
        self.tag = tag
        self.calls = []
        self.sample_rate = 48000

    def generate(self, key, *, variant=0, prompt=None):
        self.calls.append((str(key), variant, prompt))
        return AudioClip(np.full(8, 0.1, dtype=np.float32), 48000, variant_id=variant)

    def generate_batch(self, items):
        return [self.generate(k, variant=v) for k, v in items]


class FakeRemote:
    def __init__(self, log, fail=None):
        self.log, self.fail, self.prompts = log, fail, {}

    def generate(self, key, *, variant=0, prompt=None, engine=None):
        self.log.append(("gen", str(key), variant, prompt, engine))
        if self.fail:
            raise RuntimeError("model server: %s" % self.fail)
        return AudioClip(np.full(8, 0.7, dtype=np.float32), 48000, variant_id=variant)

    def control(self, obj, reply=True):
        self.log.append(("control", obj["type"], reply))
        return None if not reply else {"type": "upgrade", "state": "ready"}


def test_remote_request_carries_engine_only_when_asked(monkeypatch):
    sent = []
    r = rg.RemoteGenerator(port=1)
    r._sock = object()
    monkeypatch.setattr(r, "_roundtrip", lambda obj, pcm=b"": (sent.append(dict(obj)) or (
        {"type": "audio", "sample_rate": 48000, "channels": 1}, np.zeros(4, "<f4").tobytes())))
    r.generate(STEP, variant=3)
    r.generate(STEP, variant=3, engine="flow")
    assert "engine" not in sent[0] and sent[1]["engine"] == "flow"
    assert {k: v for k, v in sent[1].items() if k != "engine"} == sent[0]


def test_routed_without_upgrade_never_opens_a_second_connection():
    g, r = Fake("guns"), Fake("rest")
    gen = RoutedGenerator(guns=g, rest=r)
    assert gen.upgrade is None and gen.upgrade_take(STEP, 0) is None
    assert gen.upgrade_control("upgrade_state") == {"state": "off"}
    gen.generate(GUN, variant=0)
    gen.generate(STEP, variant=1)
    assert gen._up is None and gen._ctl is None and "upgrade_renders" not in gen.fast_stats()


def test_routed_upgrade_renders_rest_on_its_own_connection_and_skips_guns():
    log = []
    gen = RoutedGenerator(guns=Fake("guns"), rest=Fake("rest"), upgrade="flow")
    gen._remote = lambda: FakeRemote(log)
    assert gen.upgrade_take(GUN, 0) is None and not log
    clip = gen.upgrade_take(STEP, 2, "a step")
    assert clip.variant_id == 2 and log == [("gen", str(STEP), 2, "a step", "flow")]
    assert gen.rest.calls == [] and gen.upgrade_stats["renders"] == 1
    assert gen.upgrade_control("upgrade_load")["state"] == "ready"


def test_routed_counts_a_preempted_upgrade_and_returns_nothing():
    log = []
    gen = RoutedGenerator(guns=Fake("guns"), rest=Fake("rest"), upgrade="flow")
    gen._remote = lambda: FakeRemote(log, fail="Preempted: ")
    assert gen.upgrade_take(STEP, 0) is None
    assert gen.upgrade_stats["preempted"] == 1 and not gen._up_busy.is_set()


def test_a_gun_render_aborts_an_upgrade_in_flight_without_waiting_for_a_reply():
    log = []
    gen = RoutedGenerator(guns=Fake("guns"), rest=Fake("rest"), upgrade="flow")
    gen._remote = lambda: FakeRemote(log)
    gen.generate(GUN, variant=0)
    assert log == []
    gen._up_busy.set()
    gen.generate(GUN, variant=1)
    gen.generate_batch([(GUN, 2)])
    gen.generate(STEP, variant=0)
    assert log == [("control", "upgrade_abort", False)] * 2 and gen.upgrade_stats["aborts_sent"] == 2


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_worker_upgrade_yields_to_served_renders_and_to_an_abort():
    c = Clock()
    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: None, clock=c)
    assert not up.stop()
    up.enter()
    assert up.stop()
    up.leave()
    assert not up.stop()
    up.abort()
    assert up.stop()
    c.t += ws.ABORT_HOLD_S + 0.01
    assert not up.stop()


class Ticking:
    def __init__(self, step=0.1):
        self.t, self.step = 100.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def test_worker_upgrade_loads_once_after_a_quiet_moment_and_captures_without_the_solver(monkeypatch):
    monkeypatch.setattr(ws, "RETRY_S", 0.0)
    clock = Ticking()
    seen = []

    class G:
        should_stop = None

        def capture(self, frames):
            seen.append((dict(frames), clock.t - up.last_request, self.should_stop))
            return {}

    made = []
    gen = G()
    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: made.append(1) or gen, clock=clock,
                    commit=lambda: None)
    up.enter()
    up.leave()
    assert up.info()["state"] == "idle"
    assert up.start(wait=True) == "ready" and up.gen is gen and gen.should_stop == up.stop
    assert up.start(wait=True) == "ready" and made == [1] and len(seen) == 1
    assert seen[0][0] == {"other": 101} and seen[0][1] >= ws.LOAD_IDLE_S and seen[0][2] is None
    assert up.info()["load_s"] is not None


def test_worker_upgrade_reports_a_failed_load(monkeypatch):
    def boom():
        raise RuntimeError("no weights")

    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=boom, commit=lambda: None)
    assert up.start(wait=True) == "failed" and "no weights" in up.error
    with pytest.raises(RuntimeError):
        up.render(STEP, 0, "tuned", None, "full", True, "peak")


def test_worker_upgrade_refuses_to_load_when_commit_is_short():
    made = []
    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: made.append(1),
                    commit=lambda: ws.LOAD_COMMIT_GB - 0.1)
    assert up.start(wait=True) == "failed" and "commit" in up.error and not made


def test_stream_safetensors_copies_each_tensor_into_its_parameter(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    a, b = torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7, torch.tensor([1.5, -2.25])
    st.save_file({"x.w": a, "x.b": b}, str(tmp_path / "w.safetensors"))
    own = {"x.w": torch.zeros(2, 3, dtype=torch.bfloat16), "x.b": torch.zeros(2)}
    assert ws.stream_safetensors(str(tmp_path / "w.safetensors"), own)
    assert torch.equal(own["x.w"], a.to(torch.bfloat16)) and torch.equal(own["x.b"], b)
    assert not ws.stream_safetensors(str(tmp_path / "w.safetensors"), {"x.w": torch.zeros(3, 2)})


def test_share_parts_moves_the_served_parts_into_the_new_model():
    torch = pytest.importorskip("torch")

    class Part(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(2, 2)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.autoencoder = Part()
            self.conditioners = torch.nn.ModuleDict({"t": Part()})
            self.dit = Part()
            self._subcomponents = {"autoencoder": self.autoencoder, "conditioners.t": self.conditioners["t"]}

    a, b = Model(), Model()
    assert ws.share_parts(a, b) == ["autoencoder", "conditioners"]
    assert a.autoencoder is b.autoencoder and a.conditioners is b.conditioners and a.dit is not b.dit
    assert a._subcomponents["autoencoder"] is b.autoencoder
    assert a._subcomponents["conditioners.t"] is b.conditioners["t"]
    c = Model()
    c.autoencoder = torch.nn.Linear(3, 3)
    assert ws.share_parts(c, Model()) == ["conditioners"]


class FakeRouted:
    def __init__(self):
        self.prompts = {}
        self.calls = []
        self.state = "ready"
        self.fail = False

    def route(self, key):
        return "guns" if key.event_type == "shoot" else "rest"

    def upgrade_control(self, kind):
        self.calls.append(("control", kind))
        return {"state": self.state}

    def upgrade_take(self, key, variant, prompt=None):
        self.calls.append(("take", str(key), variant, prompt))
        if self.fail:
            return None
        return AudioClip(np.full(8, 0.9, dtype=np.float32), 48000, variant_id=variant)


def _server(monkeypatch):
    monkeypatch.setattr(sv, "UPGRADE_DELAY_S", 0.0)
    monkeypatch.setattr(sv, "UPGRADE_IDLE_S", 0.0)
    monkeypatch.setattr(sv, "UPGRADE_POLL_S", 0.0)
    g = FakeRouted()
    eng = AffordanceEngine(g)
    eng.cache[STEP] = [AudioClip(np.zeros(8, np.float32), 48000, variant_id=v) for v in (0, 1)]
    eng.cache[GUN] = [AudioClip(np.zeros(8, np.float32), 44100, variant_id=0)]
    eng.fired.add(STEP)
    s = types.SimpleNamespace(engine=eng, upgrade="flow", upgrade_state="idle", upgrade_t0=0.0, upgrade_polled=0.0,
                              upgrade_block=0.0, upgrade_done={}, upgrade_stage={}, prompt_rev={}, last_gen_t=0.0,
                              event_log=None)
    return s, g, eng


def test_upgrade_swaps_every_take_of_a_key_at_once_and_bumps_its_revision(monkeypatch):
    s, g, eng = _server(monkeypatch)
    old = list(eng.cache[STEP])
    g.state = "loading"
    assert not sv._upgrade_step(s) and s.upgrade_state == "loading" and g.calls == [("control", "upgrade_load")]
    g.state = "ready"
    assert sv._upgrade_step(s) and s.upgrade_state == "ready" and g.calls[1] == ("control", "upgrade_state")
    assert all(c is o for c, o in zip(eng.cache[STEP], old)) and s.prompt_rev.get(STEP) is None
    assert sv._upgrade_step(s)
    new = eng.cache[STEP]
    assert [c.variant_id for c in new] == [0, 1] and all(c is not o for c, o in zip(new, old))
    assert all(float(c.samples[0]) == pytest.approx(0.9) for c in new) and s.prompt_rev[STEP] == 1
    assert eng.stats["upgrades"] == 2 and eng.stats["upgrade_swaps"] == 1
    assert not sv._upgrade_step(s)
    assert [c for c in g.calls if c[0] == "take"] == [("take", str(STEP), 0, None), ("take", str(STEP), 1, None)]
    assert float(eng.cache[GUN][0].samples[0]) == 0.0


def test_upgrade_waits_for_engine_work_and_backs_off_after_a_yield(monkeypatch):
    s, g, eng = _server(monkeypatch)
    s.upgrade_state = "ready"
    eng.claims.add(STEP)
    assert not sv._upgrade_step(s) and not [c for c in g.calls if c[0] == "take"]
    eng.claims.clear()
    g.fail = True
    monkeypatch.setattr(sv, "UPGRADE_IDLE_S", 5.0)
    s.last_gen_t = -10.0
    assert not sv._upgrade_step(s) and s.upgrade_block > 0 and STEP not in s.prompt_rev
    assert not sv._upgrade_step(s) and len([c for c in g.calls if c[0] == "take"]) == 1


def test_a_prompt_change_drops_the_staged_flow_take(monkeypatch):
    s, g, eng = _server(monkeypatch)
    s.upgrade_state = "ready"
    assert sv._upgrade_step(s) and 0 in s.upgrade_stage[STEP]["clips"]
    g.prompts[str(STEP)] = "a written prompt"
    assert sv._upgrade_step(s)
    assert s.upgrade_stage[STEP]["prompt"] == "a written prompt" and STEP not in s.prompt_rev
    assert g.calls[-1] == ("take", str(STEP), 0, "a written prompt")


def test_a_worker_without_the_upgrade_marks_it_failed(monkeypatch):
    s, g, eng = _server(monkeypatch)
    g.state = "off"
    assert not sv._upgrade_step(s) and s.upgrade_state == "failed"
    assert not sv._upgrade_step(s) and g.calls == [("control", "upgrade_load")]


@pytest.fixture
def rs(monkeypatch):
    import run_sidecar
    monkeypatch.setenv("SARGE_WOOSH_EARLY", "0")
    monkeypatch.setattr(run_sidecar.prefetch, "start_hybrid", lambda *a, **k: None)
    monkeypatch.setattr(run_sidecar, "check_weights", lambda name: None)
    return run_sidecar


def _main_kwargs(rs, monkeypatch, argv):
    captured = {}

    class Stop(Exception):
        pass

    def fake_serve(**kw):
        captured.update(kw)
        raise Stop()

    monkeypatch.setattr(rs, "serve", fake_serve)
    with pytest.raises(Stop):
        rs.main(["--generator", "hybrid", "--no-cache"] + argv)
    return captured


def _worker_cmd(rs, monkeypatch, argv):
    got = {}

    class P:
        stdout = iter(())

        def poll(self):
            return 0

        def kill(self):
            return None

    def popen(cmd, **kw):
        got["cmd"] = cmd
        return P()

    monkeypatch.setattr(rs.subprocess, "Popen", popen)
    kw = _main_kwargs(rs, monkeypatch, argv)
    args = types.SimpleNamespace(woosh_python=sys.executable, woosh_model="dflow", woosh_port=8771, woosh_frames=None,
                                 amp="full", stereo=True, prompts="tuned", level=None, woosh_hf_cache="x",
                                 woosh_upgrade="flow" if "--woosh-upgrade" in argv else "off")
    rs.start_woosh(args, hold=True)
    return kw, got["cmd"]


def test_sidecar_upgrade_flag_defaults_off_and_reaches_the_worker_only_when_on(rs, monkeypatch):
    kw, cmd = _worker_cmd(rs, monkeypatch, [])
    assert "upgrade" not in kw and "upgrade" not in kw["gen_kwargs"] and "--upgrade" not in cmd
    kw, cmd = _worker_cmd(rs, monkeypatch, ["--woosh-upgrade", "flow"])
    assert kw["upgrade"] == "flow" and kw["gen_kwargs"]["upgrade"] == "flow"
    assert cmd[cmd.index("--upgrade") + 1] == "flow"
    with pytest.raises(SystemExit):
        _main_kwargs(rs, monkeypatch, ["--woosh-upgrade", "flow", "--woosh-model", "flow"])
    with pytest.raises(SystemExit):
        rs.main(["--generator", "sa3", "--no-cache", "--woosh-upgrade", "flow"])


def test_serve_keeps_revisions_off_unless_the_writer_or_the_upgrade_needs_them():
    from sarge.generators.mock import MockGenerator
    s = sv.serve(engine=AffordanceEngine(MockGenerator()), port=0, warmup=False)
    try:
        assert s.prompt_rev is None and s.upgrade is None
    finally:
        s.shutdown()
        s.server_close()
    s = sv.serve(engine=AffordanceEngine(MockGenerator()), port=0, warmup=False, upgrade="flow")
    try:
        assert s.prompt_rev == {} and s.upgrade == "flow" and s.upgrade_state == "idle"
    finally:
        s.shutdown()
        s.server_close()
