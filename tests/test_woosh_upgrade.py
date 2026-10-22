import io
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import woosh_server as ws
from sarge.affordance import AffordanceEngine
from sarge.generators import remote as rg
from sarge.generators.routed import RoutedGenerator, UpgradeFailed
from sarge.sidecar import server as sv
from sarge.types import AudioClip, CacheKey

STEP = CacheKey("footstep", "gravel", "normal")
GUN = CacheKey("shoot", "pistol", "normal")
WOOD = CacheKey("footstep", "wood", "normal")
CAP = "template:caption+tt"
HEAD_EVENT_FIELDS = ["type", "t_recv", "key", "served", "variant", "concurrent", "await_ms", "lock_wait_ms",
                     "engine_ms", "gen_ms", "encode_ms", "send_ms", "bytes", "meta_only", "client"]


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
        self.raise_for = {}

    def route(self, key):
        return "guns" if key.event_type == "shoot" else "rest"

    def upgrade_control(self, kind):
        self.calls.append(("control", kind))
        return {"state": self.state}

    def upgrade_take(self, key, variant, prompt=None):
        self.calls.append(("take", str(key), variant, prompt))
        if str(key) in self.raise_for:
            raise self.raise_for[str(key)]
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
    sv._upgrade_init(s)
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


def _logged(s):
    s.event_log = io.StringIO()
    s.event_log_lock = threading.Lock()


def _records(s, kind):
    return [r for r in map(json.loads, s.event_log.getvalue().splitlines()) if r["type"] == kind]


def _takes(g, key):
    return [c for c in g.calls if c[0] == "take" and c[1] == str(key)]


def _pool(*variants):
    return [AudioClip(np.zeros(8, np.float32), 48000, variant_id=v) for v in variants]


def test_a_key_that_always_fails_is_retried_with_backoff_then_skipped_and_later_keys_flow(monkeypatch, capsys):
    s, g, eng = _server(monkeypatch)
    _logged(s)
    s.upgrade_state = "ready"
    monkeypatch.setattr(sv, "UPGRADE_BACKOFF_S", 100.0)
    eng.cache[WOOD] = _pool(0, 1)
    g.raise_for[str(STEP)] = UpgradeFailed("model server: ValueError: no graph")
    assert sv._upgrade_step(s)
    rec = s.upgrade_fail[STEP]
    assert rec["n"] == 1 and not rec["dead"] and rec["until"] > time.monotonic() + 50
    assert sv._upgrade_step(s) and sv._upgrade_step(s)
    assert s.prompt_rev[WOOD] == 1 and len(_takes(g, STEP)) == 1 and len(_takes(g, WOOD)) == 2
    assert not sv._upgrade_step(s)
    for n in (2, 3):
        s.upgrade_fail[STEP]["until"] = 0.0
        assert sv._upgrade_step(s) and s.upgrade_fail[STEP]["n"] == n
    assert s.upgrade_fail[STEP]["dead"] and len(_takes(g, STEP)) == sv.UPGRADE_RETRIES + 1
    s.upgrade_fail[STEP]["until"] = 0.0
    assert not sv._upgrade_step(s) and not sv._upgrade_step(s) and len(_takes(g, STEP)) == 3
    errors, failed = _records(s, "upgrade_error"), _records(s, "upgrade_failed")
    assert [e["attempt"] for e in errors] == [1, 2] and [e["retry_in_s"] for e in errors] == [100.0, 200.0]
    assert len(failed) == 1 and failed[0]["attempts"] == 3 and "no graph" in failed[0]["error"]
    assert capsys.readouterr().out.count("skipped for the session") == 1
    summ = sv._upgrade_summary(s)
    assert summ["failed_keys"] == 1 and summ["skipped_keys"] == 1 and summ["errors"] == 3
    assert summ["queued_takes"] == 0 and summ["swapped_takes"] == 2 and summ["rendered"] == 2
    assert STEP not in s.prompt_rev and [c.variant_id for c in eng.cache[STEP]] == [0, 1]


def test_an_unreachable_worker_backs_off_for_every_key_and_marks_none_failed(monkeypatch):
    s, g, eng = _server(monkeypatch)
    s.upgrade_state = "ready"
    g.raise_for[str(STEP)] = UpgradeFailed("ConnectionRefusedError: refused", unreachable=True)
    t = time.monotonic()
    assert not sv._upgrade_step(s)
    assert s.upgrade_block >= t + sv.UPGRADE_BACKOFF_S and not s.upgrade_fail and s.upgrade_count["unreachable"] == 1
    assert not sv._upgrade_step(s) and len(_takes(g, STEP)) == 1
    s.upgrade_block = 0.0
    t = time.monotonic()
    assert not sv._upgrade_step(s) and s.upgrade_block >= t + 2 * sv.UPGRADE_BACKOFF_S
    s.upgrade_unreach = 30
    s.upgrade_block = 0.0
    t = time.monotonic()
    assert not sv._upgrade_step(s) and s.upgrade_block <= t + sv.UPGRADE_UNREACH_MAX_S + 1.0
    del g.raise_for[str(STEP)]
    s.upgrade_block = 0.0
    assert sv._upgrade_step(s) and s.upgrade_unreach == 0 and not s.upgrade_fail


def test_a_new_prompt_gives_a_skipped_key_a_fresh_start(monkeypatch):
    s, g, eng = _server(monkeypatch)
    s.upgrade_state = "ready"
    monkeypatch.setattr(sv, "UPGRADE_BACKOFF_S", 0.0)
    g.raise_for[str(STEP)] = RuntimeError("boom")
    for _ in range(sv.UPGRADE_RETRIES + 1):
        assert sv._upgrade_step(s)
    assert s.upgrade_fail[STEP]["dead"] and not sv._upgrade_step(s)
    del g.raise_for[str(STEP)]
    g.prompts[str(STEP)] = "a written prompt"
    assert sv._upgrade_step(s) and STEP not in s.upgrade_fail
    assert _takes(g, STEP)[-1] == ("take", str(STEP), 0, "a written prompt")


def test_a_success_clears_an_earlier_failure_of_the_key(monkeypatch):
    s, g, eng = _server(monkeypatch)
    s.upgrade_state = "ready"
    monkeypatch.setattr(sv, "UPGRADE_BACKOFF_S", 0.0)
    g.raise_for[str(STEP)] = RuntimeError("once")
    assert sv._upgrade_step(s) and s.upgrade_fail[STEP]["n"] == 1
    del g.raise_for[str(STEP)]
    assert sv._upgrade_step(s) and STEP not in s.upgrade_fail


def test_routed_upgrade_raises_a_key_failure_and_an_unreachable_worker_apart():
    log = []
    gen = RoutedGenerator(guns=Fake("guns"), rest=Fake("rest"), upgrade="flow")
    gen._remote = lambda: FakeRemote(log, fail="ValueError: no graph for 777 frames")
    with pytest.raises(UpgradeFailed) as e:
        gen.upgrade_take(STEP, 0)
    assert not e.value.unreachable and "no graph" in str(e.value) and gen.upgrade_stats["failed"] == 1
    assert not gen._up_busy.is_set() and gen._up is not None

    class Gone(FakeRemote):
        def generate(self, *a, **k):
            raise ConnectionRefusedError("refused")

    gen._up = Gone(log)
    with pytest.raises(UpgradeFailed) as e:
        gen.upgrade_take(STEP, 1)
    assert e.value.unreachable and gen._up is None and gen.upgrade_stats["failed"] == 2
    assert gen.upgrade_stats["preempted"] == 0 and not gen._up_busy.is_set()


def test_worker_upgrade_renders_with_the_style_it_is_given_and_counts_failures():
    seen = []

    def gen(prompt, seed, **kw):
        seen.append(prompt)
        return np.full((4800, 1), 0.5, dtype=np.float32)

    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: None, commit=lambda: None)
    up.gen = gen
    for style, prompt in ((CAP, None), ("tuned", None), (CAP, "a game prompt")):
        assert up.render(STEP, 0, style, prompt, None, False, "peak").dtype == np.float32
    assert seen == [ws.prompt_for(STEP, CAP), ws.prompt_for(STEP, "tuned"), "a game prompt"]
    assert seen[0] != seen[1] and up.renders == 3

    def bad(prompt, seed, **kw):
        raise ValueError("no graph")

    def stop(prompt, seed, **kw):
        raise ws.Preempted()

    up.gen = bad
    with pytest.raises(ValueError):
        up.render(STEP, 0, CAP, None, None, False, "peak")
    up.gen = stop
    with pytest.raises(ws.Preempted):
        up.render(STEP, 0, CAP, None, None, False, "peak")
    assert up.info()["failed"] == 1 and up.info()["preempted"] == 1 and up.renders == 3


def test_the_bench_fail_hook_fails_only_the_named_key_and_counts_it(monkeypatch):
    seen = []

    def gen(prompt, seed, **kw):
        seen.append(prompt)
        return np.full((4800, 1), 0.5, dtype=np.float32)

    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: None, commit=lambda: None)
    up.gen = gen
    other = WOOD
    monkeypatch.delenv(ws.FAIL_ENV, raising=False)
    up.render(STEP, 0, CAP, None, None, False, "peak")
    monkeypatch.setenv(ws.FAIL_ENV, "%s|%s" % (STEP.event_type, STEP.material))
    with pytest.raises(RuntimeError) as e:
        up.render(STEP, 1, CAP, None, None, False, "peak")
    assert ws.FAIL_ENV in str(e.value) and "Preempted" not in str(e.value)
    up.render(other, 0, CAP, None, None, False, "peak")
    assert len(seen) == 2 and up.renders == 2 and up.info()["failed"] == 1 and up.info()["preempted"] == 0


def test_worker_upgrade_style_defaults_to_the_served_style():
    assert ws.upgrade_style(types.SimpleNamespace(prompts=CAP, upgrade_prompts=None)) == CAP
    assert ws.upgrade_style(types.SimpleNamespace(prompts="tuned", upgrade_prompts=CAP)) == CAP
    assert ws.upgrade_style(types.SimpleNamespace(prompts="tuned")) == "tuned"


def _rest_cmd(rs, monkeypatch, argv):
    got = []

    class P:
        stdout = iter(())

        def poll(self):
            return 0

        def kill(self):
            return None

    def popen(cmd, **kw):
        got.append(cmd)
        return P()

    monkeypatch.setattr(rs.subprocess, "Popen", popen)
    kw = _main_kwargs(rs, monkeypatch, ["--woosh-python", sys.executable] + argv)
    with pytest.raises(SystemExit):
        kw["gen_kwargs"]["start_rest"]()
    return kw["gen_kwargs"], got[0]


def test_first_prompt_defaults_to_same_and_leaves_the_worker_and_the_generator_as_before(rs, monkeypatch):
    off_kw, off = _rest_cmd(rs, monkeypatch, ["--prompts", CAP])
    assert off[-2:] == ["--prompts", CAP] and "--upgrade-prompts" not in off and "first_prompt" not in off_kw
    up_kw, up = _rest_cmd(rs, monkeypatch, ["--prompts", CAP, "--woosh-upgrade", "flow"])
    same_kw, same = _rest_cmd(rs, monkeypatch, ["--prompts", CAP, "--woosh-upgrade", "flow", "--first-prompt", CAP])
    assert up[-4:] == ["--prompts", CAP, "--upgrade", "flow"] and same == up
    assert "first_prompt" not in up_kw and "first_prompt" not in same_kw and up_kw["prompt_style"] == CAP
    tuned_kw, tuned = _rest_cmd(rs, monkeypatch, [])
    assert "--prompts" not in tuned and "--upgrade-prompts" not in tuned and tuned_kw["prompt_style"] == "detailed"


def test_first_prompt_tuned_sends_the_phrasebook_to_first_hits_and_the_template_to_flow_and_guns(rs, monkeypatch):
    kw, cmd = _rest_cmd(rs, monkeypatch, ["--prompts", CAP, "--woosh-upgrade", "flow", "--first-prompt", "tuned"])
    assert "--prompts" not in cmd and cmd[-4:] == ["--upgrade", "flow", "--upgrade-prompts", CAP]
    assert kw["first_prompt"] == "tuned" and kw["prompt_style"] == CAP and kw["upgrade"] == "flow"
    kw, cmd = _rest_cmd(rs, monkeypatch, ["--woosh-upgrade", "flow", "--first-prompt", CAP])
    assert cmd[-6:] == ["--prompts", CAP, "--upgrade", "flow", "--upgrade-prompts", "tuned"]
    assert kw["first_prompt"] == CAP and kw["prompt_style"] == "detailed"
    with pytest.raises(SystemExit):
        _main_kwargs(rs, monkeypatch, ["--prompts", CAP, "--first-prompt", "tuned"])
    with pytest.raises(SystemExit):
        _main_kwargs(rs, monkeypatch, ["--woosh-upgrade", "flow", "--first-prompt", "template:nope"])


def test_upgrade_solver_defaults_to_dopri5_and_reaches_the_worker_only_when_set(rs, monkeypatch):
    _kw, up = _rest_cmd(rs, monkeypatch, ["--prompts", CAP, "--woosh-upgrade", "flow"])
    assert "--upgrade-solver" not in up
    _kw, fast = _rest_cmd(rs, monkeypatch, ["--prompts", CAP, "--woosh-upgrade", "flow", "--woosh-upgrade-solver", "rk4:8"])
    assert fast[-2:] == ["--upgrade-solver", "rk4:8"] and fast[:-2] == up
    _kw, same = _rest_cmd(rs, monkeypatch, ["--prompts", CAP, "--woosh-upgrade", "flow", "--woosh-upgrade-solver", "dopri5"])
    assert same == up
    with pytest.raises(SystemExit):
        _main_kwargs(rs, monkeypatch, ["--woosh-upgrade-solver", "rk4:8"])
    with pytest.raises(SystemExit):
        _main_kwargs(rs, monkeypatch, ["--woosh-upgrade", "flow", "--woosh-upgrade-solver", "euler:3"])
    assert set(rs.UPGRADE_SOLVERS) == set(ws.UPGRADE_SOLVERS)


def test_worker_upgrade_builds_flow_with_the_solver_it_is_given(monkeypatch):
    got = []

    def make_gen(kind, frames, steps, solver, **kw):
        got.append((kind, frames, steps, solver, kw.get("share")))
        return "g"

    monkeypatch.setattr(ws, "make_gen", make_gen)
    base = ws.Upgrade("base", "flow", {"other": 101}, threading.Lock(), {}, commit=lambda: None)
    assert base.make() == "g" and got[-1] == ("flow", 101, 4, "dopri5", "base") and base.info()["solver"] == "dopri5"
    for name, steps, method in (("rk4:8", 8, "rk4"), ("midpoint:16", 16, "midpoint"), ("rk4:16", 16, "rk4")):
        up = ws.Upgrade("base", "flow", {"other": 101}, threading.Lock(), {}, commit=lambda: None, solver=name)
        assert up.make() == "g" and got[-1] == ("flow", 101, steps, method, "base") and up.info()["solver"] == name


def test_first_prompt_tags_the_disk_cache_and_leaves_it_alone_when_unset(tmp_path):
    from sarge.generators.cached import CachedGenerator

    class Styled(Fake):
        prompt_style = CAP
        post_tag = ""

    base = RoutedGenerator(guns=Styled("g"), rest=Fake("r"))
    up = RoutedGenerator(guns=Styled("g"), rest=Fake("r"), upgrade="flow")
    first = RoutedGenerator(guns=Styled("g"), rest=Fake("r"), upgrade="flow", first_prompt="tuned")
    assert base.post_tag == up.post_tag == "|prompts:" + CAP
    assert first.post_tag == "|prompts:%s|first:tuned" % CAP
    paths = [CachedGenerator(g, str(tmp_path))._path(k, 0) for g in (base, up, first) for k in (STEP, GUN)]
    assert paths[:2] == paths[2:4] and not set(paths[4:]) & set(paths[:4])


def test_swap_records_the_time_from_first_hit_and_counts_swaps_before_a_hit(monkeypatch, capsys):
    s, g, eng = _server(monkeypatch)
    _logged(s)
    s.upgrade_state = "ready"
    eng.cache[WOOD] = _pool(0)
    s.upgrade_first[STEP] = time.monotonic() - 3.0
    while sv._upgrade_step(s):
        pass
    swaps = _records(s, "upgrade_swap")
    assert [r["key"] for r in swaps] == [str(STEP), str(WOOD)] and all(r["first_swap"] for r in swaps)
    assert 2.9 < swaps[0]["first_hit_s"] < 60 and swaps[1]["first_hit_s"] is None
    assert [r["queued"] for r in swaps] == [1, 0] and [r["rendered"] for r in swaps] == [2, 3]
    assert [r["swapped"] for r in swaps] == [2, 3] and all(r["failed"] == 0 for r in swaps)
    assert len(swaps[0]["take_ms"]) == 2 and len(swaps[1]["take_ms"]) == 1
    summ = sv._upgrade_summary(s)
    assert summ["first_hit_to_swap_s"]["keys"] == 1 and summ["swapped_before_hit"] == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 2 and out[0].startswith("upgrade flow: queued 1 takes (1 keys), staged 0, rendered 2")
    assert out[1].startswith("upgrade flow: queued 0 takes (0 keys), staged 0, rendered 3, swapped 3 takes in 2")
    assert "first hit to swap median 3.0 s" in out[1] and "—" not in out[1]


def test_a_game_prompt_for_a_swapped_key_bumps_its_revision_with_the_upgrade_on(monkeypatch):
    s, g, eng = _server(monkeypatch)
    s.upgrade_state = "ready"
    assert sv._upgrade_step(s) and sv._upgrade_step(s) and s.prompt_rev[STEP] == 1
    sv._register_prompt(s, STEP, "a game prompt")
    assert STEP not in eng.cache and s.prompt_rev[STEP] == 2
    assert STEP not in s.upgrade_done and STEP not in s.upgrade_rev
    eng.cache[STEP] = _pool(0)
    sv._upgrade_sync(s, STEP)
    assert s.prompt_rev[STEP] == 2
    assert sv._upgrade_step(s) and _takes(g, STEP)[-1] == ("take", str(STEP), 0, "a game prompt")
    assert s.prompt_rev[STEP] == 3


def test_a_dropped_or_replaced_pool_bumps_the_revision_once(monkeypatch):
    s, g, eng = _server(monkeypatch)
    s.upgrade_state = "ready"
    sv._upgrade_step(s)
    sv._upgrade_step(s)
    eng.cache.pop(STEP)
    assert not sv._upgrade_step(s) and s.prompt_rev[STEP] == 2 and STEP not in s.upgrade_done
    sv._upgrade_sync(s, STEP)
    assert s.prompt_rev[STEP] == 2
    eng.cache[STEP] = _pool(0, 1)
    sv._upgrade_step(s)
    sv._upgrade_step(s)
    assert s.prompt_rev[STEP] == 3
    eng.cache[STEP].append(_pool(2)[0])
    sv._upgrade_sync(s, STEP)
    assert s.prompt_rev[STEP] == 3 and STEP in s.upgrade_done
    eng.cache[STEP] = _pool(0)
    sv._bump_rev(s, STEP)
    sv._upgrade_sync(s, STEP)
    assert s.prompt_rev[STEP] == 4 and STEP not in s.upgrade_done and STEP not in s.upgrade_rev


def _live(monkeypatch, upgrade="flow", **kw):
    monkeypatch.setattr(sv, "UPGRADE_DELAY_S", 0.0)
    monkeypatch.setattr(sv, "UPGRADE_IDLE_S", 0.0)
    monkeypatch.setattr(sv, "UPGRADE_POLL_S", 0.0)
    log = []
    gen = RoutedGenerator(guns=Fake("guns"), rest=Fake("rest"), upgrade=upgrade)
    gen._remote = lambda: FakeRemote(log)
    eng = AffordanceEngine(gen, realtime=True, vram_budget_kb=200_000, authored=set())
    srv = sv.serve(engine=eng, host="127.0.0.1", port=0, warmup=False, upgrade=upgrade, **kw)
    return srv, gen, eng, log


def test_a_client_resyncs_after_a_swap_and_after_the_swapped_takes_are_evicted(monkeypatch, tmp_path):
    from sarge.sidecar.client import connect
    path = tmp_path / "ev.jsonl"
    srv, gen, eng, log = _live(monkeypatch, event_log=str(path))
    try:
        c = connect(port=srv.server_address[1])
        try:
            m0, s0 = c.event("footstep", "gravel")
            assert m0["served"] == "realtime" and "prompt_rev" not in m0 and s0[0] == pytest.approx(0.1)
            assert sv._upgrade_step(srv) and srv.prompt_rev[STEP] == 1
            assert [x for x in log if x[0] == "gen"] == [("gen", str(STEP), 0, None, "flow")]
            m1, s1 = c.event("footstep", "gravel", have=[0])
            assert m1.get("meta_only") is not True and m1["prompt_rev"] == 1 and s1[0] == pytest.approx(0.7)
            m2, _ = c.event("footstep", "gravel", have=[0], have_rev=1)
            assert m2.get("meta_only") is True
            with sv.engine_lock:
                eng.cache.pop(STEP)
            m3, s3 = c.event("footstep", "gravel", have=[0], have_rev=1)
            assert m3["served"] == "realtime" and m3.get("meta_only") is not True
            assert m3["prompt_rev"] == 2 and s3[0] == pytest.approx(0.1)
            m4, _ = c.event("footstep", "gravel", have=[0], have_rev=2)
            assert m4.get("meta_only") is True and m4["prompt_rev"] == 2
            st = c.stats()["upgrade"]
        finally:
            c.close()
    finally:
        srv.shutdown()
        srv.server_close()
    assert st["rendered"] == 1 and st["swapped_takes"] == 1 and st["swaps"] == 1 and st["queued_takes"] == 1
    assert st["first_hit_to_swap_s"]["keys"] == 1 and st["swapped_before_hit"] == 0 and st["state"] == "ready"
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events = [r for r in recs if r["type"] == "event"]
    assert [r["upgraded"] for r in events] == [False, True, True, False, False]
    assert [r["rev"] for r in events] == [0, 1, 1, 2, 2]
    swap = [r for r in recs if r["type"] == "upgrade_swap"][0]
    assert swap["first_swap"] and swap["first_hit_s"] >= 0 and swap["rendered"] == 1 and swap["queued"] == 0


def test_flags_off_events_stats_and_logs_carry_no_upgrade_fields(monkeypatch, tmp_path):
    from sarge.sidecar.client import connect
    path = tmp_path / "ev.jsonl"
    srv, gen, eng, log = _live(monkeypatch, upgrade=None, event_log=str(path))
    try:
        c = connect(port=srv.server_address[1])
        try:
            m0, _ = c.event("footstep", "gravel")
            m1, _ = c.event("footstep", "gravel", have=[0])
            st = c.stats()
        finally:
            c.close()
        sv._register_prompt(srv, STEP, "a game prompt")
    finally:
        srv.shutdown()
        srv.server_close()
    assert srv.prompt_rev is None and "prompt_rev" not in m0 and "prompt_rev" not in m1 and m1["meta_only"]
    assert "upgrade" not in st and log == [] and gen._up is None
    ev = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [list(r) for r in ev] == [HEAD_EVENT_FIELDS] * 2
