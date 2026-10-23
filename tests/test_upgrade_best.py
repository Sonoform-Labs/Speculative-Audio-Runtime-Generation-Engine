import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import woosh_server as ws
from sarge import verify
from sarge.prompts import template
from sarge.types import CacheKey, seed_for

STEP = CacheKey("footstep", "gravel", "normal")
GLASS = CacheKey("shatter", "", "hard", "default", "break;a=glass")
CAP = "template:caption+tt"


class FakeGen:
    def __init__(self):
        self.calls = []
        self.ldm = object()

    def __call__(self, prompt, seed, **kw):
        self.calls.append((prompt, seed))
        return np.full((4800 + 480 * (len(self.calls) - 1), 1), 0.5, dtype=np.float32)


class FakeVerifier:
    def __init__(self, scores, fail=None):
        self.scores = list(scores)
        self.fail = fail
        self.texts, self.scored, self.loaded = [], [], 0
        self.path = "fake"
        self.ms = []

    def load(self):
        self.loaded += 1
        if self.fail:
            raise RuntimeError(self.fail)
        return self

    def text(self, ldm, text):
        self.texts.append((ldm, text))
        return "emb:" + text

    def score(self, y, sr, temb):
        self.scored.append((len(y), sr, temb))
        return self.scores[len(self.scored) - 1]


def _up(best, verifier, gen=None):
    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {}, make=lambda: None, commit=lambda: None,
                    best=best, verifier=verifier)
    if verifier is not None:
        up._load_verifier()
    up.gen = gen or FakeGen()
    return up


def test_request_text_is_the_caption_template_without_its_frame():
    assert verify.key_text(STEP) == "gravel footstep sound"
    assert template("template:caption+tt", "footstep", "gravel") == "A single gravel footstep sound, recorded up close."
    assert verify.request_text("bark") == "bark sound"
    assert "recorded up close" not in verify.key_text(GLASS) and not verify.key_text(GLASS).startswith("A single")


def test_candidates_start_at_the_take_itself_and_never_share_a_seed():
    assert verify.candidate_variant(3, 0) == 3
    seeds = {seed_for(STEP, verify.candidate_variant(v, j)) for v in range(8) for j in range(8)}
    assert len(seeds) == 64
    assert verify.pick([(0.2, 0, "a"), (0.5, 1, "b"), (0.5, 2, "c")])[2] == "b"


def test_best_of_n_serves_the_candidate_the_verifier_scores_highest():
    v = FakeVerifier([0.1, 0.4, 0.3])
    up = _up(3, v)
    y = up.render(STEP, 2, CAP, None, None, False, "peak")
    seeds = [s for _p, s in up.gen.calls]
    assert seeds == [seed_for(STEP, verify.candidate_variant(2, j)) for j in range(3)]
    assert seeds[0] == seed_for(STEP, 2)
    want = ws.finish_key(np.full((4800 + 480, 1), 0.5, dtype=np.float32), STEP, None, False)
    assert y.shape == want.shape and np.array_equal(y, want)
    assert v.loaded == 1 and [t for _l, t in v.texts] == [verify.key_text(STEP)] * 3
    assert all(l is up.gen.ldm for l, _t in v.texts)
    assert all(p == ws.prompt_for(STEP, CAP) for p, _s in up.gen.calls)
    info = up.info()
    assert info["best"] == 3 and info["candidates"] == 3 and info["picked"] == [0, 1, 0] and up.renders == 1
    assert info["verifier"] == "ready"


def test_a_preempted_best_of_n_take_resumes_where_it_stopped():
    v = FakeVerifier([0.1, 0.2, 0.9, 0.3, 0.5, 0.1, 0.1, 0.1])
    gen = FakeGen()
    calls = {"n": 0}

    def flaky(prompt, seed, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ws.Preempted()
        return gen(prompt, seed, **kw)

    flaky.ldm = gen.ldm
    up = _up(4, v, gen=flaky)
    with pytest.raises(ws.Preempted):
        up.render(STEP, 0, CAP, None, None, False, "peak")
    assert up.preempted == 1 and len(v.scored) == 2 and len(up.pending) == 1
    up.render(STEP, 0, CAP, None, None, False, "peak")
    assert [s for _p, s in gen.calls] == [seed_for(STEP, verify.candidate_variant(0, j)) for j in range(4)]
    assert len(v.scored) == 4 and up.info()["picked"] == [0, 0, 1, 0] and not up.pending
    up.render(STEP, 0, "tuned", None, None, False, "peak")
    assert len(gen.calls) == 8


def test_best_one_and_a_failed_verifier_render_one_take_as_before():
    plain = _up(1, None)
    plain.render(STEP, 5, CAP, None, None, False, "peak")
    assert [s for _p, s in plain.gen.calls] == [seed_for(STEP, 5)]
    assert set(plain.info()) == {"type", "engine", "state", "error", "renders", "preempted", "failed", "load_s",
                                 "solver"}
    assert plain.verifier is None and plain.verifier_state == "off"
    bad = FakeVerifier([], fail="no weights")
    up = _up(4, bad)
    up.render(STEP, 5, CAP, None, None, False, "peak")
    assert [s for _p, s in up.gen.calls] == [seed_for(STEP, 5)] and not bad.scored
    assert up.info()["verifier"] == "failed" and "no weights" in up.info()["verifier_error"]


def test_the_verifier_waits_for_commit_and_the_flow_load_reports_it(monkeypatch, capsys):
    monkeypatch.setattr(ws, "RETRY_S", 0.0)
    v = FakeVerifier([])
    calls = iter([ws.LOAD_COMMIT_GB, ws.VERIFIER_COMMIT_GB - 0.1])
    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {},
                    make=lambda: types.SimpleNamespace(capture=lambda frames: {}, should_stop=None),
                    commit=lambda: next(calls), best=4, verifier=v)
    assert up.verifier_state == "idle"
    assert up.start(wait=True) == "ready"
    assert v.loaded == 0 and up.verifier_state == "failed" and "commit" in up.verifier_error
    assert "best of 4 by the verifier: failed" in capsys.readouterr().out
    seen = []

    class Watch(FakeVerifier):
        def load(self):
            seen.append((up.state, up.gen))
            return super().load()

    ok = Watch([])
    up = ws.Upgrade(None, "flow", {"other": 101}, threading.Lock(), {},
                    make=lambda: types.SimpleNamespace(capture=lambda frames: {}, should_stop=None),
                    commit=lambda: None, best=2, verifier=ok)
    assert up.start(wait=True) == "ready" and ok.loaded == 1 and up.info()["verifier"] == "ready"
    assert seen == [("loading", None)]


def test_the_bench_fail_hook_still_fails_a_best_of_n_key(monkeypatch):
    v = FakeVerifier([0.1, 0.2])
    up = _up(2, v)
    monkeypatch.setenv(ws.FAIL_ENV, "%s|%s" % (STEP.event_type, STEP.material))
    with pytest.raises(RuntimeError):
        up.render(STEP, 0, CAP, None, None, False, "peak")
    assert up.failed == 1 and not up.gen.calls and not up.pending


def _rest_cmd(monkeypatch, argv):
    import run_sidecar as rs
    monkeypatch.setenv("SARGE_WOOSH_EARLY", "0")
    monkeypatch.setattr(rs.prefetch, "start_hybrid", lambda *a, **k: None)
    monkeypatch.setattr(rs, "check_weights", lambda name: None)
    got, captured = [], {}

    class Stop(Exception):
        pass

    class P:
        stdout = iter(())

        def poll(self):
            return 0

        def kill(self):
            return None

    def fake_serve(**kw):
        captured.update(kw)
        raise Stop()

    monkeypatch.setattr(rs, "serve", fake_serve)
    monkeypatch.setattr(rs.subprocess, "Popen", lambda cmd, **kw: got.append(cmd) or P())
    with pytest.raises(Stop):
        rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", sys.executable] + argv)
    with pytest.raises(SystemExit):
        captured["gen_kwargs"]["start_rest"]()
    return got[0]


def test_worker_lines_after_listening_reach_the_log_only_with_best_of_n(capsys):
    import run_sidecar as rs
    capsys.readouterr()
    on, off = types.SimpleNamespace(woosh_upgrade_best=4), types.SimpleNamespace(woosh_upgrade_best=1)
    rs.late_line("woosh upgrade flow ready after 1.4 s, best of 4 by the verifier: ready\n", on)
    rs.late_line("some other worker line\n", on)
    rs.late_line("woosh upgrade flow ready after 1.4 s\n", off)
    rs.late_line("woosh upgrade flow ready after 1.4 s\n", None)
    assert capsys.readouterr().out == "[woosh] woosh upgrade flow ready after 1.4 s, best of 4 by the verifier: ready\n"


def test_sidecar_best_flag_defaults_off_and_reaches_the_worker_only_when_set(monkeypatch):
    import run_sidecar as rs
    base = ["--prompts", CAP, "--woosh-upgrade", "flow", "--woosh-upgrade-solver", "midpoint:16"]
    up = _rest_cmd(monkeypatch, base)
    assert "--upgrade-best" not in up and "--upgrade-verifier" not in up
    one = _rest_cmd(monkeypatch, base + ["--woosh-upgrade-best", "1"])
    assert one == up
    four = _rest_cmd(monkeypatch, base + ["--woosh-upgrade-best", "4"])
    assert four[:len(up)] == up and four[len(up):] == ["--upgrade-best", "4"]
    path = _rest_cmd(monkeypatch, base + ["--woosh-upgrade-best", "8", "--woosh-upgrade-verifier", "X:/clap"])
    assert path[len(up):] == ["--upgrade-best", "8", "--upgrade-verifier", "X:/clap"]
    for bad in (["--woosh-upgrade-best", "4"], ["--woosh-upgrade", "flow", "--woosh-upgrade-best", "0"],
                ["--woosh-upgrade", "flow", "--woosh-upgrade-verifier", "X:/clap"]):
        with pytest.raises(SystemExit):
            rs.main(["--generator", "hybrid", "--no-cache"] + bad)
