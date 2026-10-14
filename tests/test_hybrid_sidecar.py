import os
import sys
import types

import pytest

from sarge.affordance import AffordanceEngine
from sarge.generators.mock import MockGenerator
from sarge.sidecar import ProtocolError, SfxClient, serve

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


class _MissingWeights(MockGenerator):
    def generate(self, key, *, variant=0):
        if key.event_type == "shoot":
            raise FileNotFoundError("stable-audio-3-medium not in the offline cache")
        return super().generate(key, variant=variant)


def test_a_generation_oserror_comes_back_as_an_error_and_keeps_the_connection():
    eng = AffordanceEngine(_MissingWeights(), realtime=True, vram_budget_kb=200_000)
    srv = serve(engine=eng, host="127.0.0.1", port=0, warmup=False)
    try:
        with SfxClient(*srv.server_address) as c:
            with pytest.raises(ProtocolError) as e:
                c.event("shoot", "pistol", "normal")
            assert e.value.code == "internal" and "offline cache" in e.value.message and not e.value.fatal
            meta, samples = c.event("footstep", "wood", "normal")
            assert meta["served"] == "realtime" and len(samples) > 0
            assert c.ping(1.5)["t"] == 1.5
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def rs(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import run_sidecar
    return run_sidecar


def _fake_sa3(monkeypatch, found, offline=True):
    cfg = types.SimpleNamespace(repo_id="stabilityai/stable-audio-3-medium", config_path="model_config.json",
                                ckpt_path="model.safetensors")
    pkg = types.ModuleType("stable_audio_3")
    mc = types.ModuleType("stable_audio_3.model_configs")
    mc.models = {"medium": cfg}
    pkg.model_configs = mc
    monkeypatch.setitem(sys.modules, "stable_audio_3", pkg)
    monkeypatch.setitem(sys.modules, "stable_audio_3.model_configs", mc)
    import huggingface_hub
    from huggingface_hub import constants
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    seen = []

    def lookup(repo, f):
        seen.append((repo, f))
        return "/cache/" + f if found else None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lookup)
    return seen


def test_check_weights_stops_at_startup_when_the_offline_cache_lacks_the_model(rs, monkeypatch):
    pytest.importorskip("huggingface_hub")
    _fake_sa3(monkeypatch, found=False)
    with pytest.raises(SystemExit) as e:
        rs.check_weights("medium")
    assert "stable-audio-3-medium" in str(e.value) and "HF_HOME" in str(e.value)


def test_check_weights_passes_when_cached_or_online(rs, monkeypatch):
    pytest.importorskip("huggingface_hub")
    seen = _fake_sa3(monkeypatch, found=True)
    rs.check_weights("medium")
    assert [f for _r, f in seen] == ["model_config.json", "model.safetensors"]
    seen = _fake_sa3(monkeypatch, found=False, offline=False)
    rs.check_weights("medium")
    rs.check_weights("no-such-model")
    assert seen == []


def test_hybrid_starts_woosh_from_the_generator_after_the_guns(rs, monkeypatch, tmp_path):
    captured, started = {}, []

    class Stop(Exception):
        pass

    def fake_serve(**kw):
        captured.update(kw)
        raise Stop()

    py = tmp_path / "python.exe"
    py.write_bytes(b"")
    monkeypatch.setattr(rs, "serve", fake_serve)
    monkeypatch.setattr(rs, "check_weights", lambda name: started.append("check " + name))
    monkeypatch.setattr(rs, "start_woosh", lambda args: started.append("woosh %d" % args.woosh_port))
    monkeypatch.delenv("SARGE_DECODER_CKPT", raising=False)
    with pytest.raises(Stop):
        rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", str(py), "--woosh-port", "8799"])
    assert started == ["check medium"]
    kw = captured["gen_kwargs"]
    assert kw["model_name"] == "medium" and kw["woosh_port"] == 8799
    kw["start_rest"]()
    assert started == ["check medium", "woosh 8799"]


def test_hybrid_without_the_woosh_python_stops_before_loading(rs, monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "serve", lambda **kw: pytest.fail("serve must not run"))
    monkeypatch.setattr(rs, "check_weights", lambda name: None)
    with pytest.raises(SystemExit) as e:
        rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", str(tmp_path / "none.exe")])
    assert "woosh python not found" in str(e.value)
