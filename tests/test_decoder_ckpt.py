import os
import sys
import types

import pytest
import torch
from safetensors.torch import save_file

import sarge.generators.sa3_fast as fastmod
from sarge.generators.stable_audio3 import StableAudio3Generator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 3)
        self.register_buffer("inv_freq", torch.arange(3, dtype=torch.float32))
        with torch.no_grad():
            self.proj.weight.fill_(0.5)
            self.proj.bias.fill_(-0.25)


class FakeSam:
    loads = 0

    def __init__(self, half):
        dec = TinyDecoder()
        if half:
            dec = dec.half()
        self.same = types.SimpleNamespace(model=types.SimpleNamespace(decoder=dec))
        self.model = types.SimpleNamespace(pretransform=None)
        self.model_config = {"sample_rate": 44100}
        self.sample_rate = 44100

    @classmethod
    def from_pretrained(cls, model_name, device=None, model_half=True):
        cls.loads += 1
        return cls(model_half)


@pytest.fixture
def fake_sa3(monkeypatch):
    mod = types.ModuleType("stable_audio_3")
    mod.StableAudioModel = FakeSam
    monkeypatch.setitem(sys.modules, "stable_audio_3", mod)
    FakeSam.loads = 0
    return FakeSam


def tuned_state():
    d = TinyDecoder()
    with torch.no_grad():
        d.proj.weight.copy_(torch.linspace(-1.0, 1.0, 12).view(3, 4))
        d.proj.bias.copy_(torch.tensor([0.1, 0.2, 0.3]))
    return {k: v.detach().float().contiguous() for k, v in d.state_dict().items()}


def make_gen(path=None, fast=False):
    return StableAudio3Generator(device="cpu", fast=fast, hoist_prompts=None, static_guard=False, decoder_ckpt=path)


def decoder_of(gen):
    return gen._model.same.model.decoder


def test_checkpoint_is_loaded_strict_into_the_half_decoder(fake_sa3, tmp_path):
    sd = tuned_state()
    p = tmp_path / "dec.safetensors"
    save_file(sd, str(p))
    g = make_gen(str(p))
    g.ensure_ready()
    dec = decoder_of(g)
    assert dec.proj.weight.dtype == torch.float16
    assert torch.equal(dec.proj.weight, sd["proj.weight"].half())
    assert torch.equal(dec.proj.bias, sd["proj.bias"].half())
    assert g.decoder_ckpt == str(p) and fake_sa3.loads == 1


def test_no_checkpoint_keeps_the_base_decoder(fake_sa3):
    g = make_gen(None)
    g.ensure_ready()
    assert g.decoder_ckpt is None
    assert torch.all(decoder_of(g).proj.weight == 0.5)


def test_bad_checkpoint_raises_and_a_retry_never_falls_back_to_base(fake_sa3, tmp_path):
    sd = tuned_state()
    sd["extra.weight"] = torch.zeros(2)
    p = tmp_path / "bad.safetensors"
    save_file(sd, str(p))
    g = make_gen(str(p))
    with pytest.raises(RuntimeError):
        g.ensure_ready()
    assert g._model is None
    with pytest.raises(RuntimeError):
        g.ensure_ready()
    assert g._model is None and fake_sa3.loads == 1


def test_missing_key_is_rejected(fake_sa3, tmp_path):
    sd = tuned_state()
    del sd["proj.bias"]
    p = tmp_path / "short.safetensors"
    save_file(sd, str(p))
    with pytest.raises(RuntimeError):
        make_gen(str(p)).ensure_ready()


def test_checkpoint_lands_before_the_accelerator_wraps_decode(fake_sa3, tmp_path, monkeypatch):
    seen = {}

    class FakeAccel:
        def __init__(self, sam, **kw):
            seen["weight"] = sam.same.model.decoder.proj.weight.detach().clone()
            seen["kw"] = kw

    monkeypatch.setattr(fastmod, "Accelerator", FakeAccel)
    sd = tuned_state()
    p = tmp_path / "dec.safetensors"
    save_file(sd, str(p))
    g = make_gen(str(p), fast=True)
    g.ensure_ready()
    assert g._fast is not None
    assert torch.equal(seen["weight"], sd["proj.weight"].half())


@pytest.fixture
def sidecar(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import run_sidecar
    captured = {}

    class Stop(Exception):
        pass

    def fake_serve(**kw):
        captured.update(kw)
        raise Stop()

    monkeypatch.setattr(run_sidecar, "serve", fake_serve)
    monkeypatch.delenv("SARGE_DECODER_CKPT", raising=False)
    return run_sidecar, captured, Stop


def test_sidecar_rejects_a_missing_checkpoint_resolved_from_the_repo_root(sidecar):
    rs, _cap, _stop = sidecar
    rel = os.path.join("data", "decoder_ft", "__no_such_decoder__.safetensors")
    with pytest.raises(SystemExit) as e:
        rs.main(["--generator", "mock", "--no-cache", "--decoder-ckpt", rel])
    assert os.path.join(ROOT, rel) in str(e.value)


def test_sidecar_passes_the_checkpoint_to_the_generator(sidecar, tmp_path):
    rs, cap, stop = sidecar
    p = tmp_path / "dec.safetensors"
    save_file(tuned_state(), str(p))
    with pytest.raises(stop):
        rs.main(["--generator", "mock", "--no-cache", "--decoder-ckpt", str(p)])
    assert cap["gen_kwargs"]["decoder_ckpt"] == str(p)


def test_sidecar_env_fallback_and_default_base(sidecar, tmp_path, monkeypatch):
    rs, cap, stop = sidecar
    with pytest.raises(stop):
        rs.main(["--generator", "mock", "--no-cache"])
    assert "decoder_ckpt" not in cap["gen_kwargs"]
    p = tmp_path / "env.safetensors"
    save_file(tuned_state(), str(p))
    monkeypatch.setenv("SARGE_DECODER_CKPT", str(p))
    cap.clear()
    with pytest.raises(stop):
        rs.main(["--generator", "mock", "--no-cache"])
    assert cap["gen_kwargs"]["decoder_ckpt"] == str(p)
