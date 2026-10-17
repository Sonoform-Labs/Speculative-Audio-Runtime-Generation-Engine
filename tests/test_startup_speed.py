import io
import json
import os
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from sarge import prefetch
from sarge import weight_cache as wc
from sarge.generators.routed import RoutedGenerator

SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")


def _st(path, tensors):
    from safetensors.numpy import save_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(tensors, path)
    return path


def _hub(tmp_path, cache_first=False):
    snap = tmp_path / "hub" / "models--stabilityai--stable-audio-3-medium" / "snapshots" / "abc"
    (tmp_path / "hub" / "models--stabilityai--stable-audio-3-medium" / "refs").mkdir(parents=True)
    (tmp_path / "hub" / "models--stabilityai--stable-audio-3-medium" / "refs" / "main").write_text("abc\n")
    t5 = snap / "t5gemma-b-b-ul2"
    t5.mkdir(parents=True)
    (t5 / "tokenizer.json").write_text("{}")
    _st(str(t5 / "model.safetensors"), {"model.decoder.w": np.ones((64, 4), np.float32),
                                         "model.encoder.a": np.ones((8, 4), np.float32),
                                         "model.encoder.b": np.zeros((2, 4), np.float32)})
    _st(str(snap / "model.safetensors"), {"model.model.w": np.ones((4, 4), np.float32)})
    cfg = {"model": {"conditioning": {"configs": [
        {"id": "prompt", "type": "t5gemma", "config": {"repo_id": "stabilityai/stable-audio-3-medium",
                                                         "subfolder": "t5gemma-b-b-ul2"}},
        {"id": "seconds_total", "type": "number", "config": {}}]}}}
    (snap / "model_config.json").write_text(json.dumps(cfg))
    return str(tmp_path / "hub"), snap


def test_hub_dir_prefers_the_hub_cache_then_hf_home():
    assert prefetch.hub_dir({"HF_HUB_CACHE": "X", "HF_HOME": "Y"}) == "X"
    assert prefetch.hub_dir({"HF_HOME": "Y"}) == os.path.join("Y", "hub")
    assert prefetch.hub_dir({}).endswith(os.path.join(".cache", "huggingface", "hub"))


def test_sa3_plan_reads_the_t5_encoder_range_then_the_checkpoint(tmp_path):
    hub, snap = _hub(tmp_path)
    items = prefetch.sa3_items("medium", hub=hub)
    assert [x["label"] for x in items] == ["t5 tokenizer", "t5 encoder", "sa3 weights"]
    base, h = prefetch.header(str(snap / "t5gemma-b-b-ul2" / "model.safetensors"))
    enc = [h[k]["data_offsets"] for k in ("model.encoder.a", "model.encoder.b")]
    assert (items[1]["start"], items[1]["end"]) == (base + min(a for a, _ in enc), base + max(b for _, b in enc))
    assert items[1]["end"] - items[1]["start"] == 10 * 4 * 4
    assert items[2]["path"] == os.path.realpath(str(snap / "model.safetensors")) and items[2]["start"] == 0
    assert prefetch.sa3_items("no-such-model", hub=hub) == []
    assert prefetch.sa3_items("medium", hub=str(tmp_path / "empty")) == []


def test_sa3_plan_reads_the_weight_cache_in_place_of_the_checkpoint(tmp_path):
    hub, _snap = _hub(tmp_path)
    cache = _st(str(tmp_path / "c" / "w.safetensors"), {"a": np.ones(3, np.float16)})
    items = prefetch.sa3_items("medium", hub=hub, weights=cache)
    assert [x["label"] for x in items][-1] == "sa3 weight cache" and items[-1]["path"] == os.path.realpath(cache)
    assert all("sa3 weights" != x["label"] for x in items)


def test_woosh_plan_follows_the_model_config_dependencies(tmp_path):
    ck = tmp_path / "checkpoints"
    for d in ("Woosh-DFlow", "TextConditionerA", "Woosh-AE"):
        (ck / d).mkdir(parents=True)
        (ck / d / "weights.safetensors").write_bytes(b"x" * 10)
    (ck / "Woosh-DFlow" / "config.yaml").write_text(
        "ldm:\n  conditioners:\n    text:\n      path: checkpoints/TextConditionerA\n  autoencoder:\n"
        "    path: checkpoints/Woosh-AE\n")
    labels = [x["label"] for x in prefetch.woosh_items("dflow", home=str(tmp_path))]
    assert labels == ["woosh Woosh-AE", "woosh TextConditionerA", "woosh Woosh-DFlow"]
    assert prefetch.woosh_items("nope", home=str(tmp_path)) == []


def test_prefetcher_reads_every_byte_and_skips_a_file_when_ram_is_short(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(os.urandom(3 * 1024 * 1024 + 17))
    p = prefetch.Prefetcher([prefetch.item(str(a), "a", 5, None)], streams=3, block=1 << 20, reserve=0,
                            free=lambda: 10 ** 12).start()
    assert p.done.wait(30)
    assert p.log[0]["mb"] == round((a.stat().st_size - 5) / 1e6, 1) and "skipped" not in p.log[0]
    q = prefetch.Prefetcher([prefetch.item(str(a), "a")], reserve=1 << 30, free=lambda: 1 << 20).start()
    assert q.done.wait(30) and q.log[0]["skipped"].startswith("ram")


def test_prefetcher_runs_one_queue_per_drive_in_plan_order():
    items = [{"path": r"F:\x", "label": "1", "start": 0, "end": 1}, {"path": r"G:\y", "label": "2", "start": 0, "end": 1},
             {"path": r"F:\z", "label": "3", "start": 0, "end": 1}]
    q = prefetch.Prefetcher(items).queues()
    assert {k: [x["label"] for x in v] for k, v in q.items()} == {"F:": ["1", "3"], "G:": ["2"]}


def test_prefetch_switch(monkeypatch):
    assert not prefetch.enabled({"SARGE_PREFETCH": "0"}) and prefetch.enabled({})
    assert prefetch.start_hybrid("medium", env={"SARGE_PREFETCH": "off"}) is None


def _ckpt(tmp_path):
    ck = _st(str(tmp_path / "m" / "model.safetensors"), {"w": np.ones((2, 2), np.float32)})
    cfg = tmp_path / "m" / "model_config.json"
    cfg.write_text('{"a": 1}')
    return ck, str(cfg)


def test_cache_key_changes_with_the_weights_config_torch_and_code(tmp_path, monkeypatch):
    ck, cfg = _ckpt(tmp_path)
    k0 = wc.key_of(wc.fingerprint("medium", ck, cfg))
    assert k0 == wc.key_of(wc.fingerprint("medium", ck, cfg))
    st = os.stat(ck)
    os.utime(ck, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    k1 = wc.key_of(wc.fingerprint("medium", ck, cfg))
    Path(cfg).write_text('{"a": 2}')
    k2 = wc.key_of(wc.fingerprint("medium", ck, cfg))
    real = wc._version
    monkeypatch.setattr(wc, "_version", lambda n: "9.9" if n == "torch" else real(n))
    k3 = wc.key_of(wc.fingerprint("medium", ck, cfg))
    monkeypatch.setattr(wc, "_digest", lambda paths: "other code")
    k4 = wc.key_of(wc.fingerprint("medium", ck, cfg))
    _st(ck, {"w": np.ones((2, 3), np.float32)})
    os.utime(ck, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    monkeypatch.setattr(wc, "_version", real)
    assert len({k0, k1, k2, k3, k4}) == 5
    assert wc.path_for("D", "medium", k0) == os.path.join("D", "sa3-medium-fp16-%s.safetensors" % k0)


def test_locate_returns_the_cache_only_when_it_exists(tmp_path):
    hub, snap = _hub(tmp_path)
    key = wc.key_of(wc.fingerprint("medium", str(snap / "model.safetensors"), str(snap / "model_config.json")))
    d = tmp_path / "cache"
    assert wc.locate("medium", str(d), hub=hub) is None and wc.locate("medium", None, hub=hub) is None
    _st(wc.path_for(str(d), "medium", key), {"a": np.ones(1, np.float16)})
    assert wc.locate("medium", str(d), hub=hub) == wc.path_for(str(d), "medium", key)


def test_the_default_folder_is_used_only_where_its_parent_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "DEFAULT_DIR", str(tmp_path / "sarge" / "cache"))
    assert wc.default_dir({}) is None
    (tmp_path / "sarge").mkdir()
    assert wc.default_dir({}) == wc.DEFAULT_DIR
    assert wc.default_dir({"SARGE_WEIGHT_CACHE": "off"}) is None
    assert wc.default_dir({"SARGE_WEIGHT_CACHE": "X:\\elsewhere"}) == "X:\\elsewhere"


def test_the_key_follows_the_loader_code_and_ignores_the_rest_of_the_generator(tmp_path):
    src = (Path(wc.HERE) / "generators" / "stable_audio3.py").read_text(encoding="utf-8")
    p = tmp_path / "g.py"
    p.write_text(src, encoding="utf-8")
    base = wc.segments(str(p), wc.LOADER)
    assert all(("%s=None" % n) not in base for n in wc.LOADER)
    p.write_text(src.replace("    def ensure_ready(self):", "    def ensure_ready(self):\n        pass"), encoding="utf-8")
    assert wc.segments(str(p), wc.LOADER) == base
    p.write_text(src.replace("inner.lora_names = []", "inner.lora_names = [None]"), encoding="utf-8")
    assert wc.segments(str(p), wc.LOADER) != base
    p.write_text(src.replace('"orthogonal_")', '"orthogonal_", "eye_")'), encoding="utf-8")
    assert wc.segments(str(p), wc.LOADER) != base
    assert wc.loader_code().count(":") == 1


def _net(seed):
    import torch

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(5, 3)
            self.norm = torch.nn.LayerNorm(3)
            self.register_buffer("steps", torch.arange(4))
            self.register_buffer("scale", torch.linspace(0.1, 1.7, 6))
            self.register_buffer("freq", torch.linspace(1e-5, 3.3, 7), persistent=False)

    torch.manual_seed(seed)
    return Net().eval().requires_grad_(False)


def _all(m):
    out = dict(m.named_parameters())
    out.update(dict(m.named_buffers()))
    return out


def test_restore_rebuilds_the_fp16_state_bit_for_bit(tmp_path):
    import torch
    head = _net(0).to("cpu").to(torch.float16)
    path = wc.write(wc.snapshot_state(head), str(tmp_path / "c" / "sa3-medium-fp16-k.safetensors"), "k",
                    {"model": "medium"}, "medium")
    fresh = _net(1)
    assert not torch.equal(fresh.lin.weight, _net(0).lin.weight)
    wc.restore(fresh, path, "k", "cpu")
    a, b = _all(head), _all(fresh)
    assert set(a) == set(b) and "freq" in b
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]), k
    assert b["steps"].dtype == torch.int64 and b["lin.weight"].dtype == torch.float16
    assert not b["lin.weight"].requires_grad
    assert os.listdir(tmp_path / "c") == ["sa3-medium-fp16-k.safetensors"]


def test_restore_refuses_another_key_or_another_model(tmp_path):
    import torch
    head = _net(0).to(torch.float16)
    path = wc.write(wc.snapshot_state(head), str(tmp_path / "x.safetensors"), "k")
    with pytest.raises(ValueError, match="cache key"):
        wc.restore(_net(1), path, "other", "cpu")
    sd = wc.snapshot_state(head)
    sd.pop("scale")
    bad = wc.write(sd, str(tmp_path / "y.safetensors"), "k")
    with pytest.raises(ValueError, match="tensors"):
        wc.restore(_net(1), bad, "k", "cpu")


def test_restore_catches_a_changed_byte_and_load_hooks(tmp_path):
    import torch
    head = _net(0).to(torch.float16)
    path = wc.write(wc.snapshot_state(head), str(tmp_path / "x.safetensors"), "k")
    good = wc.restore(_net(1), path, "k", "cpu")
    assert torch.equal(good.lin.weight, head.lin.weight)
    raw = bytearray(Path(path).read_bytes())
    raw[-3] ^= 0x10
    Path(path).write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="checksum of 1 tensors"):
        wc.restore(_net(1), path, "k", "cpu")
    path = wc.write(wc.snapshot_state(head), str(tmp_path / "y.safetensors"), "k")
    net = _net(1)
    net.lin.register_load_state_dict_pre_hook(lambda *a, **k: None)
    with pytest.raises(ValueError, match="hooks or overrides on \\['lin'\\]"):
        wc.restore(net, path, "k", "cpu")
    sd = wc.snapshot_state(head)
    bare = str(tmp_path / "z.safetensors")
    from safetensors.torch import save_file
    save_file(sd, bare, metadata={"key": "k"})
    with pytest.raises(ValueError, match="no checksums"):
        wc.restore(_net(1), bare, "k", "cpu")


def test_checksums_see_every_byte_of_every_dtype():
    import torch
    sd = {"h": torch.arange(-5, 5, dtype=torch.float16), "f": torch.linspace(0, 1, 7), "i": torch.arange(4),
          "b": torch.tensor([True, False, True]), "e": torch.zeros(0, dtype=torch.float16),
          "t": torch.arange(12, dtype=torch.float16).reshape(3, 4).t()}
    a = wc.checksums(sd)
    assert set(a) == set(sd) and a["e"] == "0:0"
    for k in sd:
        x = {n: v.clone() for n, v in sd.items()}
        x[k] = x[k].clone().contiguous()
        if x[k].numel():
            x[k].view(-1).view(torch.uint8)[0] ^= 1
            assert wc.checksums(x)[k] != a[k], k
    assert wc.checksums({"t": sd["t"].contiguous()})["t"] == a["t"]


def test_a_failed_write_leaves_no_file_behind(tmp_path, monkeypatch):
    import safetensors.torch as stt
    import torch

    def boom(sd, path, metadata=None):
        open(path, "wb").write(b"half")
        raise KeyboardInterrupt()

    monkeypatch.setattr(stt, "save_file", boom)
    with pytest.raises(KeyboardInterrupt):
        wc.write(wc.snapshot_state(_net(0).to(torch.float16)), str(tmp_path / "z.safetensors"), "k")
    assert os.listdir(tmp_path) == []


def test_the_writer_works_in_its_own_folder_and_a_killed_one_is_swept(tmp_path, monkeypatch):
    import safetensors.torch as stt
    import torch
    seen = []
    real = stt.save_file

    def spy(sd, path, metadata=None):
        seen.append(os.path.relpath(path, tmp_path))
        sub = os.path.join(os.path.dirname(path), ".tmpAbC123")
        open(sub, "wb").write(b"half")
        real(sd, path, metadata=metadata)
        os.remove(sub)

    monkeypatch.setattr(stt, "save_file", spy)
    wc.write(wc.snapshot_state(_net(0).to(torch.float16)), str(tmp_path / "sa3-medium-fp16-k.safetensors"), "k")
    assert seen == [os.path.join(wc.PART + str(os.getpid()), "sa3-medium-fp16-k.safetensors")]
    assert os.listdir(tmp_path) == ["sa3-medium-fp16-k.safetensors"]
    dead = tmp_path / (wc.PART + "999999999")
    dead.mkdir()
    (dead / ".tmpXyZ").write_bytes(b"half")
    live = tmp_path / (wc.PART + str(os.getpid()))
    live.mkdir()
    (tmp_path / ".tmpOld123").write_bytes(b"half")
    (tmp_path / "sa3-medium-fp16-q.safetensors.999999998.tmp").write_bytes(b"half")
    (tmp_path / ".tmpNew456").write_bytes(b"half")
    now = os.path.getmtime(tmp_path / ".tmpNew456")
    old = now - 2 * wc.GRACE_S
    os.utime(tmp_path / ".tmpOld123", (old, old))
    gone = sorted(os.path.basename(p) for p in wc.sweep(str(tmp_path), now=now + 1))
    assert gone == [".part-999999999", ".tmpOld123", "sa3-medium-fp16-q.safetensors.999999998.tmp"]
    assert sorted(os.listdir(tmp_path)) == [wc.PART + str(os.getpid()), ".tmpNew456", "sa3-medium-fp16-k.safetensors"]
    gone = sorted(os.path.basename(p) for p in wc.sweep(str(tmp_path), now=now + 2 * wc.STALE_TMP_S))
    assert gone == [wc.PART + str(os.getpid()), ".tmpNew456"]
    assert not wc._alive(999999999) and wc._alive(os.getpid())


def test_prune_keeps_the_new_cache_and_the_newest_other_one(tmp_path):
    keep = tmp_path / "sa3-medium-fp16-new.safetensors"
    names = ("sa3-medium-fp16-new.safetensors", "sa3-medium-fp16-old.safetensors", "sa3-medium-fp16-older.safetensors",
             "sa3-small-fp16-old.safetensors", "other.bin")
    for n in names:
        (tmp_path / n).write_bytes(b"1")
    t = os.path.getmtime(tmp_path / "other.bin")
    for i, n in enumerate(("sa3-medium-fp16-new.safetensors", "sa3-medium-fp16-old.safetensors",
                           "sa3-medium-fp16-older.safetensors")):
        os.utime(tmp_path / n, (t - 100 * (i + 1), t - 100 * (i + 1)))
    os.utime(keep, (t - 1000, t - 1000))
    gone = wc.prune(str(tmp_path), "medium", str(keep))
    assert [os.path.basename(p) for p in gone] == ["sa3-medium-fp16-older.safetensors"]
    assert sorted(os.listdir(tmp_path)) == ["other.bin", "sa3-medium-fp16-new.safetensors",
                                            "sa3-medium-fp16-old.safetensors", "sa3-small-fp16-old.safetensors"]
    assert [os.path.basename(p) for p in wc.prune(str(tmp_path), "medium", str(keep), n=1)] == [
        "sa3-medium-fp16-old.safetensors"]


def test_a_folder_without_room_fails_before_the_state_is_copied(tmp_path, monkeypatch):
    import torch
    copied = []
    monkeypatch.setattr(wc, "snapshot_state", lambda m: copied.append(m) or {})
    monkeypatch.setattr(wc.shutil, "disk_usage", lambda d: types.SimpleNamespace(free=1 << 20))
    with pytest.raises(OSError, match="not enough free space"):
        wc.save_async(_net(0).to(torch.float16), str(tmp_path / "c" / "x.safetensors"), "k")
    assert copied == [] and os.listdir(tmp_path / "c") == []


def _fake_torch(cuda=True):
    calls = []
    return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda,
                                                            empty_cache=lambda: calls.append("empty"))), calls


def _sa3(tmp_path, monkeypatch):
    pytest.importorskip("stable_audio_3")
    import stable_audio_3.model_configs as mc
    from sarge.generators.stable_audio3 import StableAudio3Generator
    ck, cfg = _ckpt(tmp_path)
    monkeypatch.setitem(mc.all_models, "unit-test", types.SimpleNamespace(resolve=lambda: (cfg, ck)))
    g = StableAudio3Generator(model_name="unit-test", device="cuda", weight_cache=str(tmp_path / "cache"))
    key = wc.key_of(wc.fingerprint("unit-test", ck, cfg))
    return g, wc.path_for(str(tmp_path / "cache"), "unit-test", key), key


def test_the_slow_path_runs_without_cuda_or_without_a_cache(tmp_path, monkeypatch):
    g, path, key = _sa3(tmp_path, monkeypatch)
    torch, _ = _fake_torch(cuda=False)
    assert g._cached_model(torch) is None and g._cache_target is None
    torch, _ = _fake_torch()
    assert g._cached_model(torch) is None and g._cache_target == (path, key)
    assert not os.path.exists(path)


def test_an_unusable_cache_falls_back_and_is_marked_for_rebuild(tmp_path, monkeypatch):
    import stable_audio_3.loading_utils as lu
    g, path, _key = _sa3(tmp_path, monkeypatch)
    _st(path, {"w": np.ones(2, np.float16)})
    built = []
    monkeypatch.setattr(lu, "create_diffusion_cond_from_config", lambda cfg: built.append(cfg) or _net(0))
    torch, calls = _fake_torch()
    assert g._cached_model(torch) is None
    assert built == [{"a": 1}] and calls == ["empty"] and g._cache_target[0] == path
    assert not os.path.exists(path)


def test_an_unusable_cache_that_cannot_be_removed_is_not_rebuilt(tmp_path, monkeypatch):
    import stat
    import stable_audio_3.loading_utils as lu
    g, path, _key = _sa3(tmp_path, monkeypatch)
    _st(path, {"w": np.ones(2, np.float16)})
    (Path(path).parent / (wc.PART + "999999999")).mkdir()
    monkeypatch.setattr(lu, "create_diffusion_cond_from_config", lambda cfg: _net(0))
    os.chmod(path, stat.S_IREAD)
    try:
        assert g._cached_model(_fake_torch()[0]) is None
        assert g._cache_target is None and os.path.exists(path)
        assert os.listdir(Path(path).parent) == [os.path.basename(path)]
    finally:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


class _Ordered:
    def __init__(self, tag, log):
        self.tag, self.log, self.sample_rate = tag, log, 44100

    def ensure_ready(self):
        self.log.append(self.tag + " ready")

    def generate(self, key, *, variant=0):
        self.log.append(self.tag + " gen")


def test_load_rest_runs_once_right_after_the_guns_load():
    log = []
    gen = RoutedGenerator(guns=_Ordered("guns", log), rest=_Ordered("rest", log),
                          start_rest=lambda: log.append("start rest"), load_rest=lambda: log.append("load rest"))
    gen.ensure_ready()
    assert log == ["guns ready", "load rest", "guns gen", "start rest", "rest ready", "guns gen"]
    log.clear()
    gen.ensure_ready()
    assert log == ["guns ready", "rest ready", "guns gen"]


def test_a_gun_warm_up_failure_releases_the_held_worker_and_never_opens_it():
    log = []

    class Broken(_Ordered):
        def generate(self, key, *, variant=0):
            raise RuntimeError("out of memory")

    gen = RoutedGenerator(guns=Broken("guns", log), rest=_Ordered("rest", log),
                          start_rest=lambda: log.append("start rest"), load_rest=lambda: log.append("load rest"),
                          drop_rest=lambda: log.append("drop rest"))
    with pytest.raises(RuntimeError, match="out of memory"):
        gen.ensure_ready()
    assert log == ["guns ready", "load rest", "drop rest"]
    ok = RoutedGenerator(guns=_Ordered("guns", log), rest=_Ordered("rest", log),
                         start_rest=lambda: log.append("start rest"), drop_rest=lambda: log.append("drop rest"))
    log.clear()
    ok.ensure_ready()
    assert "drop rest" not in log and ok.drop_rest is None


@pytest.fixture
def rs(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import run_sidecar
    return run_sidecar


def test_early_woosh_launches_after_the_weight_check_and_opens_its_gates_in_turn(rs, monkeypatch, tmp_path):
    seen, captured = [], {}

    class Stop(Exception):
        pass

    def fake_serve(**kw):
        captured.update(kw)
        raise Stop()

    py = tmp_path / "python.exe"
    py.write_bytes(b"")
    monkeypatch.setenv("SARGE_WOOSH_EARLY", "1")
    monkeypatch.setattr(rs, "serve", fake_serve)
    monkeypatch.setattr(rs, "check_weights", lambda name: seen.append("check"))
    monkeypatch.setattr(rs, "launch_woosh", lambda args: seen.append("launch") or ("proc", "up", "state"))
    monkeypatch.setattr(rs, "gate", lambda proc, name: seen.append("gate %s %s" % (proc, name)))
    monkeypatch.setattr(rs, "open_woosh", lambda args, *w: seen.append("open %s" % (w,)))
    monkeypatch.setattr(rs, "start_woosh", lambda args: pytest.fail("the late start must not run"))
    with pytest.raises(Stop):
        rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", str(py)])
    assert seen == ["check", "launch"]
    kw = captured["gen_kwargs"]
    kw["load_rest"]()
    kw["start_rest"]()
    assert seen[2:] == ["gate proc load", "open ('proc', 'up', 'state')"]
    assert rs.early_woosh({}) and not rs.early_woosh({"SARGE_WOOSH_EARLY": "0"})


def test_the_weight_cache_folder_reaches_the_guns_unless_off(rs, monkeypatch, tmp_path):
    captured = {}

    class Stop(Exception):
        pass

    def fake_serve(**kw):
        captured.update(kw)
        raise Stop()

    py = tmp_path / "python.exe"
    py.write_bytes(b"")
    monkeypatch.setattr(rs, "serve", fake_serve)
    monkeypatch.setattr(rs, "check_weights", lambda name: None)
    for flag, want in ((["--weight-cache", str(tmp_path)], str(tmp_path)), (["--weight-cache", "off"], None)):
        with pytest.raises(Stop):
            rs.main(["--generator", "hybrid", "--no-cache", "--woosh-python", str(py)] + flag)
        assert captured["gen_kwargs"].get("weight_cache") == want


class _Proc:
    def __init__(self, cmd, lines=(), **kw):
        self.cmd, self.kw = cmd, kw
        self.stdin = io.StringIO()
        self.stdout = iter(lines)

    def poll(self):
        return 3


def test_held_woosh_gets_the_flag_and_a_stdin_and_is_not_waited_on(rs, monkeypatch, tmp_path):
    procs = []
    monkeypatch.setattr(rs.subprocess, "Popen", lambda cmd, **kw: procs.append(_Proc(cmd, **kw)) or procs[-1])
    monkeypatch.setattr(rs.atexit, "register", lambda f: None)
    (tmp_path / "py").write_bytes(b"")
    args = types.SimpleNamespace(woosh_python=str(tmp_path / "py"), woosh_model="dflow", woosh_port=8771, amp=False,
                                 stereo=False, woosh_hf_cache=str(tmp_path), woosh_frames=None)
    p = rs.start_woosh(args, hold=True)
    assert p.cmd[-1] == "--hold" and p.kw["stdin"] is rs.subprocess.PIPE


def test_open_woosh_opens_the_gpu_gate_then_waits_for_listening(rs):
    args = types.SimpleNamespace(woosh_model="dflow")
    p = _Proc([], ["[startup] 1.0 s woosh torch import\n", "woosh server listening on 127.0.0.1:8771\n", "later\n"])
    up, state = threading.Event(), {"listening": False}

    def pump():
        for line in p.stdout:
            if "listening on" in line:
                state["listening"] = True
                up.set()
        up.set()

    rs.gate(p, "load")
    threading.Thread(target=pump).start()
    assert rs.open_woosh(args, p, up, state) is p
    assert p.stdin.getvalue() == "load\ngpu\n"
    dead = _Proc([], [])
    ev = threading.Event()
    ev.set()
    with pytest.raises(SystemExit, match="exited before listening"):
        rs.open_woosh(args, dead, ev, {"listening": False})


def test_a_released_worker_gets_eof_and_a_later_start_launches_a_fresh_one(rs, monkeypatch):
    p = _Proc([], [])
    rs.close_woosh(p)
    assert p.stdin.closed
    started = []
    monkeypatch.setattr(rs, "start_woosh", lambda args: started.append(args) or "fresh")
    assert rs.open_woosh("args", p, threading.Event(), {"listening": False}) == "fresh" and started == ["args"]


def test_woosh_hold_waits_for_its_own_line_and_exits_on_eof(monkeypatch):
    if SCRIPTS not in sys.path:
        monkeypatch.syspath_prepend(SCRIPTS)
    import woosh_server as ws
    monkeypatch.setattr(ws.sys, "stdin", io.StringIO("noise\nload\ngpu\n"))
    ws.hold("load")
    monkeypatch.setitem(ws.HOLD, "on", True)
    ws.hold("load")
    assert ws.sys.stdin.readline() == "gpu\n"

    class Exit(Exception):
        pass

    def fake_exit(code):
        raise Exit(code)

    monkeypatch.setattr(ws.os, "_exit", fake_exit)
    monkeypatch.setattr(ws.sys, "stdin", io.StringIO("load\n"))
    with pytest.raises(Exit):
        ws.hold("gpu")


def test_decoder_prefill_refuses_flex_attention_in_its_own_thread_only_and_runs_once_per_shape(monkeypatch):
    import torch
    tm = pytest.importorskip("stable_audio_3.models.transformer")
    from sarge.generators.sa3_fast import GraphedDecode
    monkeypatch.setattr(tm, "flex_attention_compiled", lambda *a, **k: "real")
    g = GraphedDecode.__new__(GraphedDecode)
    g.eager, g.prefilled = None, set()
    z = torch.zeros(1, 4, 8)
    assert g._prefill(z, None) is False
    before = tm.flex_attention_compiled
    seen, other = [], []

    def eager(x, chunked=None):
        seen.append((tuple(x.shape), chunked))
        with pytest.raises(RuntimeError, match="prefill"):
            tm.flex_attention_compiled("q", "k", "v")
        th = threading.Thread(target=lambda: other.append(tm.flex_attention_compiled("q", "k", "v")))
        th.start()
        th.join()
        tm._SLIDING_WINDOW_BLOCK_MASK_CACHE.setdefault(("unit", "test"), "bm")

    g.eager = eager
    try:
        assert g._prefill(z, True) is True
        assert seen == [((1, 4, 8), True)] and other == ["real"] and tm.flex_attention_compiled is before
        assert ("unit", "test") in tm._SLIDING_WINDOW_BLOCK_MASK_CACHE
        assert g._prefill(torch.zeros(3, 4, 8), True) is False and len(seen) == 1
        assert g._prefill(torch.zeros(1, 4, 9), True) is True and len(seen) == 2
        g.eager = lambda x, chunked=None: 1 / 0
        assert g._prefill(z, None) is False and tm.flex_attention_compiled is before
        assert g._prefill(z, None) is False and (tuple(z.shape[1:]), str(z.device), None) not in g.prefilled
    finally:
        tm._SLIDING_WINDOW_BLOCK_MASK_CACHE.pop(("unit", "test"), None)


