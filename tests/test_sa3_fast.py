import torch

from sarge.generators.sa3_fast import Accelerator, CondCache


class FakeConditioner:
    def __init__(self):
        self.calls = 0

    def __call__(self, cond, device):
        self.calls += 1
        n = len(cond[0]["prompt"])
        return {"prompt": [torch.full((1, 4), float(n)), torch.ones(1, 4, dtype=torch.bool)]}


class FakeCdm:
    def __init__(self):
        self.conditioner = FakeConditioner()


def test_cond_cache_encodes_each_prompt_once():
    c = CondCache(FakeCdm(), "cpu")
    for _ in range(5):
        c.get(["a footstep"], 1.0)
    assert c.cdm.conditioner.calls == 1 and c.hits == 4


def test_cond_cache_hands_out_fresh_dicts():
    c = CondCache(FakeCdm(), "cpu")
    _, a = c.get(["x"], 1.0)
    a["inpaint_mask"] = [torch.zeros(1)]
    _, b = c.get(["x"], 1.0)
    assert "inpaint_mask" not in b


def test_cond_cache_evicts_least_recent():
    c = CondCache(FakeCdm(), "cpu", max_prompts=2)
    c.get(["a"], 1.0)
    c.get(["bb"], 1.0)
    c.get(["a"], 1.0)
    c.get(["ccc"], 1.0)
    assert set(k[0] for k in c.store) == {"a", "ccc"}


def test_batches_bypass_the_cache_so_they_stay_bit_exact():
    acc = Accelerator.__new__(Accelerator)
    acc.cache = CondCache(FakeCdm(), "cpu")
    kw = acc.kwargs(["a", "b"], 1.0)
    assert kw["prompt"] == ["a", "b"] and "conditioning_tensors" not in kw
    kw1 = acc.kwargs(["a"], 1.0)
    assert "prompt" not in kw1 and kw1["conditioning"] == [{"prompt": "a", "seconds_total": 1.0}]
    assert kw1["use_checkpointing"] is False and kw1["disable_tqdm"] is True


class FakeT5:
    def __init__(self):
        self.calls = 0
        self.dtype = torch.bfloat16

    def __call__(self, input_ids=None, attention_mask=None):
        self.calls += 1
        return {"last_hidden_state": input_ids.float().unsqueeze(-1) * attention_mask.unsqueeze(-1)}


class FakeTextConditioner:
    def __init__(self):
        self.tokenizer = object()
        self.__dict__["model"] = FakeT5()


class FakeNumberConditioner:
    pass


class FakeMulti:
    def __init__(self):
        self.conditioners = {"prompt": FakeTextConditioner(), "seconds_total": FakeNumberConditioner()}


class FakeModel:
    def __init__(self):
        self.conditioner = FakeMulti()


def test_text_graph_wraps_only_the_text_encoder_and_falls_back_off_gpu():
    from sarge.generators.sa3_fast import GraphedText, install_text_graph
    m = FakeModel()
    inner = m.conditioner.conditioners["prompt"].model
    wrapped = install_text_graph(m)
    assert len(wrapped) == 1 and isinstance(m.conditioner.conditioners["prompt"].model, GraphedText)
    assert install_text_graph(m) == []
    g = m.conditioner.conditioners["prompt"].model
    assert g.dtype == torch.bfloat16
    ids = torch.tensor([[3, 4, 0]])
    mask = torch.tensor([[True, True, False]])
    out = g(input_ids=ids, attention_mask=mask)["last_hidden_state"]
    assert inner.calls == 1 and g.captured == 0 and out[0, :, 0].tolist() == [3.0, 4.0, 0.0]


def test_fast_package_scan_matches_the_stdlib_mapping():
    import importlib.metadata as md
    from sarge import fastmeta
    fast = fastmeta.packages_distributions()
    slow = md.packages_distributions()
    assert {k: sorted(v) for k, v in fast.items()} == {k: sorted(v) for k, v in slow.items()}


def test_skip_random_init_restores_the_init_functions():
    import torch.nn.init as init
    from sarge.generators.stable_audio3 import _skip_random_init
    before = init.kaiming_uniform_
    with _skip_random_init():
        t = torch.zeros(3)
        assert init.kaiming_uniform_(t) is t and t.abs().sum() == 0
    assert init.kaiming_uniform_ is before


def test_triton_linear_falls_back_for_cpu_and_other_shapes():
    from sarge.generators.sa3_gemm import CONFIGS, TritonLinear

    def op(*a):
        raise AssertionError("the triton op ran on a cpu tensor")

    lin = torch.nn.Linear(1024, 3072)
    assert (1024, 3072) in CONFIGS
    t = TritonLinear(lin, op)
    x = torch.randn(2, 5, 1024)
    assert torch.equal(t(x), lin(x))
    lin.half()
    xh = x.half()
    assert torch.equal(t(xh), torch.nn.functional.linear(xh, lin.weight, lin.bias))


def test_prefix_decode_keeps_the_shape_and_zeroes_the_padding():
    from sarge.generators.sa3_fast import PrefixDecode

    def dec(z, chunked=None):
        return z.mean(1, keepdim=True).repeat_interleave(4, dim=-1).repeat(1, 2, 1)

    p = PrefixDecode(dec, keep_frames=3, ratio=4)
    z = torch.randn(1, 8, 5)
    y = p(z, chunked=True)
    full = dec(z)
    assert y.shape == full.shape and p.cut == 1
    assert torch.allclose(y[..., :12], full[..., :12]) and torch.count_nonzero(y[..., 12:]) == 0
    short = torch.randn(1, 8, 3)
    assert torch.equal(p(short), dec(short)) and p.cut == 1
