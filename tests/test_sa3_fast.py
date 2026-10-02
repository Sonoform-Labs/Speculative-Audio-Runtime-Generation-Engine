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
