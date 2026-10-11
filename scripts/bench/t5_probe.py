import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import json
import time

import numpy as np
import torch

import vocab_eval as ve
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey

PROMPTS = ["a probe prompt number %d, a clear distinct footstep, dry, isolated sound effect" % i for i in range(12)]


def timed(fn, n=1):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0 / n


def main():
    g = StableAudio3Generator(sampler_type="euler", steps=3, fast=False, hoist_prompts=None)
    g.ensure_ready()
    cdm = g._model.model.conditioner
    names = {n: type(c).__name__ for n, c in cdm.conditioners.items()}
    print("conditioners", names, flush=True)
    t5c = [c for c in cdm.conditioners.values() if type(c).__name__ == "T5GemmaConditioner"][0]
    cdm([{"prompt": "warm", "seconds_total": 1.0}], "cuda")
    m = t5c.model
    p0 = next(m.parameters())
    rep = {"conditioners": names, "t5_dtype": str(p0.dtype), "t5_params_m": sum(p.numel() for p in m.parameters()) / 1e6,
           "proj_out": str(t5c.proj_out), "torch": torch.__version__}
    import transformers
    rep["transformers"] = transformers.__version__
    print(rep, flush=True)
    dev = "cuda"
    enc = t5c.tokenizer(PROMPTS, truncation=True, max_length=t5c.max_length, padding="max_length", return_tensors="pt")
    ids = enc["input_ids"].to(dev)
    mask = enc["attention_mask"].to(dev).to(torch.bool)
    rep["tokens"] = [int(x) for x in enc["attention_mask"].sum(1)]

    def run(model, i):
        with torch.no_grad():
            return model(input_ids=ids[i:i + 1], attention_mask=mask[i:i + 1])["last_hidden_state"]

    ref = [run(m, i).float().clone() for i in range(len(PROMPTS))]
    _, ms = timed(lambda: run(m, 0), 10)
    rep["eager_ms"] = ms
    _, ms = timed(lambda: t5c.tokenizer(PROMPTS[:1], truncation=True, max_length=t5c.max_length,
                                         padding="max_length", return_tensors="pt"), 20)
    rep["tokenize_ms"] = ms
    _, ms = timed(lambda: cdm([{"prompt": PROMPTS[0], "seconds_total": 1.0}], dev), 10)
    rep["conditioner_ms"] = ms
    print("eager T5 %.2f ms, tokenizer %.2f ms, whole conditioner %.2f ms" % (
        rep["eager_ms"], rep["tokenize_ms"], rep["conditioner_ms"]), flush=True)

    def diff(fn):
        d = []
        for i in range(len(PROMPTS)):
            y = fn(i).float()
            msk = mask[i:i + 1].unsqueeze(-1)
            d.append(float(((y - ref[i]).abs() * msk).max()))
        return max(d)

    s_ids = ids[:1].clone()
    s_mask = mask[:1].clone()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                m(input_ids=s_ids, attention_mask=s_mask)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            s_out = m(input_ids=s_ids, attention_mask=s_mask)["last_hidden_state"]

        def graphed(i):
            s_ids.copy_(ids[i:i + 1])
            s_mask.copy_(mask[i:i + 1])
            graph.replay()
            return s_out

        rep["graph_ms"] = timed(lambda: graphed(0), 20)[1]
        rep["graph_maxdiff"] = diff(graphed)
        print("graph %.2f ms, max diff %.3g" % (rep["graph_ms"], rep["graph_maxdiff"]), flush=True)
    except Exception as e:
        rep["graph_error"] = repr(e)[:400]
        print("graph failed", rep["graph_error"], flush=True)

    for mode in ("default", "reduce-overhead"):
        try:
            torch._dynamo.reset()
            cm = torch.compile(m, mode=mode, dynamic=False)
            t0 = time.perf_counter()
            for i in range(3):
                run(cm, i)
            torch.cuda.synchronize()
            rep["compile_%s_first_s" % mode] = time.perf_counter() - t0
            rep["compile_%s_ms" % mode] = timed(lambda: run(cm, 1).clone(), 20)[1]
            rep["compile_%s_maxdiff" % mode] = diff(lambda i: run(cm, i).clone())
            print("compile %s %.2f ms, max diff %.3g, first %.1f s" % (
                mode, rep["compile_%s_ms" % mode], rep["compile_%s_maxdiff" % mode],
                rep["compile_%s_first_s" % mode]), flush=True)
        except Exception as e:
            rep["compile_%s_error" % mode] = repr(e)[:400]
            print("compile", mode, "failed", rep["compile_%s_error" % mode], flush=True)

    mh = m.half()
    try:
        rep["half_ms"] = timed(lambda: run(mh, 0), 10)[1]
        rep["half_maxdiff"] = diff(lambda i: run(mh, i))
        print("fp16 eager %.2f ms, max diff %.3g" % (rep["half_ms"], rep["half_maxdiff"]), flush=True)
    except Exception as e:
        rep["half_error"] = repr(e)[:400]
    rep["ref_absmax"] = max(float(r.abs().max()) for r in ref)
    json.dump(rep, open(os.path.join(BENCH, "out_t5_probe.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
