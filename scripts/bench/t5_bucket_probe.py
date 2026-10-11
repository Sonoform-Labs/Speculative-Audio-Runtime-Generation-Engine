import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import json
import time

import torch

import vocab_eval as ve
from sarge.generators.stable_audio import build_prompt
from sarge.generators.stable_audio3 import StableAudio3Generator


def timed(fn, n=20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0 / n


def main():
    g = StableAudio3Generator(sampler_type="euler", steps=3, fast=False, hoist_prompts=None, fast_load=True)
    g.ensure_ready()
    cdm = g._model.model.conditioner
    t5c = cdm.conditioners["prompt"]
    cdm([{"prompt": "warm", "seconds_total": 1.0}], "cuda")
    m = t5c.model
    prompts = [build_prompt(et, mat, i) for et, mat, _t in ve.KEYS for i in ("soft", "normal", "hard")]
    lens = [int(t5c.tokenizer(p, return_tensors="pt")["attention_mask"].sum()) for p in prompts]
    rep = {"prompts": len(prompts), "token_len_max": max(lens), "token_len_median": sorted(lens)[len(lens) // 2]}
    full = {}
    for p in prompts:
        enc = t5c.tokenizer([p], truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        ids, mask = enc["input_ids"].cuda(), enc["attention_mask"].cuda().bool()
        with torch.no_grad():
            full[p] = (m(input_ids=ids, attention_mask=mask)["last_hidden_state"].float(), mask)
    for bucket in (32, 48, 64):
        diffs, same = [], 0
        for p in prompts:
            enc = t5c.tokenizer([p], truncation=True, max_length=bucket, padding="max_length", return_tensors="pt")
            ids, mask = enc["input_ids"].cuda(), enc["attention_mask"].cuda().bool()
            with torch.no_grad():
                y = m(input_ids=ids, attention_mask=mask)["last_hidden_state"].float()
            ref, rmask = full[p]
            n = int(mask.sum())
            d = float((y[:, :n] - ref[:, :n]).abs().max())
            diffs.append(d)
            same += int(d == 0.0)
        enc = t5c.tokenizer([prompts[0]], truncation=True, max_length=bucket, padding="max_length", return_tensors="pt")
        ids, mask = enc["input_ids"].cuda(), enc["attention_mask"].cuda().bool()
        with torch.no_grad():
            _, ms = timed(lambda: m(input_ids=ids, attention_mask=mask))
        rep["bucket_%d" % bucket] = {"identical": same, "max_diff": max(diffs), "eager_ms": ms}
        print(bucket, rep["bucket_%d" % bucket], flush=True)
    enc = t5c.tokenizer([prompts[0]], truncation=True, max_length=128, padding="max_length", return_tensors="pt")
    ids, mask = enc["input_ids"].cuda(), enc["attention_mask"].cuda().bool()
    with torch.no_grad():
        _, ms = timed(lambda: m(input_ids=ids, attention_mask=mask))
    rep["bucket_128_eager_ms"] = ms
    rep["ref_absmax"] = max(float(v[0].abs().max()) for v in full.values())
    print(json.dumps(rep, indent=1), flush=True)
    json.dump(rep, open(os.path.join(BENCH, "out_t5_bucket_probe.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
