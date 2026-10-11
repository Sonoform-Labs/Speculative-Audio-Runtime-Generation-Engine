import time

T0 = time.perf_counter()

import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import hashlib
import json

import torch

from sarge.generators.stable_audio3 import StableAudio3Generator


def digest(t):
    t = t.detach().contiguous().cpu()
    return hashlib.sha1(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"").hexdigest() + str(t.dtype)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    fast = "--fast" in argv
    g = StableAudio3Generator(sampler_type="euler", steps=3, fast=False, hoist_prompts=None, fast_load=fast)
    t = time.perf_counter()
    g.ensure_ready()
    load_s = time.perf_counter() - t
    sam = g._model
    sums = {}
    for n, v in sam.model.state_dict().items():
        sums["model." + n] = digest(v)
    for cn, c in sam.model.conditioner.conditioners.items():
        inner = vars(c).get("model")
        if inner is not None:
            for n, v in inner.state_dict().items():
                sums["text.%s.%s" % (cn, n)] = digest(v)
    path = os.path.join(BENCH, "out_load_probe_ref.json")
    rep = {"fast_load": fast, "load_s": load_s, "since_start_s": time.perf_counter() - T0, "tensors": len(sums)}
    if fast:
        ref = json.load(open(path))
        rep["missing"] = sorted(set(ref) ^ set(sums))[:20]
        rep["mismatched"] = sorted(k for k in ref if k in sums and ref[k] != sums[k])[:50]
        json.dump(rep, open(os.path.join(BENCH, "out_load_probe.json"), "w"), indent=1)
    else:
        json.dump(sums, open(path, "w"))
    print(json.dumps(rep, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
