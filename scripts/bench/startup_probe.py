import time

T0 = time.perf_counter()

import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import json

import torch

T_TORCH = time.perf_counter()

from sarge.generators import stable_audio3 as sa3
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey

T_IMPORT = time.perf_counter()
ROOT_TMP = os.environ.get("TEMP", BENCH)

CFG = {"sampler_type": "euler", "steps": 3, "compile_mode": "default", "compile_decode": True, "graph_decode": True,
       "hoist_rope": True, "skip_first_step": True, "fp16_accumulation": True, "graph_text": True, "fast_load": True}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    tag = argv[0] if argv else ""
    marks = {"import_torch_s": T_TORCH - T0, "import_sarge_s": T_IMPORT - T_TORCH}
    phases = {}
    import torch._dynamo.utils as du

    g = StableAudio3Generator(**CFG)
    from stable_audio_3 import StableAudioModel
    orig_fp = StableAudioModel.from_pretrained

    def timed_fp(*a, **k):
        t = time.perf_counter()
        out = orig_fp(*a, **k)
        phases["from_pretrained_s"] = time.perf_counter() - t
        return out

    StableAudioModel.from_pretrained = timed_fp
    t = time.perf_counter()
    g.ensure_ready()
    marks["ensure_ready_s"] = time.perf_counter() - t
    marks.update(phases)
    sam = g._model
    pt = sam.model.pretransform
    dec = pt.decode
    dit_fwd = sam.dit.model._forward
    acc = {"dit_s": 0.0, "decode_s": 0.0}

    def wrap(name, fn):
        def w(*a, **k):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            acc[name] += time.perf_counter() - t1
            return out
        return w

    sam.dit.model._forward = wrap("dit_s", dit_fwd)
    pt.decode = wrap("decode_s", dec)
    t = time.perf_counter()
    c1 = g.generate(CacheKey("warmup", "grass", "normal"), variant=0)
    torch.cuda.synchronize()
    marks["first_generate_s"] = time.perf_counter() - t
    marks.update({"first_" + k: v for k, v in acc.items()})
    try:
        marks["dynamo_times"] = du.compile_times(repr="str", aggregate=True)
    except Exception as e:
        marks["dynamo_times_error"] = repr(e)[:200]
    try:
        marks["counters"] = {k: dict(v) for k, v in du.counters.items() if v}
    except Exception as e:
        marks["counters_error"] = repr(e)[:200]
    t = time.perf_counter()
    c2 = g.generate(CacheKey("warmup", "stone", "normal"), variant=0)
    torch.cuda.synchronize()
    marks["second_generate_s"] = time.perf_counter() - t
    clips = [c1.samples, c2.samples] + [g.generate(CacheKey(et, mat, "normal"), variant=3).samples
                                        for et, mat in (("footstep", "wood"), ("shoot", "rifle"), ("break", "window"))]
    if tag:
        import numpy as np
        np.savez(os.path.join(ROOT_TMP, "startup_%s.npz" % tag), *clips)
    marks["total_s"] = time.perf_counter() - T0
    marks["torch"] = torch.__version__
    print(json.dumps({k: v for k, v in marks.items() if k != "dynamo_times"}, indent=1)[:3000], flush=True)
    print(marks.get("dynamo_times", ""), flush=True)
    json.dump(marks, open(os.path.join(BENCH, "out_startup_probe%s.json" % ("_" + tag if tag else "")), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
