import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import argparse

from prompt_routes import VARIANTS
from sarge.generators.stable_audio import build_prompt
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey

ROOT = os.path.dirname(os.path.dirname(BENCH))
KEYS = [("footstep", "water"), ("impact", "water"), ("land", "water"), ("splash", "water")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out-dir", default=os.path.join("data", "_gate", "water_nodry"))
    args = ap.parse_args(argv)
    g = StableAudio3Generator(sampler_type="euler", steps=args.steps, detail_steps=args.steps)
    g.ensure_ready()
    g.generate(CacheKey(*KEYS[0], "normal"), variant=0)
    for name, fn in (("current", build_prompt), ("nodry", VARIANTS["nodry"])):
        d = os.path.join(ROOT, args.out_dir, "water_s%d_%s" % (args.steps, name))
        os.makedirs(d, exist_ok=True)
        g.prompts = {str(CacheKey(et, mat, "normal")): fn(et, mat, "normal") for et, mat in KEYS}
        for et, mat in KEYS:
            for v in range(args.seeds):
                g.generate(CacheKey(et, mat, "normal"), variant=v).to_wav(
                    os.path.join(d, "%s-%s_%02d.wav" % (et, mat, v)))
        print("%s: %s" % (name, " | ".join(g.prompts.values())), flush=True)
    print("static %s" % dict(g.static_stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
