import _bootstrap

import glob
import json
import os

import numpy as np
import torch
from scipy.io import wavfile

from validate_panel import _resample, judge_clap, judge_ast, judge_panns
from gate_quality import onset_window, load_wav


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    real = [load_wav(p) for p in sorted(glob.glob(os.path.join("data", "real_footsteps", "*.wav")))]
    sets = {
        "real_full_5s": [(x, r) for x, r in real],
        "real_crop_0.6s": [(onset_window(x, r, dur=0.6), r) for x, r in real],
    }
    gen = sorted(glob.glob(os.path.join("data", "_gate", "steps3", "*.wav")))
    sets["gen_steps3_0.6s"] = [load_wav(p) for p in gen]

    res = {}
    for name, items in sets.items():
        r = {}
        r["ast"] = judge_ast([_resample(x, s, 16000) for x, s in items], dev)
        r["panns"] = judge_panns([_resample(x, s, 32000) for x, s in items], dev)
        labs = ["wood"] * len(items)
        r["clap"] = judge_clap([_resample(x, s, 48000) for x, s in items], labs,
                               ["wood", "stone", "grass", "metal"], dev)
        res[name] = r
        print("%-18s n=%2d  AST %5.1f%%  PANNs %5.1f%%  CLAP-footstep %5.1f%%   AST top1 %s"
              % (name, len(items), 100 * r["ast"]["footstep_rate"],
                 100 * r["panns"]["footstep_rate"], 100 * r["clap"]["footstep_rate"],
                 r["ast"]["top1"][:3]))

    json.dump({k: {j: {kk: vv for kk, vv in v.items()} for j, v in d.items()}
               for k, d in res.items()},
              open(os.path.join("scripts", "bench", "out_calibrate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
