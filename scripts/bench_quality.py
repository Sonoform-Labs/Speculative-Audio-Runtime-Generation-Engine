import _bootstrap

import argparse
import os

import torch

from predictive_sfx.generators import get_generator
from predictive_sfx.types import AudioClip, CacheKey
from validate_panel import _resample, judge_clap

MATS = ["wood", "stone", "grass", "metal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sa3")
    ap.add_argument("--steps", default="8,4,2,1")
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--save", default="docs/step-samples")
    args = ap.parse_args()
    steps_list = [int(s) for s in args.steps.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    gen = get_generator(args.backend)
    ready = getattr(gen, "ensure_ready", None)
    if ready:
        ready()
    sr = int(getattr(gen, "sample_rate", 44100))
    os.makedirs(args.save, exist_ok=True)
    print(f"backend={args.backend} sr={sr} dev={dev} variants={args.variants}")
    print(f"{'steps':>6}{'footstep_rate':>15}{'material_acc':>14}")
    for steps in steps_list:
        if hasattr(gen, "steps"):
            gen.steps = steps
        clips, labels = [], []
        for mat in MATS:
            for v in range(args.variants):
                clip = gen.generate(CacheKey("footstep", mat, "normal"), variant=v)
                clips.append(_resample(clip.samples, sr, 48000))
                labels.append(mat)
                if v == 0:
                    AudioClip(clip.samples, sr).to_wav(f"{args.save}/footstep_{mat}_steps{steps}.wav")
        res = judge_clap(clips, labels, MATS, dev)
        print(f"{steps:>6}{res['footstep_rate']:>15.3f}{res['material_acc']:>14.3f}")


if __name__ == "__main__":
    main()
