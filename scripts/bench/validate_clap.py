import _bootstrap

import argparse

import numpy as np

from sage.generators import get_generator
from sage.types import CacheKey

CATEGORIES = ["a footstep", "a gunshot", "music", "human speech", "silence", "rain", "an explosion"]


def _to48k(x, sr):
    if sr == 48000:
        return x.astype(np.float32)
    n2 = int(round(len(x) * 48000 / sr))
    idx = np.linspace(0, len(x), n2, endpoint=False)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Validate generated SFX with CLAP (independent audio-language model).")
    p.add_argument("--generator", default="stableaudio")
    p.add_argument("--event-type", default="footstep", dest="event_type")
    p.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    p.add_argument("--intensity", default="normal")
    p.add_argument("--variants", type=int, default=5)
    p.add_argument("--model", default="laion/clap-htsat-unfused")
    args = p.parse_args(argv)

    import torch
    from transformers import ClapModel, ClapProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clap = ClapModel.from_pretrained(args.model).to(dev).eval()
    proc = ClapProcessor.from_pretrained(args.model)

    gen = get_generator(args.generator)
    clips, labels = [], []
    for m in args.materials:
        for v in range(args.variants):
            c = gen.generate(CacheKey(args.event_type, m, args.intensity), variant=v)
            clips.append(_to48k(c.samples, c.sample_rate))
            labels.append(m)

    def classify(arrs, texts):
        a = proc(audio=arrs, sampling_rate=48000, return_tensors="pt")
        t = proc(text=texts, return_tensors="pt", padding=True)
        inp = {k: v.to(dev) for k, v in {**a, **t}.items()}
        with torch.no_grad():
            out = clap(**inp)
        return out.logits_per_audio.float().cpu().numpy()

    def softmax_rows(x):
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    sim_mat = classify(clips, [f"the sound of a footstep on {m}" for m in args.materials])
    probs_mat = softmax_rows(sim_mat)
    pred_idx = sim_mat.argmax(axis=1)
    true_idx = [args.materials.index(l) for l in labels]
    acc = sum(int(pred_idx[i] == true_idx[i]) for i in range(len(labels))) / len(labels)

    sim_cat = classify(clips, CATEGORIES)
    foot_idx = CATEGORIES.index("a footstep")
    foot_top = sum(int(sim_cat[i].argmax() == foot_idx) for i in range(len(labels)))

    print(f"clips: {len(labels)}  ({args.variants} x {len(args.materials)} materials)")
    print(f"CLAP material accuracy : {acc*100:5.1f}%   (chance = {100/len(args.materials):.0f}%)")
    print(f"reads as 'a footstep'  : {foot_top}/{len(labels)} = {foot_top/len(labels)*100:.0f}%  "
          f"(top of {len(CATEGORIES)} sound categories)")
    print()
    print("confusion (row = true material, col = CLAP's pick):")
    print("          " + " ".join(f"{m[:5]:>6}" for m in args.materials))
    for ti, m in enumerate(args.materials):
        row = [0] * len(args.materials)
        for i in range(len(labels)):
            if true_idx[i] == ti:
                row[pred_idx[i]] += 1
        print(f"  {m:<7} " + " ".join(f"{x:>6}" for x in row))
    print()
    print("mean softmax prob assigned to the TRUE material:")
    for ti, m in enumerate(args.materials):
        vals = [probs_mat[i][ti] for i in range(len(labels)) if true_idx[i] == ti]
        print(f"  {m:<7} {np.mean(vals):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
