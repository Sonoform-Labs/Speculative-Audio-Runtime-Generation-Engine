import _bootstrap

import argparse

import numpy as np
import torch
import torchaudio.functional as AF

from predictive_sfx.generators import get_generator
from predictive_sfx.types import CacheKey

CATEGORIES = ["a footstep", "a gunshot", "music", "speech", "silence", "rain", "an explosion"]


def _resample(clip, src, dst):
    if src == dst:
        return clip.astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(clip)).float()
    return AF.resample(t, src, dst).numpy().astype(np.float32)


def judge_clap(clips_48k, labels, materials, dev):
    from transformers import ClapModel, ClapProcessor
    m = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    pr = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

    def classify(texts):
        a = pr(audio=clips_48k, sampling_rate=48000, return_tensors="pt")
        t = pr(text=texts, return_tensors="pt", padding=True)
        inp = {k: v.to(dev) for k, v in {**a, **t}.items()}
        with torch.no_grad():
            out = m(**inp)
        return out.logits_per_audio.float().cpu().numpy()

    cat = classify(CATEGORIES)
    foot = float((cat.argmax(1) == CATEGORIES.index("a footstep")).mean())
    mat = classify([f"the sound of a footstep on {x}" for x in materials])
    true = np.array([materials.index(l) for l in labels])
    acc = float((mat.argmax(1) == true).mean())
    del m
    torch.cuda.empty_cache()
    return {"footstep_rate": foot, "material_acc": acc}


def judge_ast(clips_16k, dev):
    from transformers import ASTForAudioClassification, AutoFeatureExtractor
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    m = ASTForAudioClassification.from_pretrained(name).to(dev).eval()
    fe = AutoFeatureExtractor.from_pretrained(name)
    id2label = m.config.id2label
    hits = 0
    tops = []
    for c in clips_16k:
        inp = fe(c, sampling_rate=16000, return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.no_grad():
            logits = m(**inp).logits[0]
        idx = torch.topk(logits, 5).indices.cpu().tolist()
        names = [id2label[i].lower() for i in idx]
        tops.append(names[0])
        if any(("footstep" in n or "walk" in n) for n in names):
            hits += 1
    del m
    torch.cuda.empty_cache()
    return {"footstep_rate": hits / len(clips_16k), "top1_examples": tops[:6]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Panel of independent models validating generated SFX.")
    p.add_argument("--generator", default="stableaudio")
    p.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    p.add_argument("--intensity", default="normal")
    p.add_argument("--variants", type=int, default=5)
    args = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = get_generator(args.generator)
    sr = g.sample_rate
    clips, labels = [], []
    for mat in args.materials:
        for v in range(args.variants):
            clips.append(g.generate(CacheKey("footstep", mat, args.intensity), variant=v).samples)
            labels.append(mat)
    del g
    torch.cuda.empty_cache()

    clips_48k = [_resample(c, sr, 48000) for c in clips]
    clips_16k = [_resample(c, sr, 16000) for c in clips]

    clap = judge_clap(clips_48k, labels, args.materials, dev)
    ast = judge_ast(clips_16k, dev)

    chance = 100.0 / len(args.materials)
    print(f"clips: {len(clips)}  ({args.variants} x {len(args.materials)} materials)\n")
    print("JUDGE 1 - CLAP (laion, text-audio contrastive)")
    print(f"  footstep-ness: {clap['footstep_rate']*100:5.1f}%   material acc: {clap['material_acc']*100:5.1f}%  (chance {chance:.0f}%)")
    print("JUDGE 2 - AST (MIT, AudioSet spectrogram transformer)")
    print(f"  footstep-ness: {ast['footstep_rate']*100:5.1f}%   (footstep/walk in top-5 of 527 AudioSet classes)")
    print(f"  AST top-1 picks: {ast['top1_examples']}\n")
    consensus = min(clap["footstep_rate"], ast["footstep_rate"])
    print(f"CONSENSUS footstep-ness (both independent models agree): >= {consensus*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
