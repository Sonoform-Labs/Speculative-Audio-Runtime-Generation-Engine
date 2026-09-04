import _bootstrap

import argparse
import os

import numpy as np
import torch
import torchaudio.functional as AF

from sarge.generators import get_generator
from sarge.types import CacheKey

CATEGORIES = ["a footstep", "a gunshot", "music", "speech", "silence", "rain", "an explosion"]
FOOT_TERMS = ("footstep", "walk")


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
        if any(t in n for n in names for t in FOOT_TERMS):
            hits += 1
    del m
    torch.cuda.empty_cache()
    return {"footstep_rate": hits / len(clips_16k), "top1": tops[:6]}


def judge_panns(clips_32k, dev):
    from panns_inference import AudioTagging
    from panns_inference.config import labels
    at = AudioTagging(checkpoint_path=None, device=dev)
    x = np.stack(clips_32k).astype(np.float32)
    clipwise, _ = at.inference(x)
    hits = 0
    tops = []
    for row in clipwise:
        idx = np.argsort(row)[::-1][:5]
        names = [labels[i].lower() for i in idx]
        tops.append(names[0])
        if any(t in n for n in names for t in FOOT_TERMS):
            hits += 1
    del at
    torch.cuda.empty_cache()
    return {"footstep_rate": hits / len(clipwise), "top1": tops[:6]}


def judge_audiobox(paths):
    from audiobox_aesthetics.infer import initialize_predictor
    p = initialize_predictor()
    out = p.forward([{"path": pth} for pth in paths])
    keys = list(out[0].keys())
    return {k: float(np.mean([o[k] for o in out])) for k in keys}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Panel of independent models validating generated SFX.")
    p.add_argument("--generator", default="stableaudio")
    p.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    p.add_argument("--intensity", default="normal")
    p.add_argument("--variants", type=int, default=5)
    p.add_argument("--judges", nargs="+", default=["clap", "ast", "panns", "audiobox"])
    args = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = get_generator(args.generator)
    sr = g.sample_rate
    clips, labels, paths = [], [], []
    tmp = os.path.join("data", "_panel")
    os.makedirs(tmp, exist_ok=True)
    for mat in args.materials:
        for v in range(args.variants):
            clip = g.generate(CacheKey("footstep", mat, args.intensity), variant=v)
            clips.append(clip.samples)
            labels.append(mat)
            pth = os.path.join(tmp, f"{mat}_{v:02d}.wav")
            clip.to_wav(pth)
            paths.append(pth)
    del g
    torch.cuda.empty_cache()

    res = {}
    if "clap" in args.judges:
        res["clap"] = judge_clap([_resample(c, sr, 48000) for c in clips], labels, args.materials, dev)
    if "ast" in args.judges:
        res["ast"] = judge_ast([_resample(c, sr, 16000) for c in clips], dev)
    if "panns" in args.judges:
        res["panns"] = judge_panns([_resample(c, sr, 32000) for c in clips], dev)
    if "audiobox" in args.judges:
        res["audiobox"] = judge_audiobox(paths)

    chance = 100.0 / len(args.materials)
    print(f"\nclips: {len(clips)}  ({args.variants} x {len(args.materials)} materials)\n")
    print("=" * 64)
    print(" AUTOMATED VALIDATION PANEL")
    print("=" * 64)
    rates = []
    if "clap" in res:
        r = res["clap"]
        rates.append(r["footstep_rate"])
        print(f" CLAP  (text-audio)      footstep {r['footstep_rate']*100:5.1f}%  | material {r['material_acc']*100:5.1f}% (chance {chance:.0f}%)")
    if "ast" in res:
        r = res["ast"]
        rates.append(r["footstep_rate"])
        print(f" AST   (AudioSet, ViT)   footstep {r['footstep_rate']*100:5.1f}%  | top1 {r['top1'][:3]}")
    if "panns" in res:
        r = res["panns"]
        rates.append(r["footstep_rate"])
        print(f" PANNs (AudioSet, CNN)   footstep {r['footstep_rate']*100:5.1f}%  | top1 {r['top1'][:3]}")
    if "audiobox" in res:
        r = res["audiobox"]
        print(" Audiobox-Aesthetics     " + "  ".join(f"{k} {v:.2f}" for k, v in r.items()) + "   (scale 1-10)")
    if rates:
        print("-" * 64)
        print(f" CONSENSUS footstep-ness (min across taggers): {min(rates)*100:.1f}%")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
