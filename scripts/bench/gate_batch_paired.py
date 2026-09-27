import _bootstrap

import argparse
import json
import os

import numpy as np
import torch

from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.types import CacheKey
from validate_panel import CATEGORIES, _resample
from gate_quality import transients, crest_db, attack_ms, onset_window


def clap_per_clip(clips_48k, materials, dev):
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

    foot = classify(CATEGORIES).argmax(1) == CATEGORIES.index("a footstep")
    mat = classify([f"the sound of a footstep on {x}" for x in materials]).argmax(1)
    del m
    torch.cuda.empty_cache()
    return foot, mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=10)
    ap.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_gate_paired.json"))
    args = ap.parse_args()
    dev = "cuda"

    g = StableAudio3Generator()
    g.ensure_ready()
    sr = int(g.sample_rate)
    real_randn = torch.randn
    state = {"noise": None, "shapes": []}

    def fake_randn(*a, **kw):
        shape = a[0] if len(a) == 1 and isinstance(a[0], (list, tuple)) else list(a)
        if len(shape) == 3:
            state["shapes"].append(list(shape))
            if state["noise"] is not None:
                return state["noise"][: shape[0]].clone()
        return real_randn(*a, **kw)

    torch.randn = fake_randn
    g.generate(CacheKey("footstep", "wood", "normal"), variant=0)
    io, L = state["shapes"][0][1], state["shapes"][0][2]

    items = [(CacheKey("footstep", mat, "normal"), v)
             for mat in args.materials for v in range(args.variants)]
    single, batched = [None] * len(items), [None] * len(items)
    for i in range(0, len(items), 2):
        pair = items[i: i + 2]
        gen = torch.Generator(device="cuda").manual_seed(5000 + i)
        noise = real_randn([len(pair), io, L], device="cuda", generator=gen)
        state["noise"] = noise
        out = g.generate_batch(pair)
        for j, (k, v) in enumerate(pair):
            batched[i + j] = out[j]
            state["noise"] = noise[j: j + 1]
            single[i + j] = g.generate(k, variant=v)
    torch.randn = real_randn
    del g
    torch.cuda.empty_cache()

    paths = {}
    for tag, clips in (("single", single), ("batch", batched)):
        d = os.path.join("data", "_gate", "paired_" + tag)
        os.makedirs(d, exist_ok=True)
        paths[tag] = []
        for (k, v), c in zip(items, clips):
            p = os.path.join(d, "%s_%02d.wav" % (k.material, v))
            c.to_wav(p)
            paths[tag].append(p)

    labels = np.array([args.materials.index(k.material) for k, _ in items])
    res = {}
    per = {}
    for tag, clips in (("single", single), ("batch", batched)):
        x = [c.samples for c in clips]
        foot, mat = clap_per_clip([_resample(c, sr, 48000) for c in x], args.materials, dev)
        crest = np.array([crest_db(onset_window(c, sr)) for c in x])
        att = np.array([attack_ms(onset_window(c, sr), sr) for c in x])
        per[tag] = {"foot": foot, "mat_ok": mat == labels, "mat": mat, "crest": crest, "attack": att}
        res[tag] = {"transients": transients(x, sr),
                    "clap_footstep_rate": float(foot.mean()),
                    "clap_material_acc": float((mat == labels).mean())}

    from audiobox_aesthetics.infer import initialize_predictor
    pred = initialize_predictor()
    for tag in ("single", "batch"):
        ab = pred.forward([{"path": p} for p in paths[tag]])
        per[tag]["PQ"] = np.array([o["PQ"] for o in ab])
        res[tag]["audiobox"] = {k: float(np.mean([o[k] for o in ab])) for k in ab[0].keys()}

    s, b = per["single"], per["batch"]
    agree = {
        "n": len(items),
        "footstep_decision_agree": float((s["foot"] == b["foot"]).mean()),
        "material_decision_agree": float((s["mat"] == b["mat"]).mean()),
        "mean_abs_crest_db": float(np.mean(np.abs(s["crest"] - b["crest"]))),
        "mean_abs_attack_ms": float(np.mean(np.abs(s["attack"] - b["attack"]))),
        "mean_abs_PQ": float(np.mean(np.abs(s["PQ"] - b["PQ"]))),
        "footstep_flips_single_to_batch": [int(np.sum(s["foot"] & ~b["foot"])), int(np.sum(~s["foot"] & b["foot"]))],
    }
    res["paired"] = agree

    print()
    print("=" * 96)
    print(" BATCH vs SINGLE, IDENTICAL NOISE PER CLIP (n=%d)" % len(items))
    print("=" * 96)
    for tag in ("single", "batch"):
        r = res[tag]
        t = r["transients"]
        print("%-7s crest %5.1f dB  attack %5.2f ms  CLAP footstep %4.0f%%  material %4.0f%%  PQ %5.2f  CE %5.2f"
              % (tag, t["crest_db"], t["attack_ms"], 100 * r["clap_footstep_rate"],
                 100 * r["clap_material_acc"], r["audiobox"]["PQ"], r["audiobox"]["CE"]))
    print("-" * 96)
    print(" same CLAP footstep decision on %.0f%% of clips, same material decision on %.0f%%"
          % (100 * agree["footstep_decision_agree"], 100 * agree["material_decision_agree"]))
    print(" per-clip mean |diff|: crest %.2f dB, attack %.2f ms, PQ %.3f"
          % (agree["mean_abs_crest_db"], agree["mean_abs_attack_ms"], agree["mean_abs_PQ"]))
    print("=" * 96)
    json.dump(res, open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()
