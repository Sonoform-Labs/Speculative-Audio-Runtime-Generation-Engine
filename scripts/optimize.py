import _bootstrap

import argparse
import json
import os

import numpy as np
import torch
import torchaudio.functional as AF

from predictive_sfx.generators import get_generator
from predictive_sfx.types import AudioClip, CacheKey

CATEGORIES = ["a footstep", "a gunshot", "music", "speech", "silence", "rain", "an explosion"]
MUSIC_IDX = {CATEGORIES.index("music"), CATEGORIES.index("speech")}
FOOT = ("footstep", "walk")

REWARD_WEIGHTS = {"footstep_seq": 0.30, "material": 0.25, "quality": 0.20, "not_music": 0.15, "variation": 0.10}

PROMPTS = {
    "default": None,
    "plural": "footsteps on {material}, dry, mono",
    "floor": "a {intensity} footstep on a {material} floor, dry, mono",
}

NEG_MUSIC = "music, melody, song, ambient, drone, hum"
NEG_BROAD = "music, speech, silence, rain, drone, hum, tone"

CONFIGS = [
    dict(name="sa3_cfg1",       prompt="default", cfg=1.0, secs=1.0, steps=8),
    dict(name="sa3_cfg2",       prompt="default", cfg=2.0, secs=1.0, steps=8),
    dict(name="sa3_cfg4",       prompt="default", cfg=4.0, secs=1.0, steps=8),
    dict(name="sa3_cfg1_floor", prompt="floor",   cfg=1.0, secs=1.0, steps=8),
    dict(name="sa3_cfg1_dur04", prompt="default", cfg=1.0, secs=1.0, steps=8, dur=0.4),
    dict(name="sa3_cfg2_s16",   prompt="default", cfg=2.0, secs=1.0, steps=16),
]


def rs(c, src, dst):
    if src == dst:
        return c.astype(np.float32)
    return AF.resample(torch.from_numpy(np.ascontiguousarray(c)).float(), src, dst).numpy().astype(np.float32)


def sequence(variants, sr, gait_s=0.45):
    step = int(gait_s * sr)
    n = step * (len(variants) - 1) + max(len(c) for c in variants)
    buf = np.zeros(n, dtype=np.float32)
    for i, c in enumerate(variants):
        buf[i * step:i * step + len(c)] += c
    peak = float(np.max(np.abs(buf))) or 1.0
    return (buf / peak * 0.9).astype(np.float32)


def pairwise_corr(variants):
    L = min(len(c) for c in variants)
    m = np.stack([c[:L].astype(np.float64) for c in variants])
    m = m - m.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(m, axis=1)
    vals = []
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            d = nrm[i] * nrm[j]
            vals.append(abs(m[i] @ m[j] / d) if d > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Critic-loop optimizer: maximize a composite multi-model reward.")
    ap.add_argument("--materials", nargs="+", default=["wood", "stone", "grass", "metal"])
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--generator", default="stableaudio")
    args = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    g = get_generator(args.generator)
    g.generate(CacheKey("footstep", "wood", "normal"), variant=0)
    sr = g.sample_rate
    tmp = os.path.join("data", "_opt")
    os.makedirs(tmp, exist_ok=True)

    iso, iso_cfg, iso_mat, iso_path = [], [], [], []
    seqs, seq_cfg = [], []
    variation = {}
    for ci, cfg in enumerate(CONFIGS):
        g.prompt_template = PROMPTS[cfg["prompt"]]
        g.cfg_scale = cfg["cfg"]
        g.seconds_total = cfg["secs"]
        g.gen_duration_s = cfg["secs"]
        g.steps = cfg["steps"]
        g.duration_s = cfg.get("dur", 0.6)
        g.negative_prompt = cfg.get("neg")
        corrs = []
        for mat in args.materials:
            vs = [g.generate(CacheKey("footstep", mat, "normal"), variant=v).samples
                  for v in range(args.variants)]
            corrs.append(pairwise_corr(vs))
            for v, s in enumerate(vs):
                iso.append(s)
                iso_cfg.append(ci)
                iso_mat.append(mat)
                pth = os.path.join(tmp, f"c{ci}_{mat}_{v}.wav")
                AudioClip(s, sr).to_wav(pth)
                iso_path.append(pth)
            seqs.append(sequence(vs, sr))
            seq_cfg.append(ci)
        variation[ci] = 1.0 - float(np.mean(corrs))
    del g
    torch.cuda.empty_cache()

    from transformers import ClapModel, ClapProcessor
    cm = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
    cp = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    iso48 = [rs(c, sr, 48000) for c in iso]

    def clap_logits(clips, texts):
        a = cp(audio=clips, sampling_rate=48000, return_tensors="pt")
        t = cp(text=texts, return_tensors="pt", padding=True)
        inp = {k: v.to(dev) for k, v in {**a, **t}.items()}
        with torch.no_grad():
            return cm(**inp).logits_per_audio.float().cpu().numpy()

    cat = clap_logits(iso48, CATEGORIES)
    matlog = clap_logits(iso48, [f"the sound of a footstep on {m}" for m in args.materials])
    del cm
    torch.cuda.empty_cache()

    from transformers import ASTForAudioClassification, AutoFeatureExtractor
    an = "MIT/ast-finetuned-audioset-10-10-0.4593"
    am = ASTForAudioClassification.from_pretrained(an).to(dev).eval()
    afe = AutoFeatureExtractor.from_pretrained(an)
    aid = am.config.id2label
    seq_foot = []
    for s in seqs:
        inp = afe(rs(s, sr, 16000), sampling_rate=16000, return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.no_grad():
            lg = am(**inp).logits[0]
        names = [aid[i].lower() for i in torch.topk(lg, 5).indices.cpu().tolist()]
        seq_foot.append(any(t in n for n in names for t in FOOT))
    del am
    torch.cuda.empty_cache()

    from audiobox_aesthetics.infer import initialize_predictor
    abp = initialize_predictor()
    ab = abp.forward([{"path": p} for p in iso_path])
    pq = np.array([float(o.get("PQ", 0.0)) for o in ab])

    rows = []
    for ci, cfg in enumerate(CONFIGS):
        ii = [i for i in range(len(iso)) if iso_cfg[i] == ci]
        si = [i for i in range(len(seqs)) if seq_cfg[i] == ci]
        material = float(np.mean([matlog[i].argmax() == args.materials.index(iso_mat[i]) for i in ii]))
        not_music = float(np.mean([cat[i].argmax() not in MUSIC_IDX for i in ii]))
        quality = float(np.mean(pq[ii])) / 10.0
        footstep_seq = float(np.mean([seq_foot[i] for i in si]))
        var = variation[ci]
        axes = {"footstep_seq": footstep_seq, "material": material, "quality": quality,
                "not_music": not_music, "variation": var}
        composite = sum(REWARD_WEIGHTS[k] * axes[k] for k in REWARD_WEIGHTS)
        rows.append((composite, cfg["name"], axes, dict(cfg)))

    rows.sort(key=lambda r: -r[0])
    print(f"\n=== OPTIMIZER ROUND {args.round}  (reward weights {REWARD_WEIGHTS}) ===")
    print(f"{'config':<14}{'REWARD':>8}{'foot_seq':>9}{'material':>9}{'quality':>8}{'not_mus':>8}{'varied':>8}")
    print("-" * 64)
    for comp, name, ax, _ in rows:
        print(f"{name:<14}{comp:8.3f}{ax['footstep_seq']*100:8.0f}%{ax['material']*100:8.0f}%"
              f"{ax['quality']*100:7.0f}%{ax['not_music']*100:7.0f}%{ax['variation']*100:7.0f}%")
    best = rows[0]
    out = {"round": args.round, "best_name": best[1], "reward": best[0], "axes": best[2], "config": best[3]}
    with open(os.path.join(tmp, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWINNER: {best[1]}  reward {best[0]:.3f}  -> data/_opt/best_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
