import _bootstrap

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch
import torchaudio.functional as AF


def load_wav(path):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(1)
    return a.astype(np.float32), int(sr)


def resample(a, src, dst):
    if src == dst:
        return a.astype(np.float32)
    t = torch.from_numpy(np.ascontiguousarray(a)).float()
    return AF.resample(t, src, dst).numpy().astype(np.float32)


def crest_factor(a):
    rms = float(np.sqrt(np.mean(a ** 2))) + 1e-9
    return float(np.max(np.abs(a))) / rms


def is_dead(a):
    return (not np.isfinite(a).all()) or float(np.max(np.abs(a))) < 1e-4


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/sampler-bench")
    ap.add_argument("--judges", nargs="+", default=["clap", "audiobox"])
    args = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows = list(csv.DictReader(open(os.path.join(args.dir, "manifest.csv"))))
    rows = [r for r in rows if not r["error"]]
    prompts = sorted(set(r["prompt"] for r in rows))
    p_idx = {p: i for i, p in enumerate(prompts)}

    clips48, clipsraw, paths, truelabel, keys, lat = [], [], [], [], [], []
    for r in rows:
        wav = os.path.join(args.dir, r["wav"])
        a, sr = load_wav(wav)
        clipsraw.append(a)
        clips48.append(resample(a, sr, 48000))
        paths.append(wav)
        truelabel.append(p_idx[r["prompt"]])
        keys.append((r["sampler"], int(r["steps"])))
        lat.append(float(r["latency_ms"]) if r["latency_ms"] else float("nan"))

    dead = np.array([is_dead(a) for a in clipsraw])
    crest = np.array([0.0 if d else crest_factor(a) for a, d in zip(clipsraw, dead)])

    clap_hit = np.zeros(len(rows))
    if "clap" in args.judges:
        from transformers import ClapModel, ClapProcessor
        m = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
        pr = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        tfeat = pr(text=prompts, return_tensors="pt", padding=True)
        for i0 in range(0, len(clips48), 16):
            batch = clips48[i0:i0 + 16]
            afeat = pr(audio=batch, sampling_rate=48000, return_tensors="pt", padding=True)
            inp = {k: v.to(dev) for k, v in {**afeat, **tfeat}.items()}
            with torch.no_grad():
                logits = m(**inp).logits_per_audio.float().cpu().numpy()
            pred = logits.argmax(1)
            for j, p in enumerate(pred):
                clap_hit[i0 + j] = float(p == truelabel[i0 + j])
        del m
        torch.cuda.empty_cache()

    pq = np.full(len(rows), np.nan)
    if "audiobox" in args.judges:
        try:
            from audiobox_aesthetics.infer import initialize_predictor
            pred = initialize_predictor()
            outs = []
            for i0 in range(0, len(paths), 8):
                outs.extend(pred.forward([{"path": p} for p in paths[i0:i0 + 8]]))
            key = "PQ" if "PQ" in outs[0] else list(outs[0].keys())[0]
            pq = np.array([float(o.get("PQ", o.get(key))) for o in outs])
        except Exception as e:
            print("(audiobox skipped:", str(e)[:80], ")")

    agg = defaultdict(lambda: dict(n=0, hit=0.0, crest=0.0, pq=[], lat=[], dead=0))
    for i, k in enumerate(keys):
        a = agg[k]
        a["n"] += 1
        a["hit"] += clap_hit[i]
        a["crest"] += crest[i]
        a["pq"].append(pq[i])
        a["lat"].append(lat[i])
        a["dead"] += int(dead[i])

    order = ([("euler", s) for s in (8, 4, 3, 2, 1)]
             + [("dpmpp", s) for s in (4, 3, 2)]
             + [("rk4", s) for s in (4, 3, 2)]
             + [("pingpong", s) for s in (4, 3, 2)])
    ref = agg[("euler", 8)]
    ref_crest = ref["crest"] / max(1, ref["n"])
    ref_pq = np.nanmean(ref["pq"])
    ref_hit = ref["hit"] / max(1, ref["n"])

    lines = []
    hdr = f"{'sampler':9s}{'steps':>6}{'lat_ms':>9}{'CLAP_match':>12}{'crest_rel':>11}{'PQ':>7}{'dead':>6}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    print("\n=== euler@8 reference: CLAP_match {:.0f}%  crest {:.2f}  PQ {:.2f} ===".format(
        ref_hit * 100, ref_crest, ref_pq))
    for k in order:
        a = agg.get(k)
        if not a or a["n"] == 0:
            continue
        hit = a["hit"] / a["n"] * 100
        cr = a["crest"] / a["n"]
        cr_rel = cr / ref_crest if ref_crest else 0.0
        pqm = np.nanmean(a["pq"])
        latm = np.nanmean(a["lat"])
        lines.append(f"{k[0]:9s}{k[1]:>6}{latm:>9.0f}{hit:>11.0f}%{cr_rel:>11.2f}{pqm:>7.2f}{a['dead']:>6}")
    table = "\n".join(lines)
    print("\n" + table)

    with open(os.path.join(args.dir, "results.md"), "w") as f:
        f.write("# Sampler-swap results (objective metrics)\n\n")
        f.write(f"Reference = euler@8: CLAP_match {ref_hit*100:.0f}%, crest {ref_crest:.2f}, PQ {ref_pq:.2f}.\n\n")
        f.write("CLAP_match = rank-1 of correct prompt among %d (chance %.0f%%). " % (len(prompts), 100.0 / len(prompts)))
        f.write("crest_rel = crest factor / euler@8 (transient sharpness, 1.0=as sharp as 8-step). ")
        f.write("PQ = Audiobox production quality 1-10. dead = silent/NaN clips.\n\n```\n")
        f.write(table + "\n```\n")
    print("\nwrote", os.path.join(args.dir, "results.md"))


if __name__ == "__main__":
    main()
