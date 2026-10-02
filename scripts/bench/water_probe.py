import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import gc
import json
import time
from math import sqrt

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import welch

from sarge.generators.stable_audio import _extract, build_prompt
from sarge.generators.stable_audio3 import StableAudio3Generator
from sarge.loudness import PEAK_CEILING, normalize_loudness, rms, target_for
from sarge.types import AudioClip
from validate_panel import _resample
from gate_quality import crest_db
from ab_clap import clap_scores, mcnemar, softmax
from tier_probe import WATER_KEYS, WATER_TEXTS

ROOT = os.path.dirname(os.path.dirname(BENCH))
LENGTHS = {
    "L0_1.0_0.6": (1.0, 0.6),
    "L1_1.5_1.0": (1.5, 1.0),
    "L2_2.0_1.5": (2.0, 1.5),
}
WORDINGS = {
    "current": "a {lead}{event} in {material}, wet splashing water with droplets and bubbling, dry, isolated sound effect",
    "no_dry": "a {lead}{event} in {material}, wet splashing water with droplets and bubbling, isolated sound effect",
    "depth": "a {lead}{event} in {material}, a deep resonant splash with a full low body and a natural decaying tail, "
             "wet splashing water with droplets and bubbling, isolated sound effect",
    "onset_depth": "a {lead}{event} in {material}, a deep resonant splash with a full low body and a natural decaying tail, "
                   "one sharp clear onset, isolated sound effect",
}
REF = "L0_1.0_0.6|current"
CLAP_TEXTS = list(WATER_TEXTS) + ["white noise static"]
STATIC_FLAT_LT = 0.15
STATIC_RISE_DB = 8.0
FADE_S = 0.05
CROP_S = 0.6
TAKES = [("splash", 0), ("land", 1)]
OUT_DIR = os.path.join("data", "_gate", "water_depth")
TIER_S3 = os.path.join("data", "_gate", "tier_water", "water_s3")


def text_for(wname, key):
    lead = "" if key.intensity in (None, "", "normal") else "%s " % key.intensity
    return WORDINGS[wname].format(lead=lead, event=key.event_type, material=key.material)


def db(a):
    return float(20 * np.log10(max(float(a), 1e-12)))


def env(x, sr, win_ms, hop_ms=1.0):
    n = max(1, int(win_ms * sr / 1000))
    h = max(1, int(hop_ms * sr / 1000))
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    f = np.lib.stride_tricks.sliding_window_view(x, n)[::h]
    return np.sqrt(np.mean(f * f, axis=1) + 1e-20), h


def lt_flat(x, sr):
    fr, P = welch(x, sr, nperseg=min(2048, len(x)), noverlap=min(1536, len(x) * 3 // 4))
    b = fr >= 50
    return float(np.exp(np.mean(np.log(P[b] + 1e-30))) / np.mean(P[b] + 1e-30)), fr, P


def metrics(x, sr):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    fo = int(FADE_S * sr)
    out = {}
    flat, fr, P = lt_flat(x, sr)
    tot = P.sum() + 1e-30
    out["flat_lt"] = flat
    out["low250"] = float(P[fr < 250].sum() / tot)
    out["low500"] = float(P[fr < 500].sum() / tot)
    out["high8k"] = float(P[fr >= 8000].sum() / tot)
    out["centroid"] = float((fr * P).sum() / tot)
    e5, _h5 = env(x, sr, 5)
    p5 = int(np.argmax(e5))
    out["rise_db"] = db(e5[p5] / e5[max(0, p5 - 25): p5 + 1].min())
    out["static"] = int(out["flat_lt"] >= STATIC_FLAT_LT and out["rise_db"] <= STATIC_RISE_DB)
    out["crest"] = crest_db(x[: int(0.3 * sr)])
    out["peak_dbfs"] = db(np.abs(x).max())
    e10, h10 = env(x, sr, 10)
    p10 = int(np.argmax(e10))
    peak = e10[p10]
    out["peak_ms"] = p10 * h10 / sr * 1000.0 + 5.0
    fade0 = max(p10 + 1, (n - fo - int(0.01 * sr)) // h10)
    for lv in (20, 30):
        hit = np.nonzero(e10[p10:fade0] < peak * 10 ** (-lv / 20))[0]
        out["t%d_ms" % lv] = float(hit[0] * h10 / sr * 1000.0) if len(hit) else None
    above = np.nonzero(e10[p10:fade0] >= peak * 10 ** (-30 / 20))[0]
    out["t30_last_ms"] = float((above[-1] + 1) * h10 / sr * 1000.0) if len(above) else 0.0
    out["above30_ms"] = float(len(above) * h10 / sr * 1000.0)
    en = x[p10 * h10: n - fo] ** 2
    edc = np.cumsum(en[::-1])[::-1] if len(en) else np.ones(1)
    k30 = np.nonzero(edc <= edc[0] * 1e-3)[0]
    out["edc30_ms"] = float(k30[0] / sr * 1000.0) if len(k30) else None
    tail = x[max(0, n - fo - int(0.1 * sr)): n - fo]
    out["tail_db"] = db(np.sqrt(np.mean(tail * tail)) / peak) if len(tail) else -120.0
    out["ring"] = int(out["tail_db"] > -30.0)
    post = e10[min(len(e10) - 1, p10 + int(0.15 * sr / h10)): fade0]
    out["second_db"] = db(post.max() / peak) if len(post) else -120.0
    out["second_hit"] = int(out["second_db"] > -10.0)
    seg = x[min(n, p10 * h10 + int(0.15 * sr)): n - fo]
    if len(seg) >= 1024:
        sf, _f, _p = lt_flat(seg, sr)
        lvl = db(np.sqrt(np.mean(seg * seg)) / peak)
    else:
        sf, lvl = 0.0, -120.0
    out["tail_flat"] = sf
    out["tail_hiss"] = int(sf >= STATIC_FLAT_LT and lvl > -40.0)
    return out


def load_wav(p):
    return AudioClip.from_wav(p).samples


def head_level(x, sr, target):
    r = rms(x[: int(CROP_S * sr)])
    a = x * np.float32(target / r) if r > 1e-9 else np.asarray(x, dtype=np.float32)
    pk = float(np.max(np.abs(a))) if len(a) else 0.0
    if pk > PEAK_CEILING:
        a = a * np.float32(PEAK_CEILING / pk)
    return a.astype(np.float32)


def paired(rows, a_cfg, b_cfg, f, binary):
    A = {(r["key"], r["v"]): r for r in rows if r["cfg"] == a_cfg}
    B = {(r["key"], r["v"]): r for r in rows if r["cfg"] == b_cfg}
    ks = [k for k in B if k in A and A[k].get(f) is not None and B[k].get(f) is not None]
    a = np.array([A[k][f] for k in ks], dtype=float)
    b = np.array([B[k][f] for k in ks], dtype=float)
    if binary:
        g = int(((a == 1) & (b == 0)).sum())
        l = int(((a == 0) & (b == 1)).sum())
        return {"n": len(ks), "rate": float(a.mean()) if len(a) else float("nan"), "gain": g, "loss": l,
                "p": mcnemar(g, l)}
    d = a - b
    se = float(d.std(ddof=1) / sqrt(len(d))) if len(d) > 1 else 0.0
    return {"n": len(ks), "mean": float(a.mean()) if len(a) else float("nan"),
            "median": float(np.median(a)) if len(a) else float("nan"),
            "delta": float(d.mean()) if len(d) else float("nan"), "se": se}


def med(rs, f):
    v = [r[f] for r in rs if r.get(f) is not None]
    return float(np.median(v)) if v else float("nan")


def cens(rs, f):
    return sum(1 for r in rs if r.get(f) is None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=10)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join("scripts", "bench", "out_water_probe.json"))
    ap.add_argument("--listen", default=os.path.join("data", "_gate", "listen", "water_depth.wav"))
    args = ap.parse_args()
    os.chdir(ROOT)

    for k in WATER_KEYS:
        a, b = text_for("current", k), build_prompt(k.event_type, k.material, k.intensity)
        if a != b:
            raise SystemExit("current wording drifted from build_prompt for %s:\n  %s\n  %s" % (k, a, b))

    g = StableAudio3Generator()
    g.ensure_ready()
    g.steps = args.steps
    g.detail_steps = args.steps
    sr = int(g.sample_rate)
    for k in WATER_KEYS:
        if g._steps_for(k) != args.steps:
            raise SystemExit("steps for %s = %d, expected %d" % (k, g._steps_for(k), args.steps))
    print("model %s fp16=%s cfg=%.1f steps=%d sampler=%s pad=%.1fs sr=%d fast=%s" % (
        g.model_name, g.model_half, g.cfg_scale, args.steps, g.sampler_type, g.duration_padding_s, sr,
        g._fast is not None))
    for wname in WORDINGS:
        print("  wording %-12s %s" % (wname, text_for(wname, WATER_KEYS[0])))
    g.gen_duration_s, g.duration_s = LENGTHS["L0_1.0_0.6"]
    g.generate(WATER_KEYS[0], variant=0)

    rows, full48, crop48, cfgs, lat = [], [], [], [], {}
    for lname, (gd, dd) in LENGTHS.items():
        g.gen_duration_s, g.duration_s = gd, dd
        for wname in WORDINGS:
            cfg = "%s|%s" % (lname, wname)
            cfgs.append(cfg)
            for k in WATER_KEYS:
                if wname == "current":
                    g.prompts.pop(str(k), None)
                else:
                    g.prompts[str(k)] = text_for(wname, k)
                if g._prompt(k) != text_for(wname, k):
                    raise SystemExit("prompt mismatch for %s %s" % (cfg, k))
            for k in WATER_KEYS:
                g.generate(k, variant=0)
            d = os.path.join(OUT_DIR, "%s_%s" % (lname, wname))
            dc = d + "_crop06"
            os.makedirs(d, exist_ok=True)
            ts = []
            for k in WATER_KEYS:
                for v in range(args.variants):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    c = g.generate(k, variant=v)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) * 1000.0
                    ts.append(ms)
                    name = "%s-%s_%02d.wav" % (k.event_type, k.material, v)
                    p = os.path.join(d, name)
                    c.to_wav(p)
                    x = c.samples.astype(np.float32)
                    if dd > CROP_S + 1e-9:
                        crop = normalize_loudness(_extract(x, sr, CROP_S), target_rms=target_for(k.intensity))
                        pc = os.path.join(dc, name)
                        AudioClip(crop.astype(np.float32), sr).to_wav(pc)
                    else:
                        crop, pc = x, p
                    m = metrics(x, sr)
                    mc = metrics(crop, sr)
                    rows.append({"cfg": cfg, "len": lname, "gen_s": gd, "dur_s": dd, "word": wname,
                                 "key": "%s|%s" % (k.event_type, k.material), "event": k.event_type,
                                 "intensity": k.intensity, "v": v, "path": p, "crop_path": pc, "ms": ms,
                                 "n_samples": int(len(x)), "prompt": g._prompt(k), **m,
                                 **{"c_" + a: b for a, b in mc.items()}})
                    full48.append(_resample(x, sr, 48000))
                    crop48.append(_resample(np.asarray(crop, dtype=np.float32), sr, 48000))
            lat[cfg] = float(np.median(ts))
            rs = [r for r in rows if r["cfg"] == cfg]
            print("generated %-26s median %6.1f ms  static %d/%d  ring %d  low500 %.3f" % (
                cfg, lat[cfg], sum(r["static"] for r in rs), len(rs), sum(r["ring"] for r in rs), med(rs, "low500")))

    det = []
    for r in rows:
        if r["cfg"] != REF:
            continue
        q = os.path.join(TIER_S3, os.path.basename(r["path"]))
        if args.steps == 3 and os.path.exists(q):
            a, b = load_wav(r["path"]), load_wav(q)
            n = min(len(a), len(b))
            det.append(float(np.max(np.abs(a[:n] - b[:n]))) if n else 1.0)
    if det:
        print("determinism %s vs %s: %d clips, max|diff| %.5f, identical %d" % (
            REF, TIER_S3, len(det), max(det), sum(x == 0.0 for x in det)))
    fs = g.fast_stats()
    print("fast stats %s" % json.dumps(fs))
    del g
    gc.collect()
    torch.cuda.empty_cache()

    sc = clap_scores({"full": full48, "crop": crop48}, {"water": CLAP_TEXTS}, "cuda", 32)
    iw, ist = CLAP_TEXTS.index("water splashing"), CLAP_TEXTS.index("white noise static")
    for pre, sname in (("", "full"), ("c_", "crop")):
        lg = sc[sname]["water"]
        pr = softmax(lg)
        top = lg.argmax(1)
        for i, r in enumerate(rows):
            r[pre + "p_water"] = float(pr[i, iw])
            r[pre + "p_static"] = float(pr[i, ist])
            r[pre + "water_top"] = int(top[i] == iw)
            r[pre + "static_top"] = int(top[i] == ist)
            r[pre + "clap_top"] = CLAP_TEXTS[int(top[i])]

    from audiobox_aesthetics.infer import initialize_predictor
    pred = initialize_predictor()
    jobs = [(r, "PQ", "CE", r["path"]) for r in rows]
    jobs += [(r, "c_PQ", "c_CE", r["crop_path"]) for r in rows if r["crop_path"] != r["path"]]
    for i in range(0, len(jobs), 40):
        chunk = jobs[i: i + 40]
        for (r, fq, fc, _p), o in zip(chunk, pred.forward([{"path": j[3]} for j in chunk])):
            r[fq] = float(o["PQ"])
            r[fc] = float(o["CE"])
    for r in rows:
        if r["crop_path"] == r["path"]:
            r["c_PQ"], r["c_CE"] = r["PQ"], r["CE"]

    W = 150
    print()
    print("=" * W)
    print(" WATER DEPTH PROBE  %d keys x %d seeds per config, %d steps, paired vs %s" % (
        len(WATER_KEYS), args.variants, args.steps, REF))
    print(" static = long-term flatness >= %.2f and onset rise <= %.1f dB (part-1 rule); hiss = flat tail after peak+150 ms"
          % (STATIC_FLAT_LT, STATIC_RISE_DB))
    print("=" * W)
    fb = lambda t: "%3.0f%% +%d/-%d p%.2f" % (100 * t["rate"], t["gain"], t["loss"], t["p"])
    fd = lambda t, f="%.2f": "%s (%s+-%s)" % (f % t["mean"], ("%+" + f[1:]) % t["delta"], f % t["se"])
    print("%-26s %7s %18s %18s %18s %18s %6s %8s %8s %20s" % (
        "config", "ms", "static", "clap water top1", "clap static top1", "static crop06", "hiss", "p_water",
        "p_static", "PQ (delta)"))
    summary = {}
    for cfg in cfgs:
        rs = [r for r in rows if r["cfg"] == cfg]
        s = {"ms": lat[cfg], "n": len(rs)}
        for f in ("static", "water_top", "static_top", "c_static", "tail_hiss", "ring", "second_hit", "c_water_top"):
            s[f] = paired(rows, cfg, REF, f, True)
        for f in ("PQ", "c_PQ", "p_water", "c_p_water", "p_static", "low250", "low500", "c_low250", "c_low500",
                  "centroid", "tail_db", "c_tail_db", "crest", "peak_dbfs", "rise_db", "flat_lt",
                  "t30_last_ms", "above30_ms", "c_t30_last_ms", "c_above30_ms"):
            s[f] = paired(rows, cfg, REF, f, False)
        s["t20_median"], s["t30_median"] = med(rs, "t20_ms"), med(rs, "t30_ms")
        s["t20_censored"], s["t30_censored"] = cens(rs, "t20_ms"), cens(rs, "t30_ms")
        s["c_t30_median"], s["c_t30_censored"] = med(rs, "c_t30_ms"), cens(rs, "c_t30_ms")
        s["PQ_by_key"] = {k: float(np.mean([r["PQ"] for r in rs if r["key"] == k])) for k in sorted({r["key"] for r in rs})}
        s["low500_by_key"] = {k: float(np.median([r["low500"] for r in rs if r["key"] == k])) for k in sorted({r["key"] for r in rs})}
        s["clap_top_hist"] = {t: sum(1 for r in rs if r["clap_top"] == t) for t in CLAP_TEXTS}
        summary[cfg] = s
        print("%-26s %7.1f %18s %18s %18s %18s %6d %8.3f %8.3f %20s" % (
            cfg, s["ms"], fb(s["static"]), fb(s["water_top"]), fb(s["static_top"]), fb(s["c_static"]),
            sum(r["tail_hiss"] for r in rs), s["p_water"]["mean"], s["p_static"]["mean"], fd(s["PQ"])))
    print("-" * W)
    print(" SHAPE (medians; ring = tail 100 ms before the fade above -30 dB re peak; 2nd = later hit within 10 dB of the peak;"
          " t30 = first dip below -30 dB, t30last = last 10 ms frame at or above -30 dB, abv30 = ms at or above -30 dB,"
          " edc30 = Schroeder decay to -30 dB)")
    print("%-26s %7s %7s %7s %7s %10s %7s %6s %6s %6s %8s %7s %7s %6s %7s %8s" % (
        "config", "low250", "low500", "cent", "t20", "t30(cens)", "t30last", "abv30", "edc30", "ring", "tail_db", "crest",
        "rise", "2nd", "flatLT", "peakdBFS"))
    for cfg in cfgs:
        rs = [r for r in rows if r["cfg"] == cfg]
        s = summary[cfg]
        print("%-26s %7.3f %7.3f %7.0f %7.0f %6.0f(%2d) %7.0f %6.0f %6.0f %6d %8.1f %7.1f %7.1f %6d %7.3f %8.1f" % (
            cfg, med(rs, "low250"), med(rs, "low500"), med(rs, "centroid"), s["t20_median"], s["t30_median"],
            s["t30_censored"], med(rs, "t30_last_ms"), med(rs, "above30_ms"), med(rs, "edc30_ms"),
            sum(r["ring"] for r in rs), med(rs, "tail_db"), med(rs, "crest"), med(rs, "rise_db"),
            sum(r["second_hit"] for r in rs), med(rs, "flat_lt"), med(rs, "peak_dbfs")))
    print("-" * W)
    print(" GENERATION LENGTH ALONE: the same clips re-extracted to 0.6 s (crop06), so longer seconds_total is"
          " separated from the longer window")
    print("%-26s %7s %7s %10s %9s %8s %8s %8s %18s %8s" % ("config", "c_lo250", "c_lo500", "c_t30(c)", "c_t30last",
                                                            "c_abv30", "c_tail", "c_PQ", "c_clap water top1", "c_pwat"))
    for cfg in cfgs:
        rs = [r for r in rows if r["cfg"] == cfg]
        s = summary[cfg]
        print("%-26s %7.3f %7.3f %6.0f(%2d) %9.0f %8.0f %8.1f %8.2f %18s %8.3f" % (
            cfg, med(rs, "c_low250"), med(rs, "c_low500"), s["c_t30_median"], s["c_t30_censored"],
            med(rs, "c_t30_last_ms"), med(rs, "c_above30_ms"), med(rs, "c_tail_db"), s["c_PQ"]["mean"],
            fb(s["c_water_top"]), s["c_p_water"]["mean"]))
    print("-" * W)
    print(" PER KEY  PQ mean / low500 median")
    keys = sorted({r["key"] for r in rows})
    print("%-26s " % "config" + " ".join("%22s" % k for k in keys))
    for cfg in cfgs:
        s = summary[cfg]
        print("%-26s " % cfg + " ".join("%13.2f / %6.3f" % (s["PQ_by_key"][k], s["low500_by_key"][k]) for k in keys))
    print("-" * W)
    print(" LADDER CONTRASTS (each changes one thing; A vs B paired on key+seed)")
    lad = []
    for lname in LENGTHS:
        for a, b in (("no_dry", "current"), ("depth", "no_dry"), ("onset_depth", "depth")):
            lad.append(("%s|%s" % (lname, a), "%s|%s" % (lname, b)))
    for wname in WORDINGS:
        for lname in list(LENGTHS)[1:]:
            lad.append(("%s|%s" % (lname, wname), "L0_1.0_0.6|%s" % wname))
    contrasts = []
    print("%-48s %18s %20s %20s %20s" % ("A vs B", "static A", "PQ", "low500", "p_water"))
    for a, b in lad:
        st = paired(rows, a, b, "static", True)
        pq = paired(rows, a, b, "PQ", False)
        lo = paired(rows, a, b, "low500", False)
        pw = paired(rows, a, b, "p_water", False)
        contrasts.append({"a": a, "b": b, "static": st, "PQ": pq, "low500": lo, "p_water": pw})
        print("%-48s %18s %20s %20s %20s" % ("%s vs %s" % (a, b), fb(st), fd(pq), fd(lo, "%.3f"), fd(pw, "%.3f")))
    print("-" * W)
    pl = {}
    for lname in LENGTHS:
        v = [r["ms"] for r in rows if r["len"] == lname]
        pl[lname] = float(np.median(v))
    print(" LATENCY through g.generate, median ms per length (pooled over wordings): " +
          "  ".join("%s %.1f" % (k, v) for k, v in pl.items()) + "   (tier_probe water_s3 at 1.0 s measured 81.0 ms)")
    print(" CLAP top-1 histogram per config:")
    for cfg in cfgs:
        h = summary[cfg]["clap_top_hist"]
        print("  %-26s %s" % (cfg, ", ".join("%s %d" % (t, c) for t, c in h.items() if c)))
    print("=" * W)

    seq, order, t = [], [], 0.0
    ev = {k.event_type: k for k in WATER_KEYS}
    for cfg in cfgs:
        lname, wname = cfg.split("|")
        order.append({"cfg": cfg, "start_s": round(t, 2),
                      "takes": ["%s|%s v%d" % (ev[e].event_type, ev[e].material, v) for e, v in TAKES]})
        for e, v in TAKES:
            k = ev[e]
            p = os.path.join(OUT_DIR, "%s_%s" % (lname, wname), "%s-%s_%02d.wav" % (k.event_type, k.material, v))
            a = head_level(load_wav(p), sr, target_for(k.intensity))
            seq += [a, np.zeros(int(0.35 * sr), np.float32)]
            t += len(a) / sr + 0.35
        seq.append(np.zeros(int(1.0 * sr), np.float32))
        t += 1.0
    x = np.concatenate(seq)
    x = x / max(1e-9, float(np.abs(x).max())) * 0.9
    os.makedirs(os.path.dirname(args.listen), exist_ok=True)
    wavfile.write(args.listen, sr, (x * 32767).astype(np.int16))
    print("listening file %s, 2 takes per config (%s), each take levelled on its first %.1f s RMS (the proposed shipping"
          " rule, so a longer tail no longer raises the hit), then one global gain:" % (
              args.listen, ", ".join("%s v%d" % (e, v) for e, v in TAKES), CROP_S))
    for i, o in enumerate(order):
        print("  %2d  %6.2f s  %s" % (i + 1, o["start_s"], o["cfg"]))
    json.dump({"lengths": LENGTHS, "wordings": WORDINGS, "steps": args.steps, "variants": args.variants,
               "static_rule": {"flat_lt_ge": STATIC_FLAT_LT, "rise_db_le": STATIC_RISE_DB},
               "clap_texts": CLAP_TEXTS, "ref": REF, "latency_ms": lat, "latency_by_length": pl,
               "summary": summary, "contrasts": contrasts, "listen_order": order, "listen_level": "head_rms_%.1fs" % CROP_S,
               "fast_stats": fs,
               "determinism": det, "rows": rows},
              open(args.out, "w"), indent=1)
    print("saved -> %s" % args.out)


if __name__ == "__main__":
    main()
