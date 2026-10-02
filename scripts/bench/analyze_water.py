import json
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

SAGE = r"D:\Python\SAGE"
GATE = os.path.join(SAGE, "data", "_gate")
LISTEN = os.path.join(GATE, "listen", "water_3steps_then_8steps.wav")
TIER = os.path.join(GATE, "tier_water")
TIER_JSON = os.path.join(SAGE, "scripts", "bench", "out_tier_probe.json")
ESC = os.path.join(SAGE, "data", "esc50", "water_drops")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out_analyze_water.json")
KEYS = ["splash-water", "footstep-water", "impact-water", "land-water"]
STEPS = [3, 4, 8]
LAYOUT = [(k, v) for k in KEYS for v in (0, 1)]
GAP_IN, GAP_OUT, CLIP_S = 0.35, 0.9, 0.6
FADE_S = 0.05

STATIC_FLAT_LT = float(os.environ.get("STATIC_FLAT_LT", "0.15"))
STATIC_RISE = float(os.environ.get("STATIC_RISE", "8.0"))


def load(p):
    sr, a = wavfile.read(p)
    a = a.astype(np.float64)
    if a.ndim > 1:
        a = a.mean(1)
    if np.abs(a).max() > 1.5:
        a = a / 32768.0
    return a, sr


def frames(x, n, hop):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    m = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(m)[:, None]
    return x[idx]


def rms_env(x, sr, win_ms, hop_ms):
    n = max(1, int(win_ms * sr / 1000))
    h = max(1, int(hop_ms * sr / 1000))
    f = frames(x, n, h)
    return np.sqrt(np.mean(f * f, axis=1) + 1e-20), h


def flatness_frames(x, sr, lo=50.0, hi=16000.0):
    n = int(0.02 * sr)
    h = int(0.01 * sr)
    f = frames(x, n, h) * np.hanning(n)[None, :]
    p = np.abs(np.fft.rfft(f, axis=1)) ** 2
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    b = (fr >= lo) & (fr <= hi)
    p = p[:, b] + 1e-20
    fl = np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)
    e = np.sqrt(np.mean(f * f, axis=1))
    return fl, e, h, n


def db(a):
    return float(20 * np.log10(max(a, 1e-12)))


def metrics(x, sr):
    out = {}
    n = len(x)
    fl, e, h, fn = flatness_frames(x, sr)
    act = e >= e.max() * 10 ** (-40 / 20)
    out["flat_med"] = float(np.median(fl[act]))
    w = int(0.2 * sr)
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    s = int(np.argmax(c[w:] - c[:-w])) if n > w else 0
    starts = np.arange(len(fl)) * h
    inwin = (starts >= s) & (starts + fn <= s + w)
    if not inwin.any():
        inwin = act
    out["flat_200"] = float(np.median(fl[inwin]))
    out["loud200_start_ms"] = s / sr * 1000.0
    f3 = x[: int(0.3 * sr)]
    out["crest300"] = db(np.abs(f3).max() / (np.sqrt(np.mean(f3 * f3)) + 1e-12))
    aw = max(1, int(0.001 * sr))
    ae = np.convolve(np.abs(f3), np.ones(aw) / aw, mode="same")
    pk = int(np.argmax(ae))
    a10 = int(np.argmax(ae[: pk + 1] >= 0.1 * ae[pk]))
    a90 = int(np.argmax(ae[: pk + 1] >= 0.9 * ae[pk]))
    out["attack_ms"] = (a90 - a10) / sr * 1000.0
    e5, h5 = rms_env(x, sr, 5, 1)
    p5 = int(np.argmax(e5))
    lo5 = e5[max(0, p5 - 25): p5 + 1].min()
    out["onset_rise_db"] = db(e5[p5] / lo5)
    d5 = 20 * np.log10(e5 + 1e-12)
    out["onset_slope_db_per_ms"] = float(np.max(d5[5:] - d5[:-5]) / 5.0)
    fr, P = welch(x, sr, nperseg=2048, noverlap=1536)
    tot = P.sum() + 1e-30
    out["centroid_hz"] = float((fr * P).sum() / tot)
    out["low250"] = float(P[fr < 250].sum() / tot)
    out["low500"] = float(P[fr < 500].sum() / tot)
    out["high4k"] = float(P[fr >= 4000].sum() / tot)
    out["high8k"] = float(P[fr >= 8000].sum() / tot)
    e10, h10 = rms_env(x, sr, 10, 1)
    p10 = int(np.argmax(e10))
    peak = e10[p10]
    out["peak_ms"] = p10 * h10 / sr * 1000.0 + 5.0
    fade0 = int((n - FADE_S * sr - int(0.01 * sr)) / h10)
    for lv in (20, 30):
        thr = peak * 10 ** (-lv / 20)
        seg = e10[p10:fade0]
        hit = np.nonzero(seg < thr)[0]
        out["t%d_ms" % lv] = float(hit[0] * h10 / sr * 1000.0) if len(hit) else None
        above = np.nonzero(seg >= thr)[0]
        out["t%d_last_ms" % lv] = float((above[-1] + 1) * h10 / sr * 1000.0) if len(above) else 0.0
        out["above%d_ms" % lv] = float(len(above) * h10 / sr * 1000.0)
    out["window_ms"] = float((fade0 - p10) * h10 / sr * 1000.0)
    out["peak_abs"] = float(np.abs(x).max())
    fs0 = n - int(FADE_S * sr)
    pk0 = p10 * h10
    en = x[pk0:fs0] ** 2
    edc = np.cumsum(en[::-1])[::-1]
    edc = 10 * np.log10(edc / (edc[0] + 1e-30) + 1e-30)
    for lv in (20, 30):
        k = np.nonzero(edc <= -lv)[0]
        out["edc%d_ms" % lv] = float(k[0] / sr * 1000.0) if len(k) else None
    ts = x[min(n, pk0 + int(0.15 * sr)): fs0]
    if len(ts) >= 1024:
        tf, tp = welch(ts, sr, nperseg=min(2048, len(ts)), noverlap=min(1536, len(ts) * 3 // 4))
        tb = tf >= 50
        out["tail_flat"] = float(np.exp(np.mean(np.log(tp[tb] + 1e-30))) / np.mean(tp[tb] + 1e-30))
        out["tail_lvl_db"] = db(np.sqrt(np.mean(ts * ts)) / peak)
    else:
        out["tail_flat"], out["tail_lvl_db"] = 0.0, -120.0
    out["tail_hiss"] = bool(out["tail_flat"] >= STATIC_FLAT_LT and out["tail_lvl_db"] > -40.0)
    t0, t1 = n - int((FADE_S + 0.1) * sr), n - int(FADE_S * sr)
    tail = x[t0:t1]
    out["tail_db"] = db(np.sqrt(np.mean(tail * tail)) / peak)
    q = x[n - int(FADE_S * sr) - int(0.05 * sr): n - int(FADE_S * sr)]
    out["prefade_db"] = db(np.sqrt(np.mean(q * q)) / peak)
    lt = np.nonzero(fr >= 50)[0]
    out["flat_longterm"] = float(np.exp(np.mean(np.log(P[lt] + 1e-30))) / np.mean(P[lt] + 1e-30))
    return out


def is_static(m, flat_lt=None, rise=None):
    flat_lt = STATIC_FLAT_LT if flat_lt is None else flat_lt
    rise = STATIC_RISE if rise is None else rise
    return bool(m["flat_longterm"] >= flat_lt and m["onset_rise_db"] <= rise)


def locate_listen():
    lx, sr = load(LISTEN)
    srcs = {}
    for st in (3, 8):
        for k, v in LAYOUT:
            srcs[(k, v, st)] = load(os.path.join(TIER, "water_s%d" % st, "%s_%02d.wav" % (k, v)))[0]
    nclip = int(CLIP_S * sr)
    pair = 2 * CLIP_S + GAP_IN + GAP_OUT
    found = []
    for i, (k, v) in enumerate(LAYOUT):
        for j, st in enumerate((3, 8)):
            t = i * pair + j * (CLIP_S + GAP_IN)
            o = int(round(t * sr))
            best = None
            for d in range(-40, 41):
                a = lx[o + d: o + d + nclip]
                if len(a) < nclip:
                    continue
                for (kk, vv, ss), b in srcs.items():
                    g = float(np.dot(a, b) / (np.dot(b, b) + 1e-20))
                    r = float(np.sqrt(np.mean((a - g * b) ** 2)) / (np.sqrt(np.mean(a * a)) + 1e-20))
                    if best is None or r < best[0]:
                        best = (r, kk, vv, ss, d, g)
            found.append({"slot": len(found) + 1, "pair": i + 1, "expected": "%s v%d s%d" % (k, v, st),
                          "matched": "%s v%d s%d" % (best[1], best[2], best[3]), "resid": best[0],
                          "offset_samples": best[4], "gain": best[5], "key": best[1], "v": best[2],
                          "steps": best[3], "t_s": (o + best[4]) / sr,
                          "listen_metrics": metrics(lx[o + best[4]: o + best[4] + nclip], sr)})
    return found, sr


def tier_rows():
    ext = {}
    if os.path.exists(TIER_JSON):
        for r in json.load(open(TIER_JSON))["rows"]:
            if r["cfg"].startswith("water"):
                ext[(r["cfg"], r["key"].replace("|", "-"), int(r["v"]))] = r
    rows = []
    for st in STEPS:
        for k in KEYS:
            for v in range(10):
                p = os.path.join(TIER, "water_s%d" % st, "%s_%02d.wav" % (k, v))
                x, sr = load(p)
                m = metrics(x, sr)
                e = ext.get(("water_s%d" % st, k, v), {})
                rows.append({"steps": st, "key": k, "v": v, "path": p, **m, "static": is_static(m),
                             "PQ": e.get("PQ"), "CE": e.get("CE"), "p_water": e.get("p_water"),
                             "water_top": e.get("water_top")})
    return rows


def esc_rows():
    out = []
    if not os.path.isdir(ESC):
        return out
    for f in sorted(os.listdir(ESC)):
        if not f.endswith(".wav"):
            continue
        x, sr = load(os.path.join(ESC, f))
        if np.abs(x).max() <= 0:
            continue
        w = int(0.6 * sr)
        c = np.concatenate([[0.0], np.cumsum(x * x)])
        e = np.convolve(x * x, np.ones(int(0.005 * sr)) / int(0.005 * sr), mode="same")
        pk = int(np.argmax(e))
        s = max(0, min(len(x) - w, pk - int(0.03 * sr)))
        out.append({"file": f, "sr": sr, **metrics(x[s: s + w], sr)})
    return out


def pct(a, q):
    a = [v for v in a if v is not None]
    return float(np.percentile(a, q)) if a else float("nan")


def summarize(rs, label):
    def col(f):
        return [r[f] for r in rs]
    n = len(rs)
    t30_cens = sum(1 for r in rs if r["t30_ms"] is None)
    t20_cens = sum(1 for r in rs if r["t20_ms"] is None)
    s = {"label": label, "n": n,
         "static": sum(1 for r in rs if r.get("static")),
         "flat_med_median": pct(col("flat_med"), 50), "flat_200_median": pct(col("flat_200"), 50),
         "crest300_median": pct(col("crest300"), 50), "onset_rise_median": pct(col("onset_rise_db"), 50),
         "centroid_median": pct(col("centroid_hz"), 50), "low250_median": pct(col("low250"), 50),
         "low500_median": pct(col("low500"), 50), "high4k_median": pct(col("high4k"), 50),
         "t20_median_uncensored": pct(col("t20_ms"), 50), "t30_median_uncensored": pct(col("t30_ms"), 50),
         "t20_censored": t20_cens, "t30_censored": t30_cens,
         "tail_db_median": pct(col("tail_db"), 50),
         "tail_gt_m20": sum(1 for r in rs if r["tail_db"] > -20), "tail_gt_m30": sum(1 for r in rs if r["tail_db"] > -30),
         "low250_lt_5pct": sum(1 for r in rs if r["low250"] < 0.05), "low500_lt_10pct": sum(1 for r in rs if r["low500"] < 0.10),
         "prefade_gt_m30": sum(1 for r in rs if r["prefade_db"] > -30), "late_peak": sum(1 for r in rs if r["peak_ms"] > 100),
         "flat_lt_median": pct(col("flat_longterm"), 50), "high8k_median": pct(col("high8k"), 50),
         "t30_last_median": pct(col("t30_last_ms"), 50), "t20_last_median": pct(col("t20_last_ms"), 50),
         "above30_median": pct(col("above30_ms"), 50), "edc30_median": pct(col("edc30_ms"), 50),
         "edc20_median": pct(col("edc20_ms"), 50), "tail_hiss": sum(1 for r in rs if r["tail_hiss"]),
         "at_peak_ceiling": sum(1 for r in rs if r["peak_abs"] > 0.965)}
    pq = [r["PQ"] for r in rs if r.get("PQ") is not None]
    if pq:
        s["PQ_mean"] = float(np.mean(pq))
    return s


def mcnemar(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def fmt(v, f="%.2f"):
    return "  -  " if v is None else f % v


def main():
    found, sr = locate_listen()
    rows = tier_rows()
    by = {(r["steps"], r["key"], r["v"]): r for r in rows}
    print("LISTENING FILE %s (sr %d)" % (LISTEN, sr))
    print("%3s %-24s %-24s %7s %6s %6s %6s %6s %6s %7s %6s %6s %6s %6s %6s %6s %6s %5s %5s" % (
        "#", "expected", "matched", "resid", "flatLT", "hi8k", "flat2", "crest", "rise", "cent", "lo250", "lo500",
        "hi4k", "t20", "t30", "tail", "pkms", "PQ", "stat"))
    for f in found:
        r = by[(f["steps"], f["key"], f["v"])]
        f["static"] = r["static"]
        f["PQ"] = r["PQ"]
        print("%3d %-24s %-24s %7.4f %6.3f %6.3f %6.3f %6.1f %6.1f %7.0f %6.3f %6.3f %6.3f %6s %6s %6.1f %6.0f %5s %5s" % (
            f["slot"], f["expected"], f["matched"], f["resid"], r["flat_longterm"], r["high8k"], r["flat_200"],
            r["crest300"], r["onset_rise_db"], r["centroid_hz"], r["low250"], r["low500"], r["high4k"],
            fmt(r["t20_ms"], "%.0f"), fmt(r["t30_ms"], "%.0f"), r["tail_db"], r["peak_ms"], fmt(r["PQ"]),
            "YES" if r["static"] else ""))
    print()
    print("ALL 120 TIER CLIPS")
    print("%-3s %-15s %2s %6s %6s %6s %6s %6s %7s %6s %6s %6s %6s %6s %6s %6s %5s %6s %4s" % (
        "st", "key", "v", "flatM", "flat2", "flatLT", "crest", "rise", "cent", "lo250", "lo500", "hi4k",
        "t20", "t30", "tail", "pkms", "PQ", "pwat", "stat"))
    for r in sorted(rows, key=lambda r: (r["key"], r["v"], r["steps"])):
        print("%-3d %-15s %2d %6.3f %6.3f %6.3f %6.1f %6.1f %7.0f %6.3f %6.3f %6.3f %6s %6s %6.1f %6.0f %5s %6s %4s" % (
            r["steps"], r["key"], r["v"], r["flat_med"], r["flat_200"], r["flat_longterm"], r["crest300"],
            r["onset_rise_db"], r["centroid_hz"], r["low250"], r["low500"], r["high4k"], fmt(r["t20_ms"], "%.0f"),
            fmt(r["t30_ms"], "%.0f"), r["tail_db"], r["peak_ms"], fmt(r["PQ"]), fmt(r["p_water"], "%.3f"),
            "YES" if r["static"] else ""))
    esc = esc_rows()
    summ = [summarize([r for r in rows if r["steps"] == st], "s%d" % st) for st in STEPS]
    for k in KEYS:
        summ.append(summarize([r for r in rows if r["key"] == k], k))
    summ.append(summarize(rows, "all120"))
    if esc:
        summ.append(summarize(esc, "esc50_water_drops"))
    print()
    print("SUMMARY (static rule: long-term flatness >= %.2f and onset rise <= %.1f dB)" % (STATIC_FLAT_LT, STATIC_RISE))
    hdr = ["label", "n", "static", "flat_lt_median", "high8k_median", "flat_200_median", "crest300_median", "onset_rise_median", "centroid_median",
           "low250_median", "low500_median", "high4k_median", "t20_median_uncensored", "t20_censored",
           "t30_median_uncensored", "t30_censored", "tail_db_median", "tail_gt_m20", "tail_gt_m30",
           "low250_lt_5pct", "prefade_gt_m30", "late_peak", "PQ_mean"]
    short = ["label", "n", "stat", "flatLT", "hi8k", "flat2", "crest", "rise", "cent", "lo250", "lo500", "hi4k", "t20",
             "t20c", "t30", "t30c", "tail", "t>-20", "t>-30", "lo<5%", "pf>-30", "late", "PQ"]
    print(" ".join("%-17s" % short[0] if i == 0 else "%7s" % s for i, s in enumerate(short)))
    for s in summ:
        cells = []
        for i, h in enumerate(hdr):
            v = s.get(h)
            if i == 0:
                cells.append("%-17s" % v)
            elif isinstance(v, int):
                cells.append("%7d" % v)
            elif v is None:
                cells.append("%7s" % "-")
            elif h.startswith("centroid") or h.startswith("t2") or h.startswith("t3"):
                cells.append("%7.0f" % v)
            else:
                cells.append("%7.3f" % v)
        print(" ".join(cells))
    print()
    print("PAIRED STATIC FLAG (same key+seed), b = A static and B clean, c = A clean and B static")
    pairs = {}
    for a, b in ((3, 8), (3, 4), (4, 8)):
        ks = [(k, v) for k in KEYS for v in range(10)]
        fa = np.array([by[(a, k, v)]["static"] for k, v in ks])
        fb = np.array([by[(b, k, v)]["static"] for k, v in ks])
        bb, cc = int((fa & ~fb).sum()), int((~fa & fb).sum())
        pairs["s%d_vs_s%d" % (a, b)] = {"static_a": int(fa.sum()), "static_b": int(fb.sum()), "b": bb, "c": cc,
                                         "p": mcnemar(bb, cc)}
        print("  s%d vs s%d: static %d vs %d of 40, discordant b=%d c=%d, McNemar p=%.3f" % (
            a, b, fa.sum(), fb.sum(), bb, cc, mcnemar(bb, cc)))
    print()
    print("CORRELATIONS over 120 clips (Spearman)")
    from scipy.stats import spearmanr
    pq = [r for r in rows if r["PQ"] is not None]
    for f in ("flat_200", "flat_med", "crest300", "onset_rise_db", "tail_db", "low250", "high4k", "centroid_hz"):
        rho, p = spearmanr([r[f] for r in pq], [r["PQ"] for r in pq])
        rho2, p2 = spearmanr([r[f] for r in pq], [r["p_water"] for r in pq])
        print("  %-14s vs PQ rho %+.2f (p %.3f)   vs p_water rho %+.2f (p %.3f)" % (f, rho, p, rho2, p2))
    print()
    print("TOP 8 BY LONG-TERM FLATNESS")
    sc = sorted(rows, key=lambda r: -r["flat_longterm"])
    for r in sc[:8]:
        print("  s%d %-15s v%d flatLT %.3f hi8k %.3f rise %.1f crest %.1f peak %.0f ms tail %.1f PQ %s pwat %s%s" % (
            r["steps"], r["key"], r["v"], r["flat_longterm"], r["high8k"], r["onset_rise_db"], r["crest300"],
            r["peak_ms"], r["tail_db"], fmt(r["PQ"]), fmt(r["p_water"], "%.3f"), "  STATIC" if r["static"] else ""))
    print()
    print("RULE SENSITIVITY: static count per step count over the threshold grid")
    grid = {}
    for fl in (0.10, 0.12, 0.15, 0.20, 0.25):
        for ri in (6.0, 8.0, 10.0, 12.0):
            c = {st: sum(1 for r in rows if r["steps"] == st and is_static(r, fl, ri)) for st in STEPS}
            grid["%.2f|%.0f" % (fl, ri)] = c
            print("  flatLT >= %.2f, rise <= %4.1f dB: s3 %d  s4 %d  s8 %d" % (fl, ri, c[3], c[4], c[8]))
    print()
    print("SINGLE-CRITERION COUNTS over 120 (s3/s4/s8)")
    single = {}
    for name, f in (("flatLT>=0.10", lambda r: r["flat_longterm"] >= 0.10), ("flatLT>=0.12", lambda r: r["flat_longterm"] >= 0.12),
                    ("rise<=8", lambda r: r["onset_rise_db"] <= 8.0), ("rise<=12", lambda r: r["onset_rise_db"] <= 12.0)):
        c = [sum(1 for r in rows if r["steps"] == st and f(r)) for st in STEPS]
        single[name] = c
        print("  %-13s s3 %d  s4 %d  s8 %d" % (name, c[0], c[1], c[2]))
    print("TAIL HISS (flat tail after peak+150 ms, flatness >= %.2f, level > -40 dB re peak): %s" % (
        STATIC_FLAT_LT, ", ".join("s%d %d" % (st, sum(1 for r in rows if r["steps"] == st and r["tail_hiss"])) for st in STEPS)))
    for r in sorted([r for r in rows if r["tail_hiss"]], key=lambda r: -r["tail_flat"]):
        print("  s%d %-15s v%d tail_flat %.3f tail_lvl %.1f dB rise %.1f static %s" % (
            r["steps"], r["key"], r["v"], r["tail_flat"], r["tail_lvl_db"], r["onset_rise_db"], r["static"]))
    print()
    print("CONDENSED: how much of the usable window (peak to fade start) holds sound at or above -30 dB re the 10 ms peak")
    use_first, use_last, occ = [], [], []
    for r in rows:
        span = r["window_ms"]
        use_first.append(r["t30_ms"] / span if r["t30_ms"] is not None else 1.0)
        use_last.append(r["t30_last_ms"] / span)
        occ.append(r["above30_ms"] / span)
    cond = {}
    for name, u in (("first_dip", use_first), ("last_above", use_last), ("occupancy", occ)):
        u = np.array(u)
        cond[name] = {"median": float(np.median(u)), "p25": float(np.percentile(u, 25)), "p75": float(np.percentile(u, 75)),
                      "gt80": int((u > 0.8).sum())}
        print("  %-11s median %.2f, p25 %.2f, p75 %.2f, clips > 80%% of the window: %d of %d" % (
            name, np.median(u), np.percentile(u, 25), np.percentile(u, 75), int((u > 0.8).sum()), len(u)))
    print("  first_dip = first 10 ms frame below -30 dB; last_above = last frame at or above -30 dB; occupancy = share of frames at or above -30 dB")
    print("DECAY medians (ms after the peak): first dip -20/-30, last above -20/-30, frames above -30, Schroeder EDC -20/-30")
    for s in summ:
        print("  %-17s %5.0f %5.0f   %5.0f %5.0f   %5.0f   %5.0f %5.0f   tail_hiss %d" % (
            s["label"], s["t20_median_uncensored"], s["t30_median_uncensored"], s["t20_last_median"], s["t30_last_median"],
            s["above30_median"], s["edc20_median"], s["edc30_median"], s["tail_hiss"]))
    print("PEAK CEILING: clips whose peak sits at normalize_loudness's 0.97 ceiling (RMS target overridden): " +
          ", ".join("%s %d/%d" % (s["label"], s["at_peak_ceiling"], s["n"]) for s in summ if s["label"] != "esc50_water_drops"))
    json.dump({"rule": {"flat_longterm_ge": STATIC_FLAT_LT, "onset_rise_db_le": STATIC_RISE}, "listen": found,
               "rows": rows, "summary": summ, "paired_static": pairs, "rule_grid": grid, "single_criterion": single,
               "condensed": cond, "esc50_water_drops": esc},
              open(OUT, "w"), indent=1)
    print("saved -> %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
