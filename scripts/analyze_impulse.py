import csv
import glob
import os
import sys

import numpy as np

DEFAULT_DIR = os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")),
                           "AppData", "LocalLow", "DefaultCompany", "PSfxUnityTest")
PATTERN = "sarge_impulse_probe_*.csv"
BINS = ("soft", "normal", "hard")
SOFT_SPEED = 1.5
HARD_SPEED = 6.0
MIN_SPEED = 0.6
MIN_HITS = 8
MIN_CV = 0.25
MIN_SPAN = 2.0
MIN_LEVELS = 6
LEVEL_STEP = 0.05
MIN_BIN_FRAC = 0.10
TRACKS_SPEED_R = 0.95


def find_csvs(args):
    paths = []
    for a in args or [DEFAULT_DIR]:
        if os.path.isdir(a):
            paths += sorted(glob.glob(os.path.join(a, PATTERN)))
        elif os.path.isfile(a):
            paths.append(a)
        else:
            print("skipping %s (not found)" % a)
    return paths


def load(paths):
    hits = {}
    zero = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                try:
                    obj = row["object"]
                    imp = float(row["impulse_Ns"])
                    rel = float(row["rel_speed"])
                except (KeyError, TypeError, ValueError):
                    continue
                if imp <= 0:
                    zero[obj] = zero.get(obj, 0) + 1
                    continue
                hits.setdefault(obj, []).append((imp, rel, row.get("other", "")))
    return hits, zero


def levels(x):
    return len(np.unique(np.round(np.log(x) / np.log1p(LEVEL_STEP))))


def summary(x):
    p5, p95 = np.percentile(x, [5, 95])
    return {"n": len(x), "min": float(x.min()), "median": float(np.median(x)),
            "max": float(x.max()), "distinct": len(np.unique(np.round(x, 3))),
            "levels": levels(x), "cv": float(x.std() / max(x.mean(), 1e-12)),
            "span": float(p95 / max(p5, 1e-12))}


def impulse_bins(x):
    lo, hi = np.log(np.percentile(x, [5, 95]))
    half = 0.5 * np.log(MIN_SPAN)
    mid = 0.5 * (lo + hi)
    lo, hi = min(lo, mid - half), max(hi, mid + half)
    edges = np.exp(np.linspace(lo, hi, len(BINS) + 1))[1:-1]
    return np.digitize(x, edges), tuple(edges)


def speed_bins(v):
    return np.where(v >= HARD_SPEED, 2, np.where(v <= SOFT_SPEED, 0, 1))


def share_line(idx):
    n = max(len(idx), 1)
    return "   ".join("%s %d (%.0f%%)" % (b, int((idx == i).sum()), 100.0 * (idx == i).sum() / n)
                       for i, b in enumerate(BINS))


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    paths = find_csvs(args)
    if not paths:
        where = ", ".join(args) if args else DEFAULT_DIR
        print("no %s under %s" % (PATTERN, where))
        print()
        print("in Unity: open WalkArena, run Sarge > Add Impulse Probe To Bouncers,")
        print("press Play, kick the bouncers with K a few times, stop, then re-run")
        print("this script. Pass a folder or csv paths to read from elsewhere.")
        return 1

    hits, zero = load(paths)
    print("=== %d probe csv file(s) ===" % len(paths))
    for p in paths:
        print("  %s" % p)
    if not hits:
        print("no collisions with impulse > 0 in those files; play longer")
        return 1

    print()
    print("=== per-object collision impulse ===")
    print("%-22s %5s %9s %9s %9s %8s %6s %6s %7s" % (
        "object", "hits", "min Ns", "median", "max Ns", "distinct", "levels", "CV", "p95/p5"))
    print("-" * 90)
    per_cv = []
    for obj in sorted(hits):
        x = np.array([h[0] for h in hits[obj]], dtype=np.float64)
        s = summary(x)
        per_cv.append(s["cv"])
        print("%-22s %5d %9.3f %9.3f %9.3f %8d %6d %6.3f %7.2f" % (
            obj[:22], s["n"], s["min"], s["median"], s["max"], s["distinct"],
            s["levels"], s["cv"], s["span"]))
        if zero.get(obj):
            print("%-22s %5d zero-impulse contacts left out" % ("", zero[obj]))

    imp = np.array([h[0] for v in hits.values() for h in v], dtype=np.float64)
    rel = np.array([h[1] for v in hits.values() for h in v], dtype=np.float64)
    s = summary(imp)
    print()
    print("=== pooled impulse from %d collisions across %d object(s) ===" % (s["n"], len(hits)))
    qs = [0, 5, 25, 50, 75, 95, 100]
    for q, v in zip(qs, np.percentile(imp, qs)):
        print("  p%-3d %10.3f Ns" % (q, v))
    print("  p95/p5 span factor %.2f" % s["span"])
    print("  coefficient of variation %.3f" % s["cv"])
    print("  distinct values %d of %d (0.001 Ns)" % (s["distinct"], s["n"]))
    print("  levels %d at %.0f%% resolution" % (s["levels"], 100 * LEVEL_STEP))

    idx, edges = impulse_bins(imp)
    print()
    print("=== soft/normal/hard from impulse, log-equal bins over p5..p95 (at least %.0fx wide) ===" % MIN_SPAN)
    print("  edges %s Ns" % ", ".join("%.3f" % e for e in edges))
    print("  " + share_line(idx))

    audible = rel >= MIN_SPEED
    print()
    print("=== same hits as SargeAutoCapture bins them, by relative speed ===")
    print("  soft <= %.1f m/s, hard >= %.1f m/s, silent below %.1f m/s (%d hits)" % (
        SOFT_SPEED, HARD_SPEED, MIN_SPEED, int((~audible).sum())))
    print("  " + share_line(speed_bins(rel[audible])))

    both = (imp > 0) & (rel > 0)
    r = float("nan")
    if both.sum() >= 3 and np.std(np.log(rel[both])) > 1e-9 and np.std(np.log(imp[both])) > 1e-9:
        r = float(np.corrcoef(np.log(imp[both]), np.log(rel[both]))[0, 1])
        print("  log impulse vs log relative speed r = %.3f, median impulse/speed %.3f kg" % (
            r, float(np.median(imp[both] / rel[both]))))

    reasons = []
    if s["n"] < MIN_HITS:
        reasons.append("only %d collisions, need %d" % (s["n"], MIN_HITS))
    if s["cv"] < MIN_CV:
        reasons.append("CV %.3f is under %.2f, hits barely differ" % (s["cv"], MIN_CV))
    if s["span"] < MIN_SPAN:
        reasons.append("p95/p5 span %.2f is under %.1f" % (s["span"], MIN_SPAN))
    if s["levels"] < MIN_LEVELS:
        reasons.append("%d distinguishable levels, need %d" % (s["levels"], MIN_LEVELS))
    for i, b in enumerate(BINS):
        frac = float((idx == i).mean())
        if frac < MIN_BIN_FRAC:
            reasons.append("%s bin holds %.0f%% of hits" % (b, 100 * frac))

    print()
    print("=" * 70)
    if reasons:
        print("VERDICT: impulse is NOT continuous enough to drive soft/normal/hard here")
        for x in reasons:
            print("  - " + x)
    else:
        print("VERDICT: impulse is continuous enough to drive soft/normal/hard")
        print("  CV %.3f, p95/p5 %.2f, %d levels, every bin above %.0f%%" % (
            s["cv"], s["span"], s["levels"], 100 * MIN_BIN_FRAC))
    med_cv = float(np.median(per_cv))
    if len(hits) > 1 and med_cv < MIN_CV <= s["cv"]:
        print("  note: median per-object CV is %.3f, under %.2f. Much of the pooled" % (med_cv, MIN_CV))
        print("  spread comes from differences between objects, so one object varies less.")
    if med_cv < MIN_CV:
        print("  hint: elastic bouncers repeat the same bounce. Press K during Play")
        print("  to kick them to new heights, then re-run.")
    if r == r and r >= TRACKS_SPEED_R:
        print("  note: impulse tracks relative speed (r %.3f). SargeAutoCapture's" % r)
        print("  speed bins already carry the same signal for these bodies.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
