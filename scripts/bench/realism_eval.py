import os
import sys

BENCH = os.path.join(os.getcwd(), "scripts", "bench")
if not os.path.isfile(os.path.join(BENCH, "_bootstrap.py")):
    BENCH = r"D:\Python\SAGE\scripts\bench"
sys.path.insert(0, BENCH)

import _bootstrap

import vocab_eval as ve

import argparse
import csv
import hashlib
import json
import math
import re
import time
import warnings
from collections import Counter, defaultdict

import numpy as np

import realism_feats as rf
import realism_refs as rr
import realism_stats as rs

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = ve.ROOT
OUT = os.path.join(ROOT, "data", "_gate", "judges")
PREREG = os.path.join(OUT, "prereg.json")
VALIDATED = os.path.join(OUT, "validated.json")
EAR = os.path.join(OUT, "ear_verdicts.json")
RATINGS = os.path.join(OUT, "ear_ratings.json")
CORE = ["realism_eval.py", "realism_refs.py", "realism_feats.py", "realism_stats.py"]
NAN = float("nan")
SEEDS_A, SEEDS_B = [0, 1, 2, 3], [4, 5, 6, 7]
TH = {
    "pool_min_clips": rr.MIN_CLIPS, "pool_min_sources": rr.MIN_SOURCES, "quality_clipped_frac": rr.Q_CLIP,
    "quality_rolloff_hz": rr.Q_ROLLOFF, "feature_nan_drop": 0.10, "soft_clip": rs.SOFT, "lat_floor": rs.FLOOR,
    "j7_floor_rel": 0.05, "j8d_eps": 0.01, "j9_avail": rr.J9_AVAIL, "null_draws": 500, "null_n": 8,
    "S0_q05": 0.60, "S0_key_frac": 0.5,
    "S1_rise": 0.80, "S1_spearman": 0.80, "S1_delta": 0.5, "S1_sparkle_mad": 0.5, "S1_controls_q": 95,
    "S1c_d4": 0.70, "S1c_d6_q": 95,
    "S2_nui_delta": 0.2, "S2_nui_iqr": 0.25, "S2_flaw_delta": -0.2,
    "S3_auc": 0.65, "S3_p": 0.05, "S3_mc": 2000,
    "S4_spearman": 0.70, "S4_splits": 50, "S4_ci_width": 0.40,
    "J2_min_two": 0.5, "J2_split_points": 0.25,
    "J8d_gate_auc": 0.75, "J8d_C": 0.1, "J8d_folds": 5, "pca_k": 16,
    "J6_p": 0.05, "J6_bacc": 0.6, "J6_converge": 0.6, "J6_shuffles": 500, "J6_sub": 6, "water_collapse": 0.6,
    "C1_correct": {"gun": 2, "glass": 3, "water": 2, "footstep": 2}, "C1_p": 0.05, "C1_ratio": 1.15,
    "C1_delta": 0.33, "perm_mc": 10000,
    "E_z": -1.0, "E2_points": 0.25, "E5": 0.6,
    "outcome_C1_families": 3, "outcome_E_frac": 0.70, "outcome_E_min": 5,
    "exp_ratio": 1.15, "exp_p": 0.05, "exp_delta": 0.33,
    "verdict_C_frac": 0.75, "gate_cons": 0.5,
    "V6_within_key": 0.40, "V6_within_model": 0.20, "slots": {"A": 2, "B": 5},
    "budget_s": {"features": 480.0, "panns": 480.0, "stats": 240.0},
}
JUDGES = {
    "J1": {"kind": "z", "bounded": True, "win": "0.6", "fams": ("gun", "glass", "footstep", "other"), "targets": "d2"},
    "J2": {"kind": "z", "bounded": False, "win": "0.6", "fams": ("gun",), "targets": "d2"},
    "J3": {"kind": "z", "bounded": False, "win": "0.6", "fams": ("gun", "glass", "footstep", "water", "other"),
           "targets": "d2&d5&d25"},
    "J4": {"kind": "z", "bounded": True, "win": "0.6", "fams": ("gun", "glass", "footstep", "water", "other"),
           "targets": "d1"},
    "J5": {"kind": "z", "bounded": False, "win": "0.6", "fams": ("glass", "footstep", "water"), "targets": "d2|d5&d25"},
    "J7": {"kind": "knn", "bounded": False, "win": "0.6", "fams": ("gun", "glass", "footstep", "water", "other"),
           "targets": "d4"},
    "J8d": {"kind": "clf", "bounded": True, "win": "0.6", "fams": ("gun", "glass", "footstep", "water", "other"),
            "targets": "d4gate"},
    "J9": {"kind": "z", "bounded": False, "win": "1.0", "fams": ("glass", "water", "other"), "targets": "d8"},
}
ZJ = ("J1", "J2", "J3", "J4", "J5")
C1_PAIRS = {
    "gun": [("medium_rec", "small_prod", "shoot|pistol"), ("medium_rec", "small_prod", "shoot|rifle")],
    "glass": [("woosh_flow", "small_prod", "break|window"), ("woosh_flow", "medium_rec", "break|window"),
              ("woosh_flow", "small_prod", "bullet_impact|glass"), ("woosh_flow", "medium_rec", "bullet_impact|glass")],
    "water": [("woosh_flow", "small_prod", "splash|water"), ("woosh_flow", "medium_rec", "splash|water")],
    "footstep": [("woosh_flow", "small_prod", "footstep|gravel"), ("woosh_flow", "small_prod", "footstep|stone"),
                 ("woosh_flow", "small_prod", "footstep|wood")],
}
E_TEXT = {
    "E1": "pistol crack: small_prod median z <= -1.0 on plr50_db and e10_100, medium_rec smaller |z| on both",
    "E2": "rifle clicks: J2 preconditions hold and small_prod two-click rate >= 25 points below real",
    "E3": "glass highs: break|window small_prod median z <= -1.0 on sparkle and hfk_hi, woosh_flow smaller |z| on sparkle",
    "E4": "gravel grain: small_prod median z <= -1.0 on hfk_lo and flux_kurt, woosh_flow smaller |z| on both",
    "E5": "convergence: R_J6(small_prod) < 0.6 and R_J6(woosh_flow) > R_J6(small_prod)",
    "E6": "water detail: splash|water small_prod median z <= -1.0 on mf_mid_b and mf_hi_b, woosh_flow smaller |z| on both",
    "E7": "small_prod surface order: combined R wood >= stone >= gravel",
    "E8": "small_prod loses: mean combined R over its scored core pool keys ranks in the bottom 3 of 9",
    "E9": "glass is small_prod's worst key: break|window lowest combined R of pistol, rifle, window, gravel, stone, wood, "
          "splash",
    "E10": "water footsteps: woosh_flow artifact-only R above small_prod on footstep|water and no water-collapse flag",
}
C3_SET = ["gun_pair", "E7", "E9"]
CORE_KEYS = ["shoot|pistol", "shoot|rifle", "break|window", "footstep|gravel", "footstep|stone", "footstep|wood",
             "splash|water"]
KNOWN_BEFORE = [
    "data/_gate/bakeoff/bandwidth.json (Woosh near-empty above 16 kHz, SA3 content to 22 kHz)",
    "9-model vocab_eval identity cons, shape in-range and Audiobox PQ per key",
    "latent_stats: generated latents carry about half the within-clip dynamics of encoded real audio",
    "decode facts: SAME-S halves high-band kurtosis (gravel 84 -> 24, stone 534 -> 99) and adds a +21 dB comb at "
    "k x 172 Hz",
    "manifest clipped_frac counts: c3gd_pistol 138/140, c3gd_rifle 105/140, esc_glass_breaking 15/34, fs_* 0",
    "the user's ear verdicts listed in ear_verdicts.json",
    "real-only iterations real_dev and real_dev2 (pools, S0, S1, S2 nuisance, J2 preconditions, J6 validity, J8d d4 "
    "gate) before this freeze",
    "dev dry runs (data/_gate/judges/dev) scored the anchor raw, SAME-L and SAME-S round-trip and replay folders "
    "(real-derived audio, no bake-off clip) under bake-off model names, with forced selections, to test code paths",
]
INTERPRETATIONS = [
    "footstep gate (chain step 7) applies to footstep-family keys only (gravel, stone, wood, metal); footstep|water "
    "is a water-family key and is cut like splash|water so the water-collapse test compares like with like",
    "J1 onset t0 walks back from the 2 kHz high-passed 1 ms envelope peak to the start of the contiguous run above 10% "
    "of the peak (dc.attack logic without its 20 ms bound)",
    "e10_100 and pre_db are energy (sum of squares) ratios in dB; pre_db floors at -60 dB and an onset with under "
    "32 samples of preroll counts as -60 dB",
    "rise_ms is log(clip(10-90% rise, 0.25 ms, 30 ms)); fwhm_ms is log of the full width at half maximum of the "
    "broadband positive-flux peak (n 256, hop 32)",
    "J2 click2_db is the second-strongest click relative to the strongest (amplitude dB); gap_ms the gap between them",
    "sparkle peaks are linear-bin local maxima (n 2048) at least 12 dB above a +-1/6-octave median computed on a "
    "48-bin-per-octave grid; +-1 linear bin around k x 172.27 Hz (every 8th bin) is excluded; the 60% persistence rule "
    "needs at least 5 active frames",
    "comb_any periodicity spectrum = mean residual at k*f0 minus at (k+0.5)*f0 over 2-15 kHz harmonics, f0 60-400 Hz "
    "in 0.5 Hz steps; prominence = peak minus median (dB)",
    "J5 envelope kurtosis is the central kurtosis of the compressed envelope (log); modulation fractions are log "
    "power fractions of the mean-removed envelope spectrum over all non-DC power",
    "J9 decay_db_s is log of the -5 to -25 dB Schroeder slope in dB/s",
    "J7 lat floor is 0.05 x the median real kNN distance (the z judges use the absolute 0.05 floor on RMS z')",
    "J8d inputs are standardised inside each fold; derived clips of a real recording (damage, nuisance, d6) are "
    "scored by the fold model that held out that recording's source, everything else by the full model",
    "S1 compares each damage dose with the same clip re-finished through chain steps 3-6 (clean2); pre25/pre35 compare "
    "with the original reference clip; S1 control rule: the median lat of the hiss set and of the silence set each "
    "exceed the real q95",
    "S1 (dose rules, d4, controls) reads raw lat for bounded judges (J1, J4, J8d) because S1 asks whether lat rises; "
    "lat+ = max(lat, 0) is how a bounded judge enters R, so S0, S2, S3, S4 and every score read lat+",
    "S2 nuisance and flaw effects are unpaired Cliff's delta (positive = transformed clip more atypical); flaws use "
    "seeds 0-3 of every model on the core pool keys plus punch",
    "S4 ranks models by the mean over the family's non-proxy pool keys of L_J; NaN Spearman (all ties) fails",
    "collapse distance uses soft-clipped two-sided z against the full pool over the selected z-judge features (J9 "
    "excluded), NaN set to 0; with no z judge selected it uses PANNs cosine distance",
    "C1 seed halves recompute the collapse penalty with a 4-clip null",
    "a pool whose quality filter would leave fewer than 20 clips or 4 sources keeps every clip",
    "J9 availability counts the final pool after the quality filter",
    "every S-test pools a family's primary (non-proxy) pool keys; S0 passes when q05 >= 0.60 on at least half of them",
]


def file_md5(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def core_md5():
    return {f: file_md5(os.path.join(BENCH, f)) for f in CORE}


def slug(s):
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(s))


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return [jsonable(x) for x in o.tolist()]
    if isinstance(o, (set, tuple)):
        return [jsonable(x) for x in o]
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, list):
        return [jsonable(x) for x in o]
    return o


def dump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(jsonable(obj), fh, indent=1, default=str)
    os.replace(tmp, p)


def fnum(x, d=2):
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else ("%.*f" % (d, x))


def judge_feats(J, fam):
    if J == "J1":
        return [f for f in rf.J1 if f != "low_share" or fam in ("gun", "other")]
    if J == "J2":
        return list(rf.J2)
    if J == "J3":
        return [f for f in rf.J3 if f != "sparkle" or fam in ("glass", "water")]
    if J == "J4":
        return list(rf.J4)
    if J == "J5":
        return list(rf.J5)
    if J == "J9":
        return [f for f in rf.J9 if f in ("decay_db_s", "late_early_db") or fam in ("glass", "water")]
    return []


class Ctx:
    def __init__(self, args):
        self.args = args
        self.t0 = time.perf_counter()
        self.stage = defaultdict(float)
        self.budget = {k: v * args.budget_scale for k, v in TH["budget_s"].items()}
        self.caches = {}
        self.panns = None
        self.emb = args.embed == "panns"
        self.fmd5 = rf.feats_md5()
        self.spec = {}
        self.boot = ve.Boot(args.boot, args.rng)
        self.rng = np.random.default_rng(args.rng)
        self.log = []

    def say(self, msg):
        line = "[%6.1f s] %s" % (time.perf_counter() - self.t0, msg)
        print(line, flush=True)
        self.log.append(line)

    def check(self, stage):
        if self.stage[stage] > self.budget[stage]:
            raise SystemExit("stage %s overran its budget: %.0f s > %.0f s" % (stage, self.stage[stage],
                                                                               self.budget[stage]))

    def cache(self, kind, group):
        p = os.path.join(self.args.out_dir, "cache", "%s_%s.npy" % (kind, slug(group)))
        if p not in self.caches:
            self.caches[p] = rf.Cache(p, "v1")
        return self.caches[p]

    def save_caches(self):
        for c in self.caches.values():
            c.save()


def spark(ctx, row, si, thr):
    fc = ctx.cache("feat", row["_group"])
    name = "sparkle" if row["win"] == "0.6" else "tail_sparkle"
    ca = row["f"].get("comb_any", NAN)
    f0 = row["f"].get("comb_f0") if (thr is not None and np.isfinite(thr) and ca == ca and ca > thr) else None
    stag = row["tag"] + "|sp|" + ("%.1f" % f0 if f0 else "-")
    v = fc.get(stag)
    if v is None:
        v = rf.sparkle(si, f0)
        fc.put(stag, v)
    row["f"][name] = v
    row["f"]["spark_f0x"] = f0 if f0 else NAN


def flush(ctx, pc, pend):
    if not pend:
        return
    t = time.perf_counter()
    if ctx.panns is None:
        ctx.panns = rf.Panns(ctx.args.threads)
    E = ctx.panns.embed([a for _r, a, _t in pend])
    for (row, _a, ptag), e in zip(pend, E):
        row["emb"] = e.astype(np.float32)
        pc.put(ptag, row["emb"])
    pend.clear()
    ctx.stage["panns"] += time.perf_counter() - t
    ctx.check("panns")


def compute(ctx, specs, group, thr=None, defer=False):
    if not specs:
        return []
    fc, pc = ctx.cache("feat", group), ctx.cache("panns", group)
    rows, pend = [], []
    for sp in specs:
        t = time.perf_counter()
        ctx.spec[sp.uid] = sp
        o = sp.make()
        a = o["ana"]
        h = rf.audio_md5(a)
        tag = "|".join([h, ctx.fmd5, rr.CHAIN_VERSION, sp.fam, str(int(sp.foot)), str(int(sp.wet)), sp.win])
        f = fc.get(tag)
        if f is None:
            f = rf.clip_features(a, o["desc"], o["dc"], sp.fam, sp.foot, sp.wet, sp.win)
            fc.put(tag, f)
        row = {"uid": sp.uid, "kind": sp.kind, "key": sp.key, "pool": sp.pool, "cluster": sp.cluster, "win": sp.win,
               "meta": sp.meta, "f": dict(f), "tag": tag, "emb": None, "_group": group}
        if sp.wet:
            si = rf.spark_input(a, sp.foot, sp.win)
            if defer:
                row["_si"] = si
            else:
                spark(ctx, row, si, thr.get((sp.pool, sp.win)) if isinstance(thr, dict) else thr)
        ctx.stage["features"] += time.perf_counter() - t
        if sp.win == "0.6" and ctx.emb:
            ptag = "|".join([h, ctx.fmd5, rr.CHAIN_VERSION])
            e = pc.get(ptag)
            if e is None:
                pend.append((row, a, ptag))
            else:
                row["emb"] = e
        rows.append(row)
        if len(pend) >= 128:
            flush(ctx, pc, pend)
        ctx.check("features")
    flush(ctx, pc, pend)
    fc.save()
    pc.save()
    return rows


def q95(rows, name):
    v = np.array([r["f"].get(name, NAN) for r in rows], float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, 95)) if len(v) else NAN


def quality(rows, spec):
    n0 = len(rows)
    info = {"before": n0, "sources_before": len({r["cluster"] for r in rows}), "applied": False}
    if not spec.get("filter"):
        info.update({"after": n0, "rule": "none (gun pool, clipped by construction)"})
        return rows, info
    bad_c = [r for r in rows if r["meta"].get("clipped_frac", 0.0) > rr.Q_CLIP]
    bad_r = [r for r in rows if not (r["f"].get("rolloff99", NAN) >= rr.Q_ROLLOFF)]
    drop = {r["uid"] for r in bad_c} | {r["uid"] for r in bad_r}
    keep = [r for r in rows if r["uid"] not in drop]
    ok = len(keep) >= rr.MIN_CLIPS and len({r["cluster"] for r in keep}) >= rr.MIN_SOURCES
    info.update({"dropped_clipped": len(bad_c), "dropped_rolloff": len(bad_r), "would_keep": len(keep),
                 "would_keep_sources": len({r["cluster"] for r in keep}), "applied": bool(ok)})
    out = keep if ok else rows
    info["after"] = len(out)
    info["sources_after"] = len({r["cluster"] for r in out})
    return out, info


def build_pools(ctx):
    man = rr.man_load(rr.HELDOUT)
    _m, sets, meta = ve.load_refs(rr.ns_refs(rr.HELDOUT), {})
    ctx.man = man
    ctx.refs_man = None
    out = {}
    for p, spec in rr.POOLS.items():
        if spec["src"] == "heldout":
            specs = rr.heldout_specs(p, man, sets, meta)
        elif spec["src"] == "cache_v1":
            specs = rr.footstep_specs(p)
        else:
            specs, ctx.refs_man = rr.refs_file_specs(p)
        rows = compute(ctx, specs, "real_" + p, defer=True)
        kept, qinfo = quality(rows, spec)
        valid = len(kept) >= rr.MIN_CLIPS and len({r["cluster"] for r in kept}) >= rr.MIN_SOURCES
        thr = q95(kept, "comb_any")
        for r in rows:
            if "_si" in r:
                spark(ctx, r, r.pop("_si"), thr)
        fam = spec["family"]
        avail, frac = rr.j9_available([ctx.spec[r["uid"]] for r in kept])
        j9 = bool(valid and avail and fam in JUDGES["J9"]["fams"])
        rows1, thr1 = [], NAN
        if j9:
            s1 = rr.w1_real_specs(p, [ctx.spec[r["uid"]] for r in kept])
            rows1 = compute(ctx, s1, "w1real_" + p, defer=True)
            thr1 = q95(rows1, "comb_any")
            for r in rows1:
                if "_si" in r:
                    spark(ctx, r, r.pop("_si"), thr1)
        ctx.cache("feat", "real_" + p).save()
        ctx.cache("feat", "w1real_" + p).save()
        clipped = sum(r["meta"].get("clipped_frac", 0.0) > 1e-4 for r in kept)
        out[p] = {"rows": kept, "all": rows, "valid": bool(valid), "quality": qinfo, "thr": thr, "rows1": rows1,
                  "thr1": thr1, "j9": j9, "j9_frac": frac, "family": fam,
                  "n": len(kept), "sources": len({r["cluster"] for r in kept}), "clipped": int(clipped),
                  "platforms": dict(Counter(r["meta"].get("platform") for r in kept)) if fam == "gun" else None}
        ctx.say("pool %-7s %3d clips / %2d sources (from %d)%s  quality %s  J9 %s (%.0f%% with 1 s after peak)  "
                "clipped %d" % (p, len(kept), out[p]["sources"], len(rows), "" if valid else "  BELOW MINIMUM",
                                "applied" if qinfo.get("applied") else "not applied (%s)" % qinfo.get("rule", "pool would "
                                "fall to %s clips / %s sources" % (qinfo.get("would_keep"), qinfo.get("would_keep_sources"))),
                                "yes" if j9 else "no", 100 * frac, clipped))
    return out


def primary_keys(pools, keys):
    prim = {}
    for k in keys:
        p = rr.KEYS[k]["pool"]
        if p and pools.get(p, {}).get("valid") and rr.KEYS[k]["status"] != "proxy" and p not in prim:
            prim[p] = k
    return prim


def derived_specs(ctx, pools, prim, mode):
    groups = defaultdict(list)
    for p, K in prim.items():
        P = pools[p]
        base = [ctx.spec[r["uid"]] for r in P["rows"]]
        sub = rr.spaced_by_source(base, rr.DAMAGE_N, cluster=lambda s: s.cluster)
        doses = [("clean", 0)] + [("d1", i + 1) for i in range(3)] + [("d2", i + 1) for i in range(3)] + \
                [("d3", 1)] + [("d5", i + 1) for i in range(3)] + [("d25", 1)]
        for j, s in enumerate(sub):
            for kind, dose in doses:
                groups["dmg_" + p].append(rr.Spec("dmg:%s:%d:%s" % (kind, dose, s.uid), "dmg", K, p, s.cluster, "0.6",
                                                  s.foot, s.fam, s.wet, {"base": s.uid, "dmg": kind, "dose": dose},
                                                  rr.damage_make(s, kind, dose, j)))
            if mode in ("real", "full"):
                for kind in rr.NUISANCE:
                    groups["nui_" + p].append(rr.Spec("nui:%s:%s" % (kind, s.uid), "nui", K, p, s.cluster, "0.6",
                                                      s.foot, s.fam, s.wet, {"base": s.uid, "nui": kind},
                                                      rr.nuisance_make(s, kind)))
        if mode in ("real", "full") and sub:
            s = sub[0]
            rng = np.random.default_rng(int(hashlib.md5(s.uid.encode()).hexdigest()[:8], 16))
            for i in range(8):
                sh, g = float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-0.5, 0.5))
                groups["d6_" + p].append(rr.Spec("d6:%d:%s" % (i, s.uid), "d6", K, p, s.cluster, "0.6", s.foot, s.fam,
                                                 s.wet, {"base": s.uid, "shift_ms": sh, "gain_db": g},
                                                 rr.d6_make(s, sh, g)))
            for c in rr.control_specs(K, P["family"], s.foot, s.wet, "0.6"):
                c.key, c.pool = K, p
                groups["ctl_" + p].append(c)
        if P["j9"] and mode in ("real", "full"):
            base1 = {r["uid"][3:]: ctx.spec[r["uid"]] for r in P["rows1"]}
            for j, s in enumerate(sub):
                w = base1.get(s.uid)
                if w is None:
                    continue
                for kind, dose in (("clean", 0), ("d8", 1)):
                    groups["dmg1_" + p].append(rr.Spec("dmg1:%s:%s" % (kind, w.uid), "dmg", K, p, w.cluster, "1.0",
                                                       False, w.fam, w.wet, {"base": w.uid, "dmg": kind, "dose": dose},
                                                       rr.damage_make(w, kind, dose, j)))
                for kind in rr.NUISANCE:
                    groups["nui1_" + p].append(rr.Spec("nui1:%s:%s" % (kind, w.uid), "nui", K, p, w.cluster, "1.0",
                                                       False, w.fam, w.wet, {"base": w.uid, "nui": kind},
                                                       rr.nuisance_make(w, kind)))
            for c in rr.control_specs(K, P["family"], False, sub[0].wet if sub else False, "1.0"):
                c.key, c.pool = K, p
                groups["ctl1_" + p].append(c)
    if mode in ("real", "full"):
        d4, pairs = rr.d4_specs(list(prim.values()))
        groups["d4"] = d4
        ctx.d4_pairs = pairs
    return groups


def flaw_specs(gen, keys):
    out = defaultdict(list)
    for g in gen:
        if g.key not in keys or g.meta["seed"] >= rr.FLAW_SEEDS:
            continue
        for kind in ("clean",) + rr.FLAWS:
            seed = int(hashlib.md5(("%s|%s" % (g.uid, kind)).encode()).hexdigest()[:8], 16)
            out["flaw_%s_%s" % (g.meta["route"], g.key)].append(
                rr.Spec("flaw:%s:%s" % (kind, g.uid), "flaw", g.key, g.pool, g.cluster, "0.6", g.foot, g.fam, g.wet,
                        {"route": g.meta["route"], "seed": g.meta["seed"], "flaw": kind, "base": g.uid},
                        rr.flaw_make(g, kind, seed)))
    return out


class KS:
    def __init__(self, key):
        self.key = key
        self.fam, self.foot, self.wet = rr.key_flags(key)
        self.status = rr.KEYS[key]["status"]
        self.pool = None
        self.primary = False
        self.real, self.q, self.real1, self.q1 = [], [], [], []
        self.ref = self.ref1 = self.bank = self.bank1 = None
        self.L = {}
        self.lat1 = {}
        self.judges = []
        self.notes = []
        self.gi = {}
        self.gi1 = {}
        self.emb_real = self.emb_q = None

    def index(self):
        self.qi = {r["uid"]: i for i, r in enumerate(self.q)}
        self.q1i = {r["uid"]: i for i, r in enumerate(self.q1)}
        self.ri = {r["uid"]: i for i, r in enumerate(self.real)}
        self.r1i = {r["uid"]: i for i, r in enumerate(self.real1)}
        self.gi, self.gi1 = defaultdict(list), defaultdict(list)
        for i, r in enumerate(self.q):
            if r["kind"] == "gen":
                self.gi[r["meta"]["route"]].append((r["meta"]["seed"], i))
        for i, r in enumerate(self.q1):
            if r["kind"] == "gen":
                self.gi1[r["meta"]["route"]].append((r["meta"]["seed"], i))
        self.gi = {k: [i for _s, i in sorted(v)] for k, v in self.gi.items()}
        self.gi1 = {k: [i for _s, i in sorted(v)] for k, v in self.gi1.items()}
        self.seeds = {k: [self.q[i]["meta"]["seed"] for i in v] for k, v in self.gi.items()}

    def find(self, kind, **kw):
        out = []
        for i, r in enumerate(self.q):
            if r["kind"] != kind:
                continue
            if all(r["meta"].get(k) == v for k, v in kw.items()):
                out.append(i)
        return out

    def find1(self, kind, **kw):
        out = []
        for i, r in enumerate(self.q1):
            if r["kind"] != kind:
                continue
            if all(r["meta"].get(k) == v for k, v in kw.items()):
                out.append(i)
        return out


def feat_matrix(rows, names):
    return np.array([[r["f"].get(n, NAN) for n in names] for r in rows], float).reshape(len(rows), len(names))


def emb_matrix(rows):
    if not rows:
        return np.zeros((0, 2048))
    if any(r.get("emb") is None for r in rows):
        return None
    return np.vstack([np.asarray(r["emb"], np.float64) for r in rows])


def kq_of(ref, row):
    if row["kind"] in ("dmg", "nui", "d6"):
        k = ref.src_of(row["cluster"])
        if k < 0:
            raise SystemExit("derived clip %s has cluster %s outside its pool" % (row["uid"], row["cluster"]))
        return k
    if row["kind"] == "d4":
        return ref.src_of(row["cluster"])
    return -1


DIRS = {n: rf.direction(n) for n in rf.ALL + rf.J9}
W1_NAMES = list(rf.J9) + list(rf.J4)


def score_key(ctx, ks, union=None):
    t = time.perf_counter()
    if ks.pool is None:
        if union is None:
            return
        Vq = feat_matrix(ks.q, union.names)
        bank = rs.Bank(union, Vq, [-1] * len(ks.q))
        ks.bank_u = bank
        ks.L["J4u"] = bank.lat(rf.J4, bounded=True)
        ctx.stage["stats"] += time.perf_counter() - t
        return
    ks.ref = rs.Ref(feat_matrix(ks.real, rf.ALL), [r["cluster"] for r in ks.real], rf.ALL, DIRS)
    kq = [kq_of(ks.ref, r) for r in ks.q]
    ks.kq = kq
    ks.bank = rs.Bank(ks.ref, feat_matrix(ks.q, rf.ALL), kq)
    for J in ks.judges:
        d = JUDGES[J]
        if d["kind"] == "z" and d["win"] == "0.6":
            ks.L[J] = ks.bank.lat(judge_feats(J, ks.fam), bounded=d["bounded"])
    if "J7" in ks.judges:
        Er, Eq = emb_matrix(ks.real), emb_matrix(ks.q)
        if Er is not None and Eq is not None and len(Er):
            ks.emb_real, ks.emb_q = Er, Eq
            ks.L["J7"] = rs.knn_lat(Er, ks.ref.si, ks.ref.k, Eq, kq, floor_rel=TH["j7_floor_rel"])
        else:
            ks.judges.remove("J7")
    if "J9" in ks.judges and ks.real1:
        ks.ref1 = rs.Ref(feat_matrix(ks.real1, W1_NAMES), [r["cluster"] for r in ks.real1], W1_NAMES, DIRS)
        kq1 = [kq_of(ks.ref1, r) for r in ks.q1]
        ks.bank1 = rs.Bank(ks.ref1, feat_matrix(ks.q1, W1_NAMES), kq1)
        l1 = ks.bank1.lat(judge_feats("J9", ks.fam), bounded=False)
        if l1 is not None:
            ks.lat1["J9"] = l1
            r_map = np.full(len(ks.real), np.nan)
            for i, r in enumerate(ks.real):
                j = ks.r1i.get("w1:" + r["uid"])
                if j is not None:
                    r_map[i] = l1["real"][j]
            q_map = np.full(len(ks.q), np.nan)
            for i, r in enumerate(ks.q):
                if r["kind"] == "gen":
                    j = ks.q1i.get("w1:" + r["uid"])
                    if j is not None:
                        q_map[i] = l1["q"][j]
            ks.L["J9"] = {"real": r_map, "q": q_map, "norm": l1["norm"], "cols": l1["cols"]}
        else:
            ks.judges.remove("J9")
    elif "J9" in ks.judges:
        ks.judges.remove("J9")
    for J in list(ks.judges):
        if J not in ks.L and J != "J8d":
            ks.judges.remove(J)
    ctx.stage["stats"] += time.perf_counter() - t


def j8d_stage(ctx, kss, fam):
    from sklearn.model_selection import GroupKFold
    keys = [ks for ks in kss.values() if ks.fam == fam and ks.ref is not None and "J8d" in ks.judges]
    prim = [ks for ks in keys if ks.primary]
    if not prim or not ctx.emb:
        for ks in keys:
            ks.judges.remove("J8d")
        return {"status": "no pool keys or no embeddings"}
    feats = []
    for J in ("J1", "J3", "J4", "J5"):
        if fam in JUDGES[J]["fams"]:
            feats += judge_feats(J, fam)
    cols = [f for f in feats if all(f in ks.ref.col and ks.ref.valid[ks.ref.col[f]] for ks in prim)]
    ci = lambda ks: [ks.ref.col[f] for f in cols]
    X, y, g, owner = [], [], [], []
    for ks in prim:
        Zr = ks.bank.real_zp[:, ci(ks)]
        Er = emb_matrix(ks.real)
        for i in range(len(ks.real)):
            X.append(Zr[i])
            y.append(1)
            g.append(ks.real[i]["cluster"])
            owner.append((ks.key, "real", i, Er[i]))
        Zq = ks.bank.zpq_mean()[:, ci(ks)]
        Eq = emb_matrix(ks.q)
        for i, r in enumerate(ks.q):
            if r["kind"] == "dmg" and r["meta"]["dmg"] in ("d1", "d2", "d5"):
                X.append(Zq[i])
                y.append(0)
                g.append(r["cluster"])
                owner.append((ks.key, "q", i, Eq[i]))
    X, y, g = np.asarray(X), np.asarray(y), np.asarray(g)
    E = np.vstack([o[3] for o in owner])
    groups = np.unique(g)
    nf = min(TH["J8d_folds"], len(groups))
    fold_of = {}
    gkf = GroupKFold(n_splits=nf)
    models = []
    oof = np.full(len(y), np.nan)
    for f, (tr, te) in enumerate(gkf.split(X, y, g)):
        rm = tr[y[tr] == 1]
        pca = rs.pca_fit(E[rm], TH["pca_k"])
        Xtr = np.hstack([X[tr], rs.pca_apply(pca, E[tr])])
        m = rs.j8d_fit(Xtr, y[tr], C=TH["J8d_C"])
        oof[te] = rs.j8d_prob(m, np.hstack([X[te], rs.pca_apply(pca, E[te])]))
        for gg in np.unique(g[te]):
            fold_of[gg] = f
        models.append((pca, m))
    pca_all = rs.pca_fit(E[y == 1], TH["pca_k"])
    full = (pca_all, rs.j8d_fit(np.hstack([X, rs.pca_apply(pca_all, E)]), y, C=TH["J8d_C"]))
    m_real = float(np.nanmedian(oof[y == 1]))
    eps = TH["j8d_eps"]

    def latp(p):
        return np.maximum(np.log((m_real + eps) / (np.asarray(p) + eps)), 0.0)

    def predict(model, Z, Em):
        pca, m = model
        return rs.j8d_prob(m, np.hstack([Z, rs.pca_apply(pca, Em)]))
    res = {"features": cols, "m_real": m_real, "oof_auc_real_vs_damage": rs.auc(oof[y == 1], oof[y == 0]),
           "folds": nf, "n_train": int(len(y))}
    d4_raw, d4_dec = [], []
    for ks in keys:
        src = next(p for p in prim if p.pool == ks.pool)
        Zr = src.bank.real_zp[:, ci(src)]
        Er = emb_matrix(src.real)
        pr = np.empty(len(src.real))
        for i, r in enumerate(src.real):
            f = fold_of.get(r["cluster"])
            pr[i] = predict(models[f] if f is not None else full, Zr[i:i + 1], Er[i:i + 1])[0]
        Zq = ks.bank.zpq_mean()[:, ci(ks)]
        Eq = emb_matrix(ks.q)
        pq = np.empty(len(ks.q))
        for i, r in enumerate(ks.q):
            if r["kind"] in ("dmg", "nui", "d6") and r["cluster"] in fold_of:
                model = models[fold_of[r["cluster"]]]
            else:
                model = full
            pq[i] = predict(model, Zq[i:i + 1], Eq[i:i + 1])[0]
        raw = lambda p: np.log((m_real + eps) / (np.asarray(p) + eps))
        ks.L["J8d"] = {"real": latp(pr), "q": latp(pq), "real_raw": raw(pr), "q_raw": raw(pq), "norm": m_real,
                       "cols": cols, "p_real": pr, "p_q": pq}
        for i, r in enumerate(ks.q):
            if r["kind"] == "d4":
                (d4_raw if r["meta"]["decoder"] == "raw" else d4_dec).append(pq[i])
    res["d4_auc"] = rs.auc(d4_raw, d4_dec) if d4_raw and d4_dec else NAN
    res["d4_n"] = [len(d4_raw), len(d4_dec)]
    return res


def j2_precond(ks):
    if ks.ref is None or "J2" not in ks.judges:
        return None
    tc = np.array([r["f"].get("two_click", NAN) for r in ks.real], float)
    cf = np.array([r["meta"].get("clipped_frac", 0.0) for r in ks.real], float)
    ok = np.isfinite(tc)
    rate = float(np.mean(tc[ok])) if ok.any() else NAN
    med = float(np.median(cf))
    hi, lo = tc[ok & (cf > med)], tc[ok & (cf <= med)]
    rh = float(np.mean(hi)) if len(hi) else NAN
    rl = float(np.mean(lo)) if len(lo) else NAN
    res = {"two_click_rate": rate, "rate_clipped_hi": rh, "rate_clipped_lo": rl, "median_clipped_frac": med,
           "n_hi": int(len(hi)), "n_lo": int(len(lo))}
    res["pass_rate"] = bool(rate >= TH["J2_min_two"])
    res["pass_split"] = bool(np.isfinite(rh) and np.isfinite(rl) and abs(rh - rl) <= TH["J2_split_points"])
    res["pass"] = bool(res["pass_rate"] and res["pass_split"])
    return res


def finite(v):
    v = np.asarray(v, float)
    return v[np.isfinite(v)]


def null_sets(ctx, ks, mask=None, n=None):
    n = n or TH["null_n"]
    k = ("null", ks.key, n, None if mask is None else int(np.sum(mask)))
    if not hasattr(ctx, "_nulls"):
        ctx._nulls = {}
    if k not in ctx._nulls:
        idx = np.arange(len(ks.real)) if mask is None else np.flatnonzero(mask)
        si = ks.ref.si[idx]
        seed = int(hashlib.md5(("%s|%d" % (ks.key, n)).encode()).hexdigest()[:8], 16)
        ctx._nulls[k] = [idx[d] for d in rs.null_draws(si, n, TH["null_draws"], seed)]
    return ctx._nulls[k]


def s0_judge(ctx, ks, J):
    real = np.asarray(ks.L[J]["real"], float)
    mask = np.isfinite(real)
    if mask.sum() < TH["null_n"]:
        return None
    draws = null_sets(ctx, ks, mask)
    mu = float(np.mean(real[mask]))
    L = np.array([float(np.mean(real[ix])) - mu for ix in draws])
    if JUDGES.get(J, {}).get("bounded"):
        L = np.maximum(L, 0.0)
    R = np.exp(-L)
    return {"q01": rs.pctl(R, 1), "q05": rs.pctl(R, 5), "q10": rs.pctl(R, 10), "q50": rs.pctl(R, 50)}


def lat_q(ks, J, idx):
    if J == "J9":
        return np.asarray(ks.lat1["J9"]["q"], float)[idx]
    return np.asarray(ks.L[J]["q"], float)[idx]


def gather_dose(kss, fam, J, kind, n_doses, win="0.6"):
    clean, doses = [], [[] for _ in range(n_doses)]
    for ks in kss.values():
        if ks.fam != fam or not ks.primary or J not in ks.judges:
            continue
        if J == "J9":
            if "J9" not in ks.lat1:
                continue
            lq = np.asarray(ks.lat1["J9"]["q_raw"], float)
            rows = ks.q1
        else:
            lq = np.asarray(ks.L[J].get("q_raw", ks.L[J]["q"]), float)
            rows = ks.q
        by = {}
        for i, r in enumerate(rows):
            if r["kind"] == "dmg":
                by[(r["meta"]["base"], r["meta"]["dmg"], r["meta"]["dose"])] = i
        bases = sorted({b for b, k, d in by if k == "clean"})
        for b in bases:
            ix = [by.get((b, kind, d + 1)) for d in range(n_doses)]
            if by.get((b, "clean", 0)) is None or any(i is None for i in ix):
                continue
            clean.append(lq[by[(b, "clean", 0)]])
            for d, i in enumerate(ix):
                doses[d].append(lq[i])
    return clean, doses


def d4_gather(kss, fam, J):
    raw, dl, ds = [], [], []
    for ks in kss.values():
        if ks.fam != fam or J not in ks.judges or J not in ks.L:
            continue
        by = {}
        for i, r in enumerate(ks.q):
            if r["kind"] == "d4":
                by[(r["meta"]["pair"], r["meta"]["decoder"])] = i
        for pid in sorted({p for p, _d in by}):
            if all((pid, d) in by for d in ("raw", "same_l", "same_s")):
                lq = ks.L[J].get("q_raw", ks.L[J]["q"])
                raw.append(lq[by[(pid, "raw")]])
                dl.append(lq[by[(pid, "same_l")]])
                ds.append(lq[by[(pid, "same_s")]])
    return raw, dl, ds


def controls_rule(kss, fam, J):
    out = {}
    ok = True
    n = 0
    for ks in kss.values():
        if ks.fam != fam or not ks.primary or J not in ks.judges:
            continue
        if J == "J9":
            if "J9" not in ks.lat1:
                continue
            real = finite(ks.lat1["J9"]["real_raw"])
            lq = np.asarray(ks.lat1["J9"]["q_raw"], float)
            rows = ks.q1
        else:
            real = finite(ks.L[J].get("real_raw", ks.L[J]["real"]))
            lq = np.asarray(ks.L[J].get("q_raw", ks.L[J]["q"]), float)
            rows = ks.q
        q = rs.pctl(real, TH["S1_controls_q"])
        rec = {"real_q95": q}
        for c in ("hiss", "silent"):
            v = finite([lq[i] for i, r in enumerate(rows) if r["kind"] == "ctl" and r["meta"].get("control") == c])
            rec[c] = float(np.median(v)) if len(v) else NAN
            if not (len(v) and np.median(v) > q):
                ok = False
        n += 1
        out[ks.key] = rec
    return {"keys": out, "pass": bool(ok and n > 0)}


def sparkle_rule(kss, fam):
    if fam not in ("glass", "water"):
        return None
    out = {}
    ok = True
    for ks in kss.values():
        if ks.fam != fam or not ks.primary or ks.ref is None:
            continue
        pool = finite([r["f"].get("sparkle", NAN) for r in ks.real])
        mad = float(np.median(np.abs(pool - np.median(pool)))) if len(pool) else NAN
        by = {}
        for r in ks.q:
            if r["kind"] == "dmg":
                by[(r["meta"]["base"], r["meta"]["dmg"], r["meta"]["dose"])] = r["f"].get("sparkle", NAN)
        diffs = [by[(b, "d1", 3)] - by[(b, "clean", 0)] for b, k, d in by if k == "clean" and (b, "d1", 3) in by]
        diffs = finite(diffs)
        med = float(np.median(diffs)) if len(diffs) else NAN
        p = bool(len(diffs) and np.isfinite(mad) and med <= TH["S1_sparkle_mad"] * max(mad, 1e-9))
        out[ks.key] = {"median_rise": med, "pool_mad": mad, "pass": p}
        ok &= p
    return {"keys": out, "pass": bool(ok and out)}


def s1_judge(ctx, kss, fam, J, j8d_res):
    th = {"rise": TH["S1_rise"], "spearman": TH["S1_spearman"], "delta": TH["S1_delta"]}
    res = {"targets": {}}
    tgt = JUDGES[J]["targets"]

    def dose(kind, n):
        c, d = gather_dose(kss, fam, J, kind, n)
        if not c:
            return {"n": 0, "pass": False, "why": "no damaged clips"}
        r = rs.dose_rules(c, d)
        r["pass"] = rs.dose_pass(r, th)
        return r
    if J == "J8d":
        auc = (j8d_res or {}).get("d4_auc", NAN)
        res["targets"]["d4gate"] = {"auc": auc, "n": (j8d_res or {}).get("d4_n"),
                                    "pass": bool(np.isfinite(auc) and auc >= TH["J8d_gate_auc"])}
        ok = res["targets"]["d4gate"]["pass"]
    elif J == "J7":
        raw, dl, ds = d4_gather(kss, fam, J)
        if raw:
            r = rs.dose_rules(raw, [dl, ds])
            r["pass"] = rs.dose_pass(r, th)
        else:
            r = {"n": 0, "pass": False, "why": "no d4 pairs in this family"}
        res["targets"]["d4"] = r
        ok = r["pass"]
    elif J == "J9":
        r = dose("d8", 1)
        res["targets"]["d8"] = r
        ok = r["pass"]
    elif J in ("J1", "J2"):
        r = dose("d2", 3)
        res["targets"]["d2"] = r
        ok = r["pass"]
    elif J == "J4":
        r = dose("d1", 3)
        res["targets"]["d1"] = r
        ok = r["pass"]
    else:
        r2, r5 = dose("d2", 3), dose("d5", 3)
        c2, d2 = gather_dose(kss, fam, J, "d2", 3)
        c25, d25 = gather_dose(kss, fam, J, "d25", 1)
        m2 = float(np.mean(d2[1])) if c2 else NAN
        m25 = float(np.mean(d25[0])) if c25 else NAN
        rule25 = bool(np.isfinite(m2) and np.isfinite(m25) and m25 >= m2)
        res["targets"].update({"d2": r2, "d5": r5, "d25_vs_d2": {"d25_mean": m25, "d2_tau3_mean": m2, "pass": rule25}})
        ok = (r2["pass"] and r5["pass"] and rule25) if J == "J3" else ((r2["pass"] or r5["pass"]) and rule25)
    res["controls"] = controls_rule(kss, fam, J)
    ok = ok and res["controls"]["pass"]
    if J == "J3":
        sp = sparkle_rule(kss, fam)
        if sp is not None:
            res["sparkle_d1"] = sp
            ok = ok and sp["pass"]
    res["pass"] = bool(ok)
    return res


def nuisance_rule(kss, fam, J, comb=None):
    out = {}
    ok = True
    for kind in rr.NUISANCE:
        cl, nu, ratio = [], [], []
        for ks in kss.values():
            if ks.fam != fam or not ks.primary:
                continue
            if comb is None and J not in ks.judges:
                continue
            if comb is not None and ks.key not in comb:
                continue
            if J == "J9":
                if "J9" not in ks.lat1:
                    continue
                lq = np.asarray(ks.lat1["J9"]["q"], float)
                lr = np.asarray(ks.lat1["J9"]["real"], float)
                rows, rmap = ks.q1, ks.r1i
            else:
                lq = np.asarray(comb[ks.key]["q"] if comb is not None else ks.L[J]["q"], float)
                lr = np.asarray(comb[ks.key]["real"] if comb is not None else ks.L[J]["real"], float)
                rows, rmap = ks.q, ks.ri
            iqr = rs.pctl(lr, 75) - rs.pctl(lr, 25)
            by = {}
            for i, r in enumerate(rows):
                if r["kind"] == "dmg" and r["meta"]["dmg"] == "clean":
                    by[("c2", r["meta"]["base"])] = i
                if r["kind"] == "nui" and r["meta"]["nui"] == kind:
                    by[("n", r["meta"]["base"])] = i
            for (t, b), i in list(by.items()):
                if t != "n":
                    continue
                if kind in ("pre25", "pre35"):
                    j = rmap.get(b)
                    c = lr[j] if j is not None else NAN
                else:
                    j = by.get(("c2", b))
                    c = lq[j] if j is not None else NAN
                v = lq[i]
                if np.isfinite(c) and np.isfinite(v):
                    cl.append(c)
                    nu.append(v)
                    ratio.append(abs(v - c) / max(iqr, 1e-9))
        if not cl:
            out[kind] = {"n": 0, "pass": False}
            ok = False
            continue
        d = rs.cliff(cl, nu)
        mr = float(np.median(ratio))
        p = bool(abs(d) < TH["S2_nui_delta"] and mr < TH["S2_nui_iqr"])
        out[kind] = {"n": len(cl), "delta": d, "median_abs_change_over_iqr": mr, "pass": p}
        ok &= p
    return {"kinds": out, "pass": bool(ok)}


def flaw_rule(kss, fam, J, comb=None):
    out = {}
    ok = True
    for kind in rr.FLAWS:
        cl, fl = [], []
        for ks in kss.values():
            if ks.fam != fam or ks.ref is None or ks.status == "proxy":
                continue
            if comb is None and (J not in ks.judges or J == "J9"):
                continue
            if comb is not None and ks.key not in comb:
                continue
            lq = np.asarray(comb[ks.key]["q"] if comb is not None else ks.L[J]["q"], float)
            by = {}
            for i, r in enumerate(ks.q):
                if r["kind"] == "flaw":
                    by[(r["meta"]["base"], r["meta"]["flaw"])] = i
            for (b, k), i in by.items():
                if k != kind or (b, "clean") not in by:
                    continue
                c, v = lq[by[(b, "clean")]], lq[i]
                if np.isfinite(c) and np.isfinite(v):
                    cl.append(c)
                    fl.append(v)
        if not cl:
            out[kind] = {"n": 0, "pass": None}
            continue
        d = rs.cliff(cl, fl)
        p = bool(d > TH["S2_flaw_delta"])
        out[kind] = {"n": len(cl), "delta": d, "mean_change": float(np.mean(np.subtract(fl, cl))), "pass": p}
        ok &= p
    return {"kinds": out, "pass": bool(ok)}


def cell_R(ctx, ks, J, route, seeds=None, comb=None, pen=0.0, bounded=None):
    if route not in ks.gi:
        return None
    gi = ks.gi[route]
    sd = ks.seeds[route]
    if seeds is not None:
        gi = [i for i, s in zip(gi, sd) if s in seeds]
    src = comb if comb is not None else ks.L[J]
    real = np.asarray(src["real"], float)
    g = np.asarray(src["q"], float)[gi]
    g = g[np.isfinite(g)]
    mask = np.isfinite(real)
    if not len(g) or mask.sum() < 2:
        return None
    bnd = JUDGES.get(J, {}).get("bounded", False) if bounded is None else bounded
    L_pt = float(np.mean(g) - np.mean(real[mask]) + pen)
    cls = [r["cluster"] for r, m in zip(ks.real, mask) if m]
    tag = "real|%s|%s|%d" % (ks.key, J if comb is None else "comb", int(mask.sum()))
    rb = ctx.boot.rate(tag, cls, real[mask])
    rk = ("gen", ks.key, route, len(g))
    if not hasattr(ctx, "_gidx"):
        ctx._gidx = {}
    if rk not in ctx._gidx:
        ctx._gidx[rk] = ctx.rng.integers(0, len(g), size=(ctx.args.boot, len(g)))
    gb = g[ctx._gidx[rk]].mean(1)
    out = rs.summarize_R(L_pt, gb - rb + pen, bnd, len(g))
    out["n"] = int(len(g))
    return out


def s3_judge(ctx, kss, fam, J, routes, comb=None):
    pos, neg = [], []
    cells = {}
    two = comb is not None or not JUDGES[J]["bounded"]
    for ks in kss.values():
        if ks.fam != fam or ks.ref is None or ks.status == "proxy":
            continue
        if comb is None and J not in ks.judges:
            continue
        if comb is not None and ks.key not in comb:
            continue
        src = comb[ks.key] if comb is not None else ks.L[J]
        real = finite(src["real"])
        neg += list(real)
        for m in routes:
            if m not in ks.gi:
                continue
            g = finite(np.asarray(src["q"], float)[ks.gi[m]])
            pos += list(g)
            if two and len(g):
                seed = int(hashlib.md5(("%s|%s|%s" % (ks.key, m, J)).encode()).hexdigest()[:8], 16)
                p = rs.mc_less(g, real, TH["S3_mc"], seed)
                cells["%s|%s" % (m, ks.key)] = p
    a = rs.auc(pos, neg)
    sig = sorted(c for c, p in cells.items() if p < TH["S3_p"])
    return {"auc": a, "n_gen": len(pos), "n_real": len(neg), "more_typical_cells": sig,
            "pass": bool(np.isfinite(a) and a >= TH["S3_auc"] and not sig)}


def fam_score(kss, fam, J, route, seeds, comb=None):
    vals = []
    for ks in kss.values():
        if ks.fam != fam or ks.ref is None or ks.status == "proxy" or route not in ks.gi:
            continue
        if comb is None and J not in ks.judges:
            continue
        if comb is not None and ks.key not in comb:
            continue
        src = comb[ks.key] if comb is not None else ks.L[J]
        gi = [i for i, s in zip(ks.gi[route], ks.seeds[route]) if s in seeds]
        g = finite(np.asarray(src["q"], float)[gi])
        r = finite(src["real"])
        if not len(g) or not len(r):
            continue
        L = float(np.mean(g) - np.mean(r))
        if comb is None and JUDGES[J]["bounded"]:
            L = max(L, 0.0)
        vals.append(L)
    return float(np.mean(vals)) if vals else NAN


def s4_judge(ctx, kss, fam, J, routes, comb=None, cells=None):
    seeds = sorted({s for ks in kss.values() for m in routes for s in ks.seeds.get(m, [])})
    sh = rs.split_half_spearman(lambda m, S: fam_score(kss, fam, J, m, S, comb), list(routes), seeds,
                                TH["S4_splits"], seed=7)
    widths, ses = [], []
    for c in (cells or {}).values():
        if c is None:
            continue
        widths.append(c["ci"][1] - c["ci"][0])
        ses.append(c["se_L"])
    mw = float(np.median(widths)) if widths else NAN
    mdr = float(math.exp(2.8 * float(np.median(ses)))) if ses else NAN
    return {"split_spearman": sh["median"], "split_nan_frac": sh["nan_frac"], "median_ci_width": mw, "mdr_n8": mdr,
            "pass": bool(np.isfinite(sh["median"]) and sh["median"] >= TH["S4_spearman"] and np.isfinite(mw) and
                         mw <= TH["S4_ci_width"])}


def comb_arrays(ks, sel):
    parts_r, parts_q = [], []
    for J in sel:
        if J in ks.L:
            parts_r.append(np.asarray(ks.L[J]["real"], float))
            parts_q.append(np.asarray(ks.L[J]["q"], float))
    if not parts_r:
        return None
    R = np.vstack(parts_r)
    Q = np.vstack(parts_q) if len(ks.q) else np.zeros((len(parts_r), 0))
    return {"real": np.nanmean(R, axis=0), "q": np.nanmean(Q, axis=0) if Q.shape[1] else np.zeros(0),
            "judges": [J for J in sel if J in ks.L]}


def zspace(ks, sel):
    feats = []
    for J in sel:
        if J in ZJ:
            feats += judge_feats(J, ks.fam)
    cols = ks.ref.cols(sorted(set(feats), key=feats.index))
    if cols:
        Zr = rs.raw_z(ks.ref.V[:, cols], tuple(s[cols] for s in ks.ref.full))
        Zq = rs.raw_z(ks.bank.Vq[:, cols], tuple(s[cols] for s in ks.ref.full))
        f = lambda Z: np.nan_to_num(rs.SOFT * np.tanh(Z / rs.SOFT), nan=0.0)
        return f(Zr), f(Zq), "z"
    if ks.emb_real is not None:
        return ks.emb_real, ks.emb_q, "emb"
    return None, None, None


def median_pd(Z, kind):
    if Z is None or len(Z) < 2:
        return NAN
    if kind == "emb":
        D = 1.0 - Z @ Z.T
        iu = np.triu_indices(len(Z), 1)
        return float(np.median(D[iu]))
    return rs.median_pairwise(Z)


def collapse_setup(ctx, ks, sel, n=None):
    n = n or TH["null_n"]
    Zr, Zq, kind = zspace(ks, sel)
    if Zr is None:
        return None
    draws = null_sets(ctx, ks, None, n)
    D = np.array([median_pd(Zr[ix], kind) for ix in draws])
    cn = {"q05": rs.pctl(D, 5), "q50": rs.pctl(D, 50), "q95": rs.pctl(D, 95), "kind": kind, "n": n}
    return {"Zr": Zr, "Zq": Zq, "cn": cn, "kind": kind, "draws": draws}


def pen_of(cs, idx, rows="q"):
    if cs is None or not len(idx):
        return 0.0, NAN
    Z = cs["Zq"][idx] if rows == "q" else cs["Zr"][idx]
    D = median_pd(Z, cs["kind"])
    return rs.penalty(D, cs["cn"]), D


def comb_null(ctx, ks, cmb, cs):
    real = np.asarray(cmb["real"], float)
    draws = null_sets(ctx, ks, None)
    mu = float(np.nanmean(real))
    L = []
    for ix in draws:
        p = 0.0
        if cs is not None:
            p = rs.penalty(median_pd(cs["Zr"][ix], cs["kind"]), cs["cn"])
        L.append(float(np.nanmean(real[ix])) - mu + p)
    L = np.asarray(L)
    R = np.exp(-L)
    return {"q01": rs.pctl(R, 1), "q05": rs.pctl(R, 5), "q10": rs.pctl(R, 10), "q50": rs.pctl(R, 50),
            "L_q95": rs.pctl(L, 95)}


def pair_test(ctx, ks, cmb, cs, better, worse, seeds=None):
    if better not in ks.gi or worse not in ks.gi:
        return None

    def pick(m):
        return [i for i, s in zip(ks.gi[m], ks.seeds[m]) if seeds is None or s in seeds]
    ia, ib = pick(better), pick(worse)
    a = np.asarray(cmb["q"], float)[ia]
    b = np.asarray(cmb["q"], float)[ib]
    Dm, cn, pa, pb = None, None, 0.0, 0.0
    if cs is not None:
        cn = cs["cn"] if len(ia) == cs["cn"]["n"] else collapse_setup(ctx, ks, cmb["judges"], n=len(ia))["cn"]
        Z = np.vstack([cs["Zq"][ia], cs["Zq"][ib]])
        Dm = rs.dist_matrix(Z, cs["kind"])
        pa = rs.penalty(median_pd(cs["Zq"][ia], cs["kind"]), cn)
        pb = rs.penalty(median_pd(cs["Zq"][ib], cs["kind"]), cn)
    mu = float(np.nanmean(cmb["real"]))
    La = float(np.mean(a) - mu + pa)
    Lb = float(np.mean(b) - mu + pb)
    t = rs.perm_test(a, b, Dm, cn, seed=3)
    return {"L_better": La, "L_worse": Lb, "ratio": float(math.exp(Lb - La)), "p": t["p"], "p_rev": t["p_rev"],
            "delta": rs.cliff(a, b), "a": a, "b": b, "Dm": Dm, "cn": cn, "pen": [pa, pb], "exact": t.get("exact"),
            "better": better, "worse": worse, "key": ks.key}


def pooled(ctx, tests, use_pen=True):
    strata = [{"a": t["a"], "b": t["b"], "Dm": t["Dm"] if use_pen else None, "cn": t["cn"] if use_pen else None}
              for t in tests]
    sp = rs.strat_perm(strata, TH["perm_mc"], seed=11, pen=use_pen)
    return {"p": sp["p"], "p_rev": sp["p_rev"], "stat": sp["stat"],
            "ratio": float(math.exp(np.mean([t["L_worse"] - t["L_better"] for t in tests]))),
            "delta": rs.cliff_strat([(t["a"], t["b"]) for t in tests])}


def c1_family(ctx, kss, fam, sel, combs, css, routes):
    pairs = C1_PAIRS.get(fam, [])
    rows, tests = [], []
    for better, worse, key in pairs:
        ks = kss.get(key)
        if ks is None or key not in combs or better not in routes or worse not in routes:
            rows.append({"pair": [better, worse, key], "testable": False})
            continue
        t = pair_test(ctx, ks, combs[key], css.get(key), better, worse)
        if t is None:
            rows.append({"pair": [better, worse, key], "testable": False})
            continue
        tests.append(t)
        rows.append({"pair": [better, worse, key], "testable": True, "ratio": t["ratio"], "p": t["p"],
                     "p_rev": t["p_rev"], "delta": t["delta"], "L_better": t["L_better"], "L_worse": t["L_worse"],
                     "correct": bool(t["L_worse"] > t["L_better"]), "sig_wrong": bool(t["p_rev"] < TH["C1_p"]),
                     "proxy": rr.KEYS[key]["status"] == "proxy"})
    if not tests:
        return {"pairs": rows, "testable": False, "pass": False}
    po = pooled(ctx, tests)
    halves = {}
    for name, S in (("seeds_0_3", SEEDS_A), ("seeds_4_7", SEEDS_B)):
        hs = []
        for t in tests:
            h = pair_test(ctx, kss[t["key"]], combs[t["key"]], css.get(t["key"]), t["better"], t["worse"], seeds=S)
            if h is not None:
                hs.append(h["L_worse"] - h["L_better"])
        halves[name] = float(np.sum(hs)) if hs else NAN
    correct = sum(r["correct"] for r in rows if r.get("testable"))
    need = TH["C1_correct"][fam]
    ok = bool(correct >= need and po["p"] < TH["C1_p"] and po["ratio"] >= TH["C1_ratio"] and
              po["delta"] >= TH["C1_delta"] and not any(r.get("sig_wrong") for r in rows) and
              all(np.isfinite(v) and v > 0 for v in halves.values()))
    return {"pairs": rows, "testable": True, "correct": int(correct), "need": need, "pooled": po, "halves": halves,
            "pass": ok}


def seeded(name):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


def median_z(ks, route, feat):
    if ks is None or ks.ref is None or route not in ks.gi or feat not in ks.ref.col:
        return NAN
    c = ks.ref.col[feat]
    if not ks.ref.valid[c]:
        return NAN
    Z = ks.bank.zq_mean()[ks.gi[route], c]
    Z = Z[np.isfinite(Z)]
    return float(np.median(Z)) if len(Z) else NAN


def e_tests(ctx, kss, combs, css, cells, j6, j2p, art, wc, routes):
    sp, wf, mr = "small_prod", "woosh_flow", "medium_rec"
    out = {}
    have = lambda *ms: all(m in routes for m in ms)

    def zt(key, feats, other, both=True):
        ks = kss.get(key)
        zs = {f: median_z(ks, sp, f) for f in feats}
        zo = {f: median_z(ks, other, f) for f in feats}
        if not all(np.isfinite(v) for v in list(zs.values()) + list(zo.values())):
            return {"testable": False, "z_small_prod": zs, "z_%s" % other: zo}
        low = all(zs[f] <= TH["E_z"] for f in feats)
        fs = feats if both else feats[:1]
        closer = all(abs(zo[f]) < abs(zs[f]) for f in fs)
        return {"testable": True, "confirm": bool(low and closer), "z_small_prod": zs, "z_%s" % other: zo}
    out["E1"] = zt("shoot|pistol", ["plr50_db", "e10_100"], mr) if have(sp, mr) else {"testable": False}
    rk = kss.get("shoot|rifle")
    pc = j2p.get("shoot|rifle")
    if rk is not None and pc is not None and pc["pass"] and sp in rk.gi:
        tc = finite([rk.q[i]["f"].get("two_click", NAN) for i in rk.gi[sp]])
        rate = float(np.mean(tc)) if len(tc) else NAN
        out["E2"] = {"testable": True, "confirm": bool(rate <= pc["two_click_rate"] - TH["E2_points"]),
                     "small_prod_rate": rate, "real_rate": pc["two_click_rate"]}
    else:
        out["E2"] = {"testable": False, "why": "J2 preconditions failed" if pc is not None else "no rifle J2",
                     "precondition": pc}
    out["E3"] = zt("break|window", ["sparkle", "hfk_hi"], wf, both=False) if have(sp, wf) else {"testable": False}
    out["E4"] = zt("footstep|gravel", ["hfk_lo", "flux_kurt"], wf) if have(sp, wf) else {"testable": False}
    if j6 and j6.get("valid") and sp in j6.get("models", {}) and wf in j6.get("models", {}):
        a, b = j6["models"][sp]["R"], j6["models"][wf]["R"]
        out["E5"] = {"testable": True, "confirm": bool(a < TH["E5"] and b > a), "R_small_prod": a, "R_woosh_flow": b}
    else:
        out["E5"] = {"testable": False, "why": "J6 not valid" if j6 else "no J6"}
    out["E6"] = zt("splash|water", ["mf_mid_b", "mf_hi_b"], wf) if have(sp, wf) else {"testable": False}

    def cR(key, m):
        c = cells.get(("C", key, m))
        return c["R"] if c else NAN
    w, s_, g = cR("footstep|wood", sp), cR("footstep|stone", sp), cR("footstep|gravel", sp)
    out["E7"] = {"testable": bool(all(np.isfinite([w, s_, g]))), "confirm": bool(w >= s_ >= g),
                 "R": {"wood": w, "stone": s_, "gravel": g}} if all(np.isfinite([w, s_, g])) else {"testable": False}
    means = {}
    for m in routes:
        v = [cR(k, m) for k in CORE_KEYS]
        v = [x for x in v if np.isfinite(x)]
        if v:
            means[m] = float(np.mean(v))
    if sp in means and len(means) >= 9:
        order = sorted(means, key=means.get)
        out["E8"] = {"testable": True, "confirm": bool(order.index(sp) < 3), "rank_from_bottom": order.index(sp) + 1,
                     "means": means}
    else:
        out["E8"] = {"testable": False, "means": means}
    vals = {k: cR(k, sp) for k in CORE_KEYS}
    if all(np.isfinite(v) for v in vals.values()):
        out["E9"] = {"testable": True, "confirm": bool(min(vals, key=vals.get) == "break|window"), "R": vals}
    else:
        out["E9"] = {"testable": False, "R": vals}
    a, b = art.get(("footstep|water", wf)), art.get(("footstep|water", sp))
    if a is not None and b is not None and wf in wc:
        out["E10"] = {"testable": True, "confirm": bool(a["R"] > b["R"] and not wc[wf]["collapse"]),
                      "R_woosh_flow": a["R"], "R_small_prod": b["R"], "woosh_collapse": wc[wf]}
    else:
        out["E10"] = {"testable": False}
    for k, v in out.items():
        v["text"] = E_TEXT[k]
    return out


def j6_stage(ctx, kss, routes):
    surf = {"footstep|gravel": "gravel", "footstep|stone": "stone", "footstep|wood": "wood"}
    fks = [kss[k] for k in surf if k in kss and kss[k].ref is not None]
    if len(fks) < 3 or not ctx.emb:
        return {"valid": False, "why": "needs all three surface pools and embeddings"}
    feats = [f for f in rf.J3 if f != "sparkle"] + list(rf.J5)
    rows = [r for ks in fks for r in ks.real]
    lab = np.array([surf[ks.key] for ks in fks for _r in ks.real])
    cl = np.array([r["cluster"] for r in rows])
    V = feat_matrix(rows, feats)
    E = emb_matrix(rows)
    st = rs.robust_stats(V)

    def zf(Vx, stt):
        return np.nan_to_num(rs.SOFT * np.tanh(rs.raw_z(Vx, stt) / rs.SOFT), nan=0.0)
    pca = rs.pca_fit(E, TH["pca_k"])
    X = np.hstack([zf(V, st), rs.pca_apply(pca, E)])
    S_full = rs.sep_S(X, lab)
    rng = np.random.default_rng(seeded("j6"))
    sh = np.array([rs.sep_S(X, rng.permutation(lab)) for _ in range(TH["J6_shuffles"])])
    p = float((np.sum(sh >= S_full) + 1) / (len(sh) + 1))
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import balanced_accuracy_score
    pred = np.empty(len(lab), dtype=lab.dtype)
    for g in np.unique(cl):
        te = cl == g
        tr = ~te
        stt = rs.robust_stats(V[tr])
        pc = rs.pca_fit(E[tr], TH["pca_k"])
        Xtr = np.hstack([zf(V[tr], stt), rs.pca_apply(pc, E[tr])])
        Xte = np.hstack([zf(V[te], stt), rs.pca_apply(pc, E[te])])
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, lab[tr])
        pred[te] = m.predict(Xte)
    bacc = float(balanced_accuracy_score(lab, pred))
    valid = bool(p < TH["J6_p"] and bacc >= TH["J6_bacc"])
    by = {s: np.flatnonzero(lab == s) for s in ("gravel", "stone", "wood")}
    S_draws = []
    for _ in range(TH["null_draws"]):
        ix = np.concatenate([b[rs.distinct_draw(rng, cl[b], 8)] for b in by.values()])
        S_draws.append(rs.sep_S(X[ix], lab[ix]))
    S_ref = float(np.median(S_draws))
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, lab)
    res = {"valid": valid, "S_full": S_full, "shuffle_p": p, "loso_bacc": bacc, "S_ref": S_ref, "models": {},
           "features": feats}
    for m in routes:
        Xm, lm = [], []
        for ks in fks:
            if m not in ks.gi:
                continue
            idx = ks.gi[m]
            Vq = ks.bank.Vq[idx][:, [ks.ref.col[f] for f in feats]]
            Eq = ks.emb_q[idx] if ks.emb_q is not None else emb_matrix([ks.q[i] for i in idx])
            Xm.append(np.hstack([zf(Vq, st), rs.pca_apply(pca, Eq)]))
            lm += [surf[ks.key]] * len(idx)
        if len(Xm) < 3:
            continue
        Xm = np.vstack(Xm)
        lm = np.asarray(lm)
        S_m = rs.sep_S(Xm, lm)
        subs = []
        r2 = np.random.default_rng(seeded("j6sub|" + m))
        for _ in range(TH["null_draws"]):
            ix = np.concatenate([r2.choice(np.flatnonzero(lm == s), min(TH["J6_sub"], int(np.sum(lm == s))),
                                           replace=False) for s in ("gravel", "stone", "wood")])
            subs.append(rs.sep_S(Xm[ix], lm[ix]) / S_ref)
        acc = float(np.mean(lda.predict(Xm) == lm))
        rec = {"S": S_m, "R": S_m / S_ref, "ci": [rs.pctl(subs, 2.5), rs.pctl(subs, 97.5)], "lda_acc": acc,
               "lda_ratio": acc / max(bacc, 1e-9), "converge": bool(S_m / S_ref < TH["J6_converge"])}
        mk = kss.get("footstep|metal")
        if mk is not None and m in mk.gi and mk.q:
            Vmt = feat_matrix([mk.q[i] for i in mk.gi[m]], feats)
            Emt = emb_matrix([mk.q[i] for i in mk.gi[m]])
            if Emt is not None:
                Xmt = np.hstack([zf(Vmt, st), rs.pca_apply(pca, Emt)])
                cents = [Xm[lm == s].mean(0) for s in ("gravel", "stone", "wood")]
                within = np.mean([np.mean(np.linalg.norm(Xm[lm == s] - c, axis=1))
                                  for s, c in zip(("gravel", "stone", "wood"), cents)])
                cm = Xmt.mean(0)
                rec["metal_distinctness"] = float(min(np.linalg.norm(cm - c) for c in cents) / max(within, 1e-9))
        res["models"][m] = rec
    return res


def water_collapse(ctx, kss, routes):
    sk, fk = kss.get("splash|water"), kss.get("footstep|water")
    if sk is None or fk is None or sk.ref is None or not ctx.emb:
        return {}
    feats = [f for f in rf.J3 if f != "sparkle"] + list(rf.J5)
    V = feat_matrix(sk.real, feats)
    st = rs.robust_stats(V)
    E = emb_matrix(sk.real)
    pca = rs.pca_fit(E, TH["pca_k"])
    zf = lambda Vx: np.nan_to_num(rs.SOFT * np.tanh(rs.raw_z(Vx, st) / rs.SOFT), nan=0.0)
    out = {}
    for m in routes:
        if m not in sk.gi or m not in fk.gi:
            continue
        a = [sk.q[i] for i in sk.gi[m]]
        b = [fk.q[i] for i in fk.gi[m]]
        Ea, Eb = emb_matrix(a), emb_matrix(b)
        if Ea is None or Eb is None:
            continue
        X = np.vstack([np.hstack([zf(feat_matrix(a, feats)), rs.pca_apply(pca, Ea)]),
                       np.hstack([zf(feat_matrix(b, feats)), rs.pca_apply(pca, Eb)])])
        y = np.array([0] * len(a) + [1] * len(b))
        acc = rs.lda_loo(X, y)
        out[m] = {"loo_acc": acc, "collapse": bool(acc <= TH["water_collapse"])}
    return out


def identity_gate(args):
    gate = {}
    water = {"footstep|water", "splash|water"}
    for p in args.identity or []:
        fp = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if not os.path.isfile(fp):
            continue
        with open(fp, encoding="utf-8") as fh:
            V = json.load(fh)
        for k, row in V.get("summary", {}).items():
            if k in water:
                continue
            for s, v in row.get("settings", {}).items():
                if (s, k) not in gate and v.get("cons") is not None:
                    gate[(s, k)] = {"cons": float(v["cons"]), "file": ve.rel(fp), "pq": v.get("pq"),
                                    "pq_real": v.get("pq_real")}
    if args.water_identity:
        fp = args.water_identity if os.path.isabs(args.water_identity) else os.path.join(ROOT, args.water_identity)
        if os.path.isfile(fp):
            from prompt_search import water_scores
            with open(fp, encoding="utf-8") as fh:
                W = json.load(fh)
            wsc, _ch = water_scores(W)
            for (arm, key), w in wsc.items():
                if key in water and w.get("cons") is not None:
                    gate[(arm, key)] = {"cons": float(w["cons"]), "file": ve.rel(fp)}
    return gate


def gate_of(gate, route, key):
    g = gate.get((route, key))
    if g is None:
        return "?", None
    return ("ok" if g["cons"] >= TH["gate_cons"] else "failed"), g["cons"]


def verdict(R, ci, band, gstate, status, has_pool):
    if gstate == "failed":
        return "X", True
    if R is None or not np.isfinite(R) or band is None:
        return "-", False
    q10, q01 = band["q10"], band["q01"]
    cut = TH["verdict_C_frac"] * q01
    lo, hi = ci
    if R >= q10:
        if status == "proxy":
            return "CP", bool(lo >= q10)
        if gstate == "?":
            return "C?", bool(lo >= q10)
        return "R", bool(lo >= q10)
    if R >= cut:
        return "C", bool(lo >= cut and hi < q10)
    return "A", bool(hi < cut)


def reason(ks, route, sel):
    feats = []
    for J in sel:
        if J in ZJ:
            feats += judge_feats(J, ks.fam)
    zs = []
    for f in dict.fromkeys(feats):
        z = median_z(ks, route, f)
        if np.isfinite(z):
            c = ks.ref.col[f]
            g = finite(ks.bank.Vq[ks.gi[route], c])
            r = finite(ks.ref.V[:, c])
            zs.append((abs(z), f, z, float(np.median(g)) if len(g) else NAN, float(np.median(r)) if len(r) else NAN))
    zs.sort(reverse=True)
    return ["%s z %+.1f (gen %s, real %s)" % (f, z, fnum(gm), fnum(rm)) for _a, f, z, gm, rm in zs[:2]]


def write_csv(path, kss, routes_all):
    names = list(dict.fromkeys(rf.ALL + rf.J9 + rf.DESC + ["spark_f0x"]))
    lj = sorted({J for ks in kss.values() for J in ks.L})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "uid", "kind", "route", "seed", "cluster", "loso_src", "variant"] + names +
                   ["lat_%s" % J for J in lj] + ["lat_comb", "p_real"])
        for ks in kss.values():
            cmb = getattr(ks, "comb", None)
            for side, rows in (("real", ks.real), ("q", ks.q)):
                for i, r in enumerate(rows):
                    m = r["meta"]
                    var = m.get("dmg") or m.get("nui") or m.get("flaw") or m.get("decoder") or m.get("control") or ""
                    if m.get("dose"):
                        var = "%s_%s" % (var, m["dose"])
                    lat = []
                    for J in lj:
                        v = ks.L[J][side][i] if J in ks.L else NAN
                        lat.append(fnum(float(v), 4))
                    lc = fnum(float(cmb[side][i]), 4) if cmb is not None else "-"
                    pr = fnum(float(ks.L["J8d"]["p_real" if side == "real" else "p_q"][i]), 4) if "J8d" in ks.L else "-"
                    w.writerow([ks.key, r["uid"], r["kind"], m.get("route", ""), m.get("seed", ""), r["cluster"],
                                "", var] + [fnum(r["f"].get(n, NAN), 4) if isinstance(r["f"].get(n, NAN), float)
                                            else r["f"].get(n, "") for n in names] + lat + [lc, pr])


def load_ratings():
    if not os.path.isfile(RATINGS):
        return None
    with open(RATINGS, encoding="utf-8") as fh:
        return json.load(fh)


def v6(ctx, kss, ratings, slot):
    if ratings is None:
        return {"status": "pending: no ear_ratings.json yet"}
    rows = ratings.get("ratings", [])
    gen = [r for r in rows if r.get("kind") == "gen" and r.get("seed") is not None and int(r["seed"]) == slot and
           not r.get("repeat_of")]
    pts = []
    for r in gen:
        ks = kss.get(r["key"])
        if ks is None or getattr(ks, "comb", None) is None or r["model"] not in ks.gi:
            continue
        ix = [i for i, sd in zip(ks.gi[r["model"]], ks.seeds[r["model"]]) if sd == slot]
        if not ix or not np.isfinite(ks.comb["q"][ix[0]]):
            continue
        pts.append({"fam": ks.fam, "key": r["key"], "model": r["model"], "s": -float(ks.comb["q"][ix[0]]),
                    "y": float(r["realism"])})
    rng = np.random.default_rng(seeded("v6"))
    out = {"status": "computed", "slot": slot, "n": len(pts), "families": {}}
    for fam in sorted({p["fam"] for p in pts}):
        P = [p for p in pts if p["fam"] == fam]
        keys = sorted({p["key"] for p in P})

        def within_key(ks_):
            v = []
            for k in ks_:
                a = [(p["s"], p["y"]) for p in P if p["key"] == k]
                if len(a) >= 3:
                    v.append(rs.spearman([x for x, _ in a], [y for _, y in a]))
            v = [x for x in v if np.isfinite(x)]
            return float(np.mean(v)) if v else NAN
        wk = within_key(keys)
        bk = [within_key(list(rng.choice(keys, len(keys), replace=True))) for _ in range(1000)]
        S = np.array([p["s"] for p in P])
        Y = np.array([p["y"] for p in P])
        kk = np.array([p["key"] for p in P])
        mm = np.array([p["model"] for p in P])

        def centre(v):
            v = v.copy()
            for _ in range(10):
                for g in (kk, mm):
                    for u in np.unique(g):
                        v[g == u] -= v[g == u].mean()
            return v
        Sc, Yc = centre(S), centre(Y)
        wm = rs.spearman(Sc, Yc)
        bm = []
        for _ in range(1000):
            ix = rng.integers(0, len(P), len(P))
            bm.append(rs.spearman(Sc[ix], Yc[ix]))
        lo_k, lo_m = rs.pctl(bk, 2.5), rs.pctl(bm, 2.5)
        out["families"][fam] = {
            "within_key_spearman": wk, "within_key_lo": lo_k, "within_model_spearman": wm, "within_model_lo": lo_m,
            "n": len(P),
            "pass": bool(np.isfinite(wk) and wk >= TH["V6_within_key"] and lo_k > 0 and np.isfinite(wm) and
                         wm >= TH["V6_within_model"] and lo_m > 0)}
    rep = {r["id"]: r for r in rows}
    pairs = [(rep[r["repeat_of"]]["realism"], r["realism"]) for r in rows if r.get("repeat_of") in rep]
    out["test_retest"] = rs.spearman([a for a, _ in pairs], [b for _, b in pairs]) if len(pairs) >= 3 else NAN
    ceil = {}
    for key in sorted({r["key"] for r in rows}):
        real = [r["realism"] for r in rows if r["key"] == key and r.get("kind") == "real" and not r.get("repeat_of")]
        if not real:
            continue
        mr = float(np.mean(real))
        mods = {}
        for m in sorted({r["model"] for r in rows if r["key"] == key and r.get("kind") == "gen"}):
            v = [r["realism"] for r in rows if r["key"] == key and r.get("model") == m and not r.get("repeat_of")]
            mods[m] = float(np.mean(v))
        ceil[key] = {"real_mean": mr, "models": mods, "reference_ceiling": sorted(m for m, v in mods.items() if v > mr)}
    out["real_vs_models"] = ceil
    return out


def run(args, mode):
    ctx = Ctx(args)
    os.makedirs(args.out_dir, exist_ok=True)
    pre = None
    post_hoc = False
    if mode == "full" and not args.dev:
        if not os.path.isfile(PREREG):
            raise SystemExit("validate needs %s; run `prereg` after the --real-only iterations" % PREREG)
        with open(PREREG, encoding="utf-8") as fh:
            pre = json.load(fh)
        now = core_md5()
        post_hoc = any(pre["md5"].get(f) != now[f] for f in CORE)
        if post_hoc:
            ctx.say("POST-HOC: core files changed since prereg.json: %s" % [f for f in CORE if pre["md5"].get(f) !=
                                                                             now[f]])
    sel_frozen = None
    if mode == "score" and args.dev and args.force_select:
        sel_frozen = {"families": {f: {"selected": [J for J in js.split(",") if J]} for f, _, js in
                                   (x.partition("=") for x in args.force_select)}}
    elif mode == "score":
        if not os.path.isfile(VALIDATED):
            raise SystemExit("score needs %s from a full validate run" % VALIDATED)
        with open(VALIDATED, encoding="utf-8") as fh:
            sel_frozen = json.load(fh)
    keys = args.keys or list(rr.KEYS)
    pools = build_pools(ctx)
    prim = primary_keys(pools, keys)
    routes, cfgs, gen, raw = [], {}, [], []
    if mode in ("full", "score"):
        specs = list(args.route)
        if args.bakeoff:
            bd = args.bakeoff if os.path.isabs(args.bakeoff) else os.path.join(ROOT, args.bakeoff)
            models = args.models or sorted(d for d in os.listdir(os.path.join(bd, "routes"))
                                           if os.path.isdir(os.path.join(bd, "routes", d)))
            specs += ["%s=%s" % (m, os.path.join(bd, "routes", m)) for m in models]
            raw_dirs = {m: os.path.join(bd, "raw", m) for m in models}
        else:
            raw_dirs = {}
        if args.baseline:
            specs.append(args.baseline)
        for sp in args.raw:
            n, _, d = sp.partition("=")
            raw_dirs[n] = d if os.path.isabs(d) else os.path.join(ROOT, d)
        for sp in specs:
            n, _, d = sp.partition("=")
            dd = d if os.path.isabs(d) else os.path.join(ROOT, d)
            if n not in raw_dirs:
                cand = os.path.join(os.path.dirname(os.path.dirname(dd)), "raw", os.path.basename(dd))
                if os.path.isfile(os.path.join(cand, "gen.json")):
                    raw_dirs[n] = cand
        ns = argparse.Namespace(route=specs, seeds=args.seeds)
        cfgs, gen = rr.gen_specs(ns, ctx.man)
        gen = [g for g in gen if g.key in keys]
        routes = list(cfgs)
        for n in routes:
            drop = set(rr.route_drop(cfgs[n]))
            for p in pools:
                hit = drop & {r["cluster"] for r in pools[p]["rows"]}
                if hit:
                    raise SystemExit("route %s lists %d clusters inside pool %s; v1.1 scores only routes that share "
                                     "no source with a pool" % (n, len(hit), p))
        j9keys = [k for k in keys if rr.KEYS[k]["pool"] and pools.get(rr.KEYS[k]["pool"], {}).get("j9")]
        for n in routes:
            if n in raw_dirs:
                raw += rr.raw_specs(n, raw_dirs[n], args.seeds, j9keys)
    groups = derived_specs(ctx, pools, prim, mode)
    if mode == "full":
        core = {k for k in keys if rr.KEYS[k]["pool"] in prim and rr.KEYS[k]["status"] != "proxy"}
        groups.update(flaw_specs(gen, core))
    thrs = {}
    for p, P in pools.items():
        thrs[(p, "0.6")] = P["thr"] if P["valid"] else None
        thrs[(p, "1.0")] = P["thr1"] if P["j9"] else None
    rows_all = []
    for g in sorted(groups):
        rows_all += compute(ctx, groups[g], g, thr=thrs)
        ctx.say("computed %-16s %4d clips  (features %.0f s, panns %.0f s)" % (g, len(groups[g]),
                                                                              ctx.stage["features"],
                                                                              ctx.stage["panns"]))
    by_route = defaultdict(list)
    for s in gen:
        by_route[s.meta["route"]].append(s)
    for n, ss in by_route.items():
        for key in sorted({s.key for s in ss}):
            sub = [s for s in ss if s.key == key]
            rows_all += compute(ctx, sub, "gen_%s_%s" % (n, key), thr=thrs)
        ctx.say("computed gen %-12s %4d clips" % (n, len(ss)))
    for n in routes:
        sub = [s for s in raw if s.meta["route"] == n]
        for key in sorted({s.key for s in sub}):
            kk = [s for s in sub if s.key == key]
            rows_all += compute(ctx, kk, "w1gen_%s_%s" % (n, key), thr=thrs)
    ctx.save_caches()
    ctx.say("features done: %.0f s features, %.0f s panns" % (ctx.stage["features"], ctx.stage["panns"]))
    kss = {}
    for key in keys:
        ks = KS(key)
        p = rr.KEYS[key]["pool"]
        if p and pools.get(p, {}).get("valid"):
            ks.pool = p
            ks.primary = prim.get(p) == key
            ks.real = pools[p]["rows"]
            ks.real1 = pools[p]["rows1"]
            for J, d in JUDGES.items():
                if ks.fam in d["fams"]:
                    if J == "J9" and not pools[p]["j9"]:
                        continue
                    if J in ("J7", "J8d") and not ctx.emb:
                        continue
                    ks.judges.append(J)
        else:
            ks.notes.append("pool-free" if not p else "pool %s below minimum" % p)
        ks.q = [r for r in rows_all if r["key"] == key and r["win"] == "0.6"]
        ks.q1 = [r for r in rows_all if r["key"] == key and r["win"] == "1.0"]
        ks.index()
        kss[key] = ks
    urows = [r for p, P in pools.items() if P["valid"] and p != "reload" for r in P["rows"]]
    union = rs.Ref(feat_matrix(urows, rf.J4), [r["cluster"] for r in urows], list(rf.J4), DIRS)
    for ks in kss.values():
        score_key(ctx, ks, union)
    j2p = {}
    for ks in kss.values():
        pc = j2_precond(ks)
        if pc is not None:
            j2p[ks.key] = pc
            if not pc["pass"] and "J2" in ks.judges:
                ks.judges.remove("J2")
                ks.L.pop("J2", None)
                ks.notes.append("J2 dropped: precondition failed")
    j8 = {}
    for fam in rr.FAMILIES:
        if any(ks.fam == fam and "J8d" in ks.judges for ks in kss.values()):
            t = time.perf_counter()
            j8[fam] = j8d_stage(ctx, kss, fam)
            ctx.stage["stats"] += time.perf_counter() - t
    ctx.say("judges scored (stats %.0f s)" % ctx.stage["stats"])
    sel_res = {}
    for fam in rr.FAMILIES:
        fk = [ks for ks in kss.values() if ks.fam == fam and ks.ref is not None]
        if not fk:
            continue
        cands = sorted({J for ks in fk for J in ks.judges}, key=list(JUDGES).index)
        fr = {}
        for J in cands:
            t = time.perf_counter()
            pk = [ks for ks in fk if ks.status != "proxy" and J in ks.judges and J in ks.L]
            s0 = {ks.key: s0_judge(ctx, ks, J) for ks in pk}
            s0_ok = [v is not None and v["q05"] >= TH["S0_q05"] for v in s0.values()]
            r = {"S0": {"keys": s0, "pass": bool(s0_ok and np.mean(s0_ok) >= TH["S0_key_frac"])}}
            r["S1"] = s1_judge(ctx, kss, fam, J, j8.get(fam))
            r["S2"] = {"nuisance": nuisance_rule(kss, fam, J)}
            if mode == "full":
                r["S2"]["flaws"] = flaw_rule(kss, fam, J) if J != "J9" else {"pass": True, "note": "W1.0 judge"}
                r["S2"]["pass"] = bool(r["S2"]["nuisance"]["pass"] and r["S2"]["flaws"]["pass"])
                cells = {}
                for ks in pk:
                    for m in routes:
                        cells[(m, ks.key)] = cell_R(ctx, ks, J, m)
                r["cells"] = {"%s|%s" % k: v for k, v in cells.items()}
                r["S3"] = s3_judge(ctx, kss, fam, J, routes)
                r["S4"] = s4_judge(ctx, kss, fam, J, routes, cells=cells)
                r["pass"] = bool(r["S0"]["pass"] and r["S1"]["pass"] and r["S2"]["pass"] and r["S3"]["pass"] and
                                 r["S4"]["pass"])
            else:
                r["S2"]["pass"] = r["S2"]["nuisance"]["pass"]
                r["pass_real_only"] = bool(r["S0"]["pass"] and r["S1"]["pass"] and r["S2"]["pass"])
            fr[J] = r
            ctx.stage["stats"] += time.perf_counter() - t
            ctx.check("stats")
        sel_res[fam] = fr
        ctx.say("family %-8s " % fam + "  ".join("%s %s" % (J, ("PASS" if v.get("pass", v.get("pass_real_only"))
                                                              else "fail")) for J, v in fr.items()))
    j6 = j6_stage(ctx, kss, routes) if any(k.startswith("footstep|") for k in keys) else None
    if j6:
        ctx.say("J6 valid %s (shuffle p %.3f, LOSO balanced accuracy %.2f)" % (j6.get("valid"), j6.get("shuffle_p", NAN),
                                                                             j6.get("loso_bacc", NAN)))
    wc = water_collapse(ctx, kss, routes) if mode != "real" else {}
    result = {"meta": {"mode": mode, "args": {k: v for k, v in vars(args).items()}, "md5": core_md5(),
                       "feats_md5": ctx.fmd5, "chain": rr.CHAIN_VERSION, "post_hoc": post_hoc,
                       "prereg": ve.rel(PREREG) if pre else None, "prereg_time": (pre or {}).get("time"),
                       "dev": bool(args.dev), "routes": routes},
              "pools": {p: {k: v for k, v in P.items() if k not in ("rows", "all", "rows1")} for p, P in pools.items()},
              "primary": prim, "keys": {}, "j2_preconditions": j2p, "j8d": j8, "selection_tests": sel_res, "j6": j6,
              "water_collapse": wc}
    for ks in kss.values():
        result["keys"][ks.key] = {"family": ks.fam, "status": ks.status, "pool": ks.pool, "judges": ks.judges,
                                  "notes": ks.notes, "n_real": len(ks.real), "n_real_w1": len(ks.real1),
                                  "routes": sorted(ks.gi)}
    if mode == "real":
        finish_real(ctx, result, kss, sel_res, args)
        return result
    selected = {}
    if mode == "score":
        selected = {fam: v["selected"] for fam, v in sel_frozen["families"].items()}
    else:
        for fam, fr in sel_res.items():
            selected[fam] = [J for J, r in fr.items() if r.get("pass")]
        if args.dev and args.force_select:
            for item in args.force_select:
                fam, _, js = item.partition("=")
                selected[fam] = [J for J in js.split(",") if J]
    combs, css, cnull, cells, comb_checks = {}, {}, {}, {}, {}
    for ks in kss.values():
        if ks.ref is None:
            continue
        sel = [J for J in selected.get(ks.fam, []) if J in ks.L]
        if not sel:
            continue
        cmb = comb_arrays(ks, sel)
        ks.comb = cmb
        combs[ks.key] = cmb
        css[ks.key] = collapse_setup(ctx, ks, sel)
        cnull[ks.key] = comb_null(ctx, ks, cmb, css[ks.key])
        for m in routes:
            if m not in ks.gi:
                continue
            pen, D = pen_of(css[ks.key], ks.gi[m])
            c = cell_R(ctx, ks, "C", m, comb=cmb, pen=pen, bounded=False)
            if c is not None:
                c["penalty"], c["D"] = pen, D
                c["collapse"] = bool(pen > 0)
                cells[("C", ks.key, m)] = c
            for J in sel:
                cj = cell_R(ctx, ks, J, m)
                if cj is not None:
                    cells[(J, ks.key, m)] = cj
    if mode == "full":
        for fam in sel_res:
            sel = selected.get(fam, [])
            if not sel:
                comb_checks[fam] = {"status": "unvalidated"}
                continue
            comb_checks[fam] = combined_checks(ctx, kss, fam, sel, combs, css, cnull, routes, cells)
    art = {}
    for ks in kss.values():
        if ks.pool is None and "J4u" in ks.L:
            for m in routes:
                if m in ks.gi:
                    c = cell_R(ctx, ks, "J4u", m, bounded=True) if ks.real else None
                    gi = ks.gi[m]
                    g = finite(np.asarray(ks.L["J4u"]["q"], float)[gi])
                    r = finite(ks.L["J4u"]["real"])
                    L = max(float(np.mean(g) - np.mean(r)), 0.0) if len(g) and len(r) else NAN
                    art[(ks.key, m)] = {"R": math.exp(-L) if np.isfinite(L) else NAN, "L": L}
    gate = identity_gate(args)
    stage_c = None
    if mode == "full":
        validated = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "post_hoc": post_hoc, "dev": bool(args.dev),
                     "prereg_md5": (pre or {}).get("md5"), "families": {}}
        for fam in sel_res:
            sel = selected.get(fam, [])
            st = "unvalidated" if not sel else "thin" if len(sel) == 1 else "validated"
            validated["families"][fam] = {"selected": sel, "status": st,
                                          "combined_pass": comb_checks.get(fam, {}).get("pass"),
                                          "failing": {J: failing(r) for J, r in sel_res[fam].items() if not r["pass"]}}
        if not args.dev:
            dump(validated, VALIDATED)
        result["validated"] = validated
        stage_c = stage_confirm(ctx, kss, selected, combs, css, cells, j6, j2p, art, wc, routes, sel_res)
    result["selected"] = selected
    result["combined_checks"] = comb_checks
    result["stage_c"] = stage_c
    result["combined_null"] = cnull
    rows_out = {}
    for ks in kss.values():
        for m in routes:
            gs, gc = gate_of(gate, m, ks.key)
            rec = {"gate": gs, "cons": gc, "pq": (gate.get((m, ks.key)) or {}).get("pq")}
            c = cells.get(("C", ks.key, m))
            if c is not None:
                v, firm = verdict(c["R"], c["ci"], cnull.get(ks.key), gs, ks.status, ks.pool is not None)
                rec.update({"R": c["R"], "ci": c["ci"], "verdict": v, "firm": firm, "collapse": c["collapse"],
                            "penalty": c["penalty"],
                            "R_J": {J: cells[(J, ks.key, m)]["R"] for J in selected.get(ks.fam, [])
                                    if (J, ks.key, m) in cells},
                            "R_J_ci": {J: cells[(J, ks.key, m)]["ci"] for J in selected.get(ks.fam, [])
                                       if (J, ks.key, m) in cells},
                            "reason": reason(ks, m, selected.get(ks.fam, [])),
                            "carried_by": carried(ctx, ks, m, selected.get(ks.fam, []))})
                if ks.fam == "gun":
                    rec["flags"] = ["clipped pool"]
            if (ks.key, m) in art:
                rec["artifact_R"] = art[(ks.key, m)]["R"]
            if ks.ref is not None and ks.emb_q is not None and m in ks.gi:
                rec["kad"] = rs.kad(ks.emb_q[ks.gi[m]], ks.emb_real, perms=100, seed=seeded(m + ks.key))
            if m in ks.gi:
                rec["descriptive"] = {d: float(np.nanmedian([ks.q[i]["f"].get(d, NAN) for i in ks.gi[m]]))
                                      for d in ("rolloff99", "hf16_db", "clip_frac", "peak", "zero_frac")}
            rows_out["%s|%s" % (m, ks.key)] = rec
    result["cells"] = rows_out
    if j6:
        result["j6"] = j6
    result["decoder_ceiling"] = decoder_ceiling(kss, combs)
    if mode == "score" and args.baseline:
        result["experiment"] = experiment(ctx, kss, combs, css, cells, routes, args, cfgs)
    if mode == "full":
        result["V6"] = v6(ctx, kss, load_ratings(), TH["slots"]["B"])
    result["meta"]["runtime_s"] = {k: round(v, 1) for k, v in ctx.stage.items()}
    result["meta"]["runtime_s"]["total"] = round(time.perf_counter() - ctx.t0, 1)
    d = os.path.join(args.out_dir, args.tag)
    os.makedirs(d, exist_ok=True)
    dump(result, os.path.join(d, "results.json"))
    write_csv(os.path.join(d, "clips.csv"), kss, routes)
    txt = table(result, kss, routes, selected, cnull, gate)
    with open(os.path.join(d, "table.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)
    ctx.say("saved %s" % ve.rel(d))
    return result


def failing(r):
    out = []
    for s in ("S0", "S1", "S2", "S3", "S4"):
        if s in r and not r[s].get("pass"):
            out.append(s)
    return out


def carried(ctx, ks, m, sel):
    if len(sel) < 2 or m not in ks.gi:
        return []
    base = comb_arrays(ks, sel)
    g = finite(np.asarray(base["q"], float)[ks.gi[m]])
    L0 = float(np.mean(g) - np.nanmean(base["real"]))
    out = []
    for J in sel:
        rest = [x for x in sel if x != J]
        c = comb_arrays(ks, rest)
        if c is None:
            continue
        g2 = finite(np.asarray(c["q"], float)[ks.gi[m]])
        L1 = float(np.mean(g2) - np.nanmean(c["real"]))
        if np.sign(L1) != np.sign(L0) and abs(L0) > 0.05:
            out.append(J)
    return out


def combined_checks(ctx, kss, fam, sel, combs, css, cnull, routes, cells):
    fk = [ks for ks in kss.values() if ks.fam == fam and ks.key in combs]
    out = {"judges": sel}
    raise_n, tot = 0, 0
    for ks in fk:
        by = {}
        for i, r in enumerate(ks.q):
            if r["kind"] == "d4":
                by[(r["meta"]["pair"], r["meta"]["decoder"])] = i
        lc = combs[ks.key]["q"]
        for pid in {p for p, _d in by}:
            if (pid, "raw") not in by:
                continue
            for dec in ("same_l", "same_s"):
                if (pid, dec) in by:
                    tot += 1
                    raise_n += int(lc[by[(pid, dec)]] > lc[by[(pid, "raw")]])
    out["d4"] = {"raised": raise_n, "pairs": tot, "frac": raise_n / tot if tot else NAN,
                 "pass": bool(tot and raise_n / tot >= TH["S1c_d4"])}
    d6 = {}
    for ks in fk:
        if not ks.primary:
            continue
        idx = [i for i, r in enumerate(ks.q) if r["kind"] == "d6"]
        if not idx:
            continue
        pen, D = pen_of(css[ks.key], idx)
        L = float(np.nanmean(np.asarray(combs[ks.key]["q"])[idx]) - np.nanmean(combs[ks.key]["real"]) + pen)
        d6[ks.key] = {"L": L, "penalty": pen, "null_L_q95": cnull[ks.key]["L_q95"],
                      "pass": bool(L > cnull[ks.key]["L_q95"])}
    out["d6"] = {"keys": d6, "pass": bool(d6 and all(v["pass"] for v in d6.values()))}
    d3, d2 = [], []
    for ks in fk:
        if not ks.primary:
            continue
        lc = combs[ks.key]["q"]
        by = {}
        for i, r in enumerate(ks.q):
            if r["kind"] == "dmg":
                by[(r["meta"]["base"], r["meta"]["dmg"], r["meta"]["dose"])] = i
        for b in {b for b, k, d in by if k == "clean"}:
            c = lc[by[(b, "clean", 0)]]
            if (b, "d3", 1) in by:
                d3.append(lc[by[(b, "d3", 1)]] - c)
            if (b, "d2", 2) in by:
                d2.append(lc[by[(b, "d2", 2)]] - c)
    m3, m2 = (float(np.nanmean(d3)) if d3 else NAN), (float(np.nanmean(d2)) if d2 else NAN)
    out["P1"] = {"d3_mean_rise": m3, "d2_tau3_mean_rise": m2, "pass": bool(np.isfinite(m3) and np.isfinite(m2) and
                                                                           m3 < m2)}
    out["S2"] = {"nuisance": nuisance_rule(kss, fam, "C", comb=combs), "flaws": flaw_rule(kss, fam, "C", comb=combs)}
    out["S2"]["pass"] = bool(out["S2"]["nuisance"]["pass"] and out["S2"]["flaws"]["pass"])
    out["S3"] = s3_judge(ctx, kss, fam, "C", routes, comb=combs)
    cc = {k: v for k, v in cells.items() if k[0] == "C" and kss[k[1]].fam == fam and kss[k[1]].status != "proxy"}
    out["S4"] = s4_judge(ctx, kss, fam, "C", routes, comb=combs, cells=cc)
    out["S1_pass"] = bool(out["d4"]["pass"] and out["d6"]["pass"] and out["P1"]["pass"])
    out["pass"] = bool(out["S1_pass"] and out["S2"]["pass"] and out["S3"]["pass"] and out["S4"]["pass"])
    return out


def stage_confirm(ctx, kss, selected, combs, css, cells, j6, j2p, art, wc, routes, sel_res):
    out = {"C1": {}, "per_judge_C1": {}}
    for fam in C1_PAIRS:
        sel = selected.get(fam, [])
        if not sel:
            out["C1"][fam] = {"testable": False, "pass": False, "why": "no selected judge"}
            continue
        r = c1_family(ctx, kss, fam, sel, combs, css, routes)
        lo = {}
        if r.get("testable"):
            for J in sel:
                rest = [x for x in sel if x != J]
                if not rest:
                    continue
                cb = {k: comb_arrays(kss[k], rest) for k in combs if kss[k].fam == fam}
                cb = {k: v for k, v in cb.items() if v is not None}
                cs2 = {k: collapse_setup(ctx, kss[k], rest) for k in cb}
                rr2 = c1_family(ctx, kss, fam, rest, cb, cs2, routes)
                if rr2.get("testable") and np.sign(rr2["pooled"]["stat"]) != np.sign(r["pooled"]["stat"]):
                    lo[J] = "pooled sign flips without %s" % J
        r["carried_by"] = lo
        out["C1"][fam] = r
        pj = {}
        cand = sorted({J for ks in kss.values() if ks.fam == fam for J in ks.L if J in JUDGES})
        for J in cand:
            cb = {k: comb_arrays(kss[k], [J]) for k in combs if kss[k].fam == fam}
            cb = {k: v for k, v in cb.items() if v is not None}
            for ks in kss.values():
                if ks.fam == fam and ks.ref is not None and ks.key not in cb and J in ks.L:
                    cb[ks.key] = comb_arrays(ks, [J])
            cs2 = {k: None for k in cb}
            rj = c1_family(ctx, kss, fam, [J], cb, cs2, routes)
            if rj.get("testable"):
                pj[J] = {"correct": rj["correct"], "pooled_p": rj["pooled"]["p"], "pooled_p_rev": rj["pooled"]["p_rev"],
                         "pooled_ratio": rj["pooled"]["ratio"], "pooled_delta": rj["pooled"]["delta"],
                         "pass": rj["pass"]}
        out["per_judge_C1"][fam] = pj
    E = e_tests(ctx, kss, combs, css, cells, j6, j2p, art, wc, routes)
    out["E"] = E
    test = [k for k, v in E.items() if v.get("testable")]
    conf = [k for k in test if E[k].get("confirm")]
    out["E_summary"] = {"testable": test, "confirmed": conf,
                        "frac": len(conf) / len(test) if test else NAN}
    gun = out["C1"].get("gun", {}).get("pass", False)
    c3 = {"gun_pair": bool(gun), "E7": bool(E["E7"].get("testable") and E["E7"].get("confirm")),
          "E9": bool(E["E9"].get("testable") and E["E9"].get("confirm"))}
    out["C3"] = c3
    n_c1 = sum(bool(out["C1"][f].get("pass")) for f in C1_PAIRS)
    c2_ok = bool(len(test) >= TH["outcome_E_min"] and len(conf) / max(1, len(test)) >= TH["outcome_E_frac"])
    c1_ok = n_c1 >= TH["outcome_C1_families"]
    if c1_ok and c2_ok and all(c3.values()):
        oc = "PASS"
    elif c1_ok and c2_ok:
        oc = "PASS-FP-RISK"
    else:
        oc = "FAIL"
    out["outcome"] = {"result": oc, "C1_families_passed": n_c1, "C2_pass": c2_ok, "C3_all": all(c3.values())}
    fb = {}
    for fam in C1_PAIRS:
        if out["C1"][fam].get("pass") or not selected.get(fam):
            continue
        pj = out["per_judge_C1"].get(fam, {})
        wrong = [J for J, v in pj.items() if v["pooled_p_rev"] < TH["C1_p"]]
        keep = [J for J in selected[fam] if J not in wrong]
        if not wrong or not keep:
            fb[fam] = {"dropped": wrong, "kept": keep, "note": "nothing to drop" if not wrong else "nothing left"}
            continue
        cb = {k: comb_arrays(kss[k], keep) for k in combs if kss[k].fam == fam}
        cb = {k: v for k, v in cb.items() if v is not None}
        cs2 = {k: collapse_setup(ctx, kss[k], keep) for k in cb}
        r2 = c1_family(ctx, kss, fam, keep, cb, cs2, routes)
        fb[fam] = {"dropped": wrong, "kept": keep, "C1": {"pass": r2.get("pass"), "pooled": r2.get("pooled")},
                   "label": "ear-tuned (in-sample)"}
    out["fallback"] = fb
    return out


def decoder_ceiling(kss, combs):
    out = {}
    for ks in kss.values():
        if ks.key not in combs:
            continue
        lc = np.asarray(combs[ks.key]["q"], float)
        mu = float(np.nanmean(combs[ks.key]["real"]))
        rec = {}
        for dec in ("raw", "same_l", "same_s"):
            idx = [i for i, r in enumerate(ks.q) if r["kind"] == "d4" and r["meta"]["decoder"] == dec]
            if idx:
                L = float(np.nanmean(lc[idx]) - mu)
                rec[dec] = {"R": math.exp(-L), "n": len(idx)}
        if rec:
            out[ks.key] = rec
    return out


def experiment(ctx, kss, combs, css, cells, routes, args, cfgs):
    base = args.baseline.partition("=")[0]
    out = {"baseline": base, "routes": {}}
    for m in routes:
        if m == base:
            continue
        per, fam_tests = {}, defaultdict(list)
        for ks in kss.values():
            if ks.key not in combs or m not in ks.gi or base not in ks.gi:
                continue
            t = pair_test(ctx, ks, combs[ks.key], css.get(ks.key), m, base)
            if t is None:
                continue
            key = ks.key
            ratio = float(math.exp(t["L_worse"] - t["L_better"]))
            per[key] = {"ratio": ratio, "p_better": t["p"], "p_worse": t["p_rev"], "delta": t["delta"],
                        "collapse": bool(cells.get(("C", key, m), {}).get("collapse"))}
            if ks.status != "proxy":
                fam_tests[ks.fam].append(t)
        fams = {}
        for fam, ts in fam_tests.items():
            po = pooled(ctx, ts)
            coll = any(per[t["key"]]["collapse"] for t in ts)
            if po["ratio"] >= TH["exp_ratio"] and po["p"] < TH["exp_p"] and po["delta"] >= TH["exp_delta"] and not coll:
                call = "Better (pending ear check)"
            elif 1.0 / max(po["ratio"], 1e-9) >= TH["exp_ratio"] and po["p_rev"] < TH["exp_p"] and \
                    -po["delta"] >= TH["exp_delta"]:
                call = "Worse"
            else:
                call = "Same"
            fams[fam] = {"pooled": po, "call": call, "collapse": coll}
        out["routes"][m] = {"keys": per, "families": fams, "decoder_route": rr.decoder_named(cfgs.get(m)),
                            "note": "cache_v1 footstep pools are selection-exposed for decoder routes"
                            if rr.decoder_named(cfgs.get(m)) else None}
    return out


def finish_real(ctx, result, kss, sel_res, args):
    null = {}
    for ks in kss.values():
        if ks.ref is None:
            continue
        null[ks.key] = {J: s0_judge(ctx, ks, J) for J in ks.judges if J in ks.L}
    result["null"] = null
    result["meta"]["runtime_s"] = {k: round(v, 1) for k, v in ctx.stage.items()}
    result["meta"]["runtime_s"]["total"] = round(time.perf_counter() - ctx.t0, 1)
    d = os.path.join(args.out_dir, args.tag)
    os.makedirs(d, exist_ok=True)
    dump(result, os.path.join(d, "results.json"))
    write_csv(os.path.join(d, "clips.csv"), kss, [])
    txt = table_real(result, kss)
    with open(os.path.join(d, "table.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)
    ctx.say("saved %s" % ve.rel(d))


def sel_line(J, r, full):
    parts = []
    s0 = r["S0"]
    q = [v["q05"] for v in s0["keys"].values() if v]
    parts.append("S0 %s (q05 %s)" % ("ok" if s0["pass"] else "FAIL", "/".join(fnum(x) for x in q)))
    s1 = r["S1"]
    tg = []
    for t, v in s1["targets"].items():
        if "auc" in v:
            tg.append("%s auc %s" % (t, fnum(v["auc"])))
        elif "rise_frac" in v:
            tg.append("%s rise %s sp %s d %s" % (t, fnum(v["rise_frac"]), fnum(v["spearman_pos_frac"]),
                                                 fnum(v["delta"])))
        elif "d25_mean" in v:
            tg.append("d25>=d2 %s" % ("ok" if v["pass"] else "no"))
        else:
            tg.append("%s n/a" % t)
    ctl = "ctl ok" if s1["controls"]["pass"] else "ctl FAIL"
    sp = s1.get("sparkle_d1")
    parts.append("S1 %s [%s; %s%s]" % ("ok" if s1["pass"] else "FAIL", "; ".join(tg), ctl,
                                       "" if sp is None else "; sparkle-d1 %s" % ("ok" if sp["pass"] else "FAIL")))
    nu = r["S2"]["nuisance"]
    bad = [k for k, v in nu["kinds"].items() if not v.get("pass")]
    s2 = "S2 nuisance %s" % ("ok" if nu["pass"] else "FAIL %s" % ",".join(bad))
    if full and "flaws" in r["S2"]:
        fl = r["S2"]["flaws"]
        fb = ["%s %s" % (k, fnum(v.get("delta"))) for k, v in fl.get("kinds", {}).items() if v.get("pass") is False]
        s2 += ", flaws %s" % ("ok" if fl["pass"] else "FAIL %s" % ",".join(fb))
    parts.append(s2)
    if full:
        s3 = r["S3"]
        parts.append("S3 %s (AUC %s%s)" % ("ok" if s3["pass"] else "FAIL", fnum(s3["auc"]),
                                           ", more-typical %s" % ",".join(s3["more_typical_cells"])
                                           if s3["more_typical_cells"] else ""))
        s4 = r["S4"]
        parts.append("S4 %s (split rho %s, CI width %s, MDR %s)" % ("ok" if s4["pass"] else "FAIL",
                                                                   fnum(s4["split_spearman"]),
                                                                   fnum(s4["median_ci_width"]), fnum(s4["mdr_n8"])))
    return "   %-4s %s  %s" % (J, ("PASS" if r.get("pass", r.get("pass_real_only")) else "fail"), " | ".join(parts))


def table_real(result, kss):
    L = []
    L.append("SARGE realism judges v1.1, real-only validation (%s)" % result["meta"]["args"].get("tag"))
    L.append("pools")
    for p, P in result["pools"].items():
        q = P["quality"]
        L.append("   %-7s %3d clips %2d sources valid %s  quality %s (dropped clipped %s, rolloff %s)  J9 %s  clipped %d%s"
                 % (p, P["n"], P["sources"], P["valid"], "applied" if q.get("applied") else "kept all",
                    q.get("dropped_clipped", "-"), q.get("dropped_rolloff", "-"), P["j9"], P["clipped"],
                    ("  platforms %s" % P["platforms"]) if P.get("platforms") else ""))
    L.append("J2 preconditions")
    for k, v in result["j2_preconditions"].items():
        L.append("   %s two-click %s (clipped hi %s, lo %s) -> %s" % (k, fnum(v["two_click_rate"]),
                                                                    fnum(v["rate_clipped_hi"]),
                                                                    fnum(v["rate_clipped_lo"]), v["pass"]))
    L.append("J8d")
    for f, v in result["j8d"].items():
        L.append("   %-8s d4 AUC %s (n %s), OOF real-vs-damage AUC %s, m_real %s" % (
            f, fnum(v.get("d4_auc")), v.get("d4_n"), fnum(v.get("oof_auc_real_vs_damage")), fnum(v.get("m_real"))))
    if result.get("j6"):
        j = result["j6"]
        L.append("J6 valid %s  shuffle p %s  LOSO balanced acc %s  S_ref %s" % (
            j.get("valid"), fnum(j.get("shuffle_p"), 3), fnum(j.get("loso_bacc")), fnum(j.get("S_ref"))))
    L.append("selection tests without generated audio (S0, S1, S2 nuisance)")
    for fam, fr in result["selection_tests"].items():
        L.append(" %s" % fam)
        for J, r in fr.items():
            L.append(sel_line(J, r, False))
    L.append("runtime %s" % result["meta"]["runtime_s"])
    return "\n".join(L) + "\n"


def table(result, kss, routes, selected, cnull, gate):
    L = []
    meta = result["meta"]
    L.append("SARGE realism judges v1.1  tag %s  mode %s%s%s" % (meta["args"].get("tag"), meta["mode"],
                                                             "  POST-HOC" if meta.get("post_hoc") else "",
                                                             "  DEV (not a validation)" if meta.get("dev") else ""))
    L.append("pools: " + ", ".join("%s %d/%d%s" % (p, P["n"], P["sources"], "" if P["valid"] else " (below minimum)")
                                   for p, P in result["pools"].items()))
    if result.get("validated"):
        L.append("SELECTION (no ear data)")
        for fam, v in result["validated"]["families"].items():
            L.append("   %-8s %-11s selected %s  combined %s" % (fam, v["status"], ",".join(v["selected"]) or "-",
                                                               "PASS" if v.get("combined_pass") else
                                                               "fail" if v.get("combined_pass") is False else "-"))
        for fam, fr in result["selection_tests"].items():
            L.append(" %s" % fam)
            for J, r in fr.items():
                L.append(sel_line(J, r, True))
        for fam, c in result.get("combined_checks", {}).items():
            if "d4" not in c:
                continue
            L.append("   combined %-8s d4 %s/%s  d6 %s  P1 %s (d3 %s < d2 %s)  S2 %s  S3 AUC %s %s  S4 rho %s width %s -> %s"
                     % (fam, c["d4"]["raised"], c["d4"]["pairs"], "ok" if c["d6"]["pass"] else "FAIL",
                        "ok" if c["P1"]["pass"] else "FAIL", fnum(c["P1"]["d3_mean_rise"]),
                        fnum(c["P1"]["d2_tau3_mean_rise"]), "ok" if c["S2"]["pass"] else "FAIL", fnum(c["S3"]["auc"]),
                        "ok" if c["S3"]["pass"] else "FAIL", fnum(c["S4"]["split_spearman"]),
                        fnum(c["S4"]["median_ci_width"]), "PASS" if c["pass"] else "FAIL"))
    L.append("")
    L.append("RE-SCORED BAKE-OFF  combined R = exp(-(mean lat gen - mean lat real) - collapse penalty); verdict R/C/A/X, "
             "lower-case = tentative; band = null q10 / 0.75 x q01")
    W = 14
    L.append("%-20s %-14s " % ("key", "band") + "".join("%-*s" % (W, m[:W - 1]) for m in routes))
    for ks in kss.values():
        cn = cnull.get(ks.key)
        band = "%s/%s" % (fnum(cn["q10"]), fnum(TH["verdict_C_frac"] * cn["q01"])) if cn else "-"
        line = "%-20s %-14s " % (ks.key, band)
        why = []
        for m in routes:
            c = result["cells"].get("%s|%s" % (m, ks.key), {})
            if "R" in c:
                v = c["verdict"] if c["firm"] else c["verdict"].lower()
                txt = "%s %s%s" % (fnum(c["R"]), v, "*" if c.get("collapse") else "")
                if c["verdict"] in ("A", "X"):
                    txt2 = "; ".join(c.get("reason", [])) or "no z-feature judge selected"
                    if c["verdict"] == "X":
                        txt2 = "identity gate failed (cons %s); %s" % (fnum(c.get("cons")), txt2)
                    why.append("      %s: %s" % (m, txt2))
            elif "artifact_R" in c:
                txt = "art %s" % fnum(c["artifact_R"])
            else:
                txt = "-"
            line += "%-*s" % (W, txt)
        L.append(line)
        if ks.fam == "gun":
            L.append("      clipped pool: real C3GD shots are 97-100% clipped at the source")
        L += why[:4]
    L.append("   * = collapse penalty applied")
    L.append("")
    L.append("PER-JUDGE R (selected judges)")
    for fam in rr.FAMILIES:
        sel = selected.get(fam, [])
        if not sel:
            continue
        for ks in kss.values():
            if ks.fam != fam or ks.ref is None:
                continue
            for J in sel:
                line = "%-20s %-4s " % (ks.key, J)
                for m in routes:
                    c = result["cells"].get("%s|%s" % (m, ks.key), {}).get("R_J", {}).get(J)
                    line += "%-*s" % (W, fnum(c))
                L.append(line)
    L.append("")
    L.append("ALL CANDIDATE JUDGES, R per key (diagnostic only: includes judges that FAILED selection)")
    for fam, fr in result.get("selection_tests", {}).items():
        for J, r in fr.items():
            cells = r.get("cells") or {}
            keys_ = sorted({k.split("|", 1)[1] for k in cells}, key=list(rr.KEYS).index)
            for k in keys_:
                line = "%-20s %-4s %-4s " % (k, J, "ok" if r.get("pass") else "fail")
                for m in routes:
                    c = cells.get("%s|%s" % (m, k))
                    line += "%-*s" % (W, fnum(c["R"]) if c else "-")
                L.append(line)
    L.append("")
    L.append("OLD IDENTITY JUDGES (vocab_eval cons, gate >= 0.5) and Audiobox PQ (context only)")
    for ks in kss.values():
        line = "%-20s %-4s " % (ks.key, "cons")
        line2 = "%-20s %-4s " % ("", "PQ")
        for m in routes:
            c = result["cells"].get("%s|%s" % (m, ks.key), {})
            line += "%-*s" % (W, fnum(c.get("cons")) if c.get("cons") is not None else "?")
            line2 += "%-*s" % (W, fnum(c.get("pq")) if c.get("pq") is not None else "-")
        L.append(line)
        L.append(line2)
    if result.get("j6"):
        j = result["j6"]
        L.append("")
        L.append("J6 surfaces: valid %s (shuffle p %s, LOSO bacc %s)" % (j.get("valid"), fnum(j.get("shuffle_p"), 3),
                                                                       fnum(j.get("loso_bacc"))))
        for m, v in j.get("models", {}).items():
            L.append("   %-13s R_J6 %s [%s,%s] lda ratio %s%s%s" % (
                m, fnum(v["R"]), fnum(v["ci"][0]), fnum(v["ci"][1]), fnum(v["lda_ratio"]),
                "  surfaces converge" if v["converge"] else "",
                "  metal distinctness %s" % fnum(v["metal_distinctness"]) if "metal_distinctness" in v else ""))
    if result.get("water_collapse"):
        L.append("water keys collapse (LDA LOO acc <= 0.6): " + ", ".join(
            "%s %s%s" % (m, fnum(v["loo_acc"]), " COLLAPSE" if v["collapse"] else "")
            for m, v in result["water_collapse"].items()))
    sc = result.get("stage_c")
    if sc:
        L.append("")
        L.append("STAGE C (ear confirmation, never feeds back)")
        for fam, r in sc["C1"].items():
            if not r.get("testable"):
                L.append("   C1 %-8s untestable (%s)" % (fam, r.get("why", "")))
                continue
            po = r["pooled"]
            L.append("   C1 %-8s %s  correct %d/%d need %d  pooled p %s ratio %s delta %s  halves %s%s" % (
                fam, "PASS" if r["pass"] else "FAIL", r["correct"], sum(1 for x in r["pairs"] if x.get("testable")),
                r["need"], fnum(po["p"], 4), fnum(po["ratio"]), fnum(po["delta"]),
                "/".join(fnum(v) for v in r["halves"].values()),
                ("  carried by %s" % ",".join(r["carried_by"])) if r.get("carried_by") else ""))
            for x in r["pairs"]:
                if x.get("testable"):
                    L.append("        %-11s > %-11s %-20s ratio %s p %s p_rev %s delta %s%s" % (
                        x["pair"][0], x["pair"][1], x["pair"][2], fnum(x["ratio"]), fnum(x["p"], 4), fnum(x["p_rev"], 4),
                        fnum(x["delta"]), "  WRONG SIGN" if not x["correct"] else ""))
            pj = sc["per_judge_C1"].get(fam, {})
            if pj:
                L.append("        per judge (diagnosis): " + "  ".join(
                    "%s %d/%s p %s r %s" % (J, v["correct"], r["need"], fnum(v["pooled_p"], 3), fnum(v["pooled_ratio"]))
                    for J, v in pj.items()))
        for k, v in sc["E"].items():
            L.append("   %-4s %-12s %s" % (k, ("CONFIRM" if v.get("confirm") else "no") if v.get("testable")
                                         else "untestable", v["text"]))
        L.append("   C3 %s" % sc["C3"])
        L.append("   OUTCOME %s  (C1 families passed %d, C2 %s, C3 %s)" % (
            sc["outcome"]["result"], sc["outcome"]["C1_families_passed"], sc["outcome"]["C2_pass"],
            sc["outcome"]["C3_all"]))
        if sc.get("fallback"):
            L.append("   fallback (ear-tuned, in-sample): %s" % {f: {"dropped": v["dropped"], "C1": (v.get("C1") or {})
                                                                     .get("pass")} for f, v in sc["fallback"].items()})
    if result.get("decoder_ceiling"):
        L.append("")
        L.append("decoder ceiling (anchor round trips, combined R): " + "; ".join(
            "%s %s" % (k, " ".join("%s %s" % (d, fnum(v["R"])) for d, v in r.items()))
            for k, r in result["decoder_ceiling"].items()))
    ex = result.get("experiment")
    if ex:
        L.append("")
        L.append("EXPERIMENT vs %s" % ex["baseline"])
        for m, r in ex["routes"].items():
            for fam, v in r["families"].items():
                L.append("   %-12s %-8s %s  pooled ratio %s p %s delta %s" % (
                    m, fam, v["call"], fnum(v["pooled"]["ratio"]), fnum(v["pooled"]["p"], 4),
                    fnum(v["pooled"]["delta"])))
            for k, v in r["keys"].items():
                L.append("        %-20s ratio %s p_better %s delta %s%s" % (k, fnum(v["ratio"]), fnum(v["p_better"], 4),
                                                                         fnum(v["delta"]),
                                                                         " collapse" if v["collapse"] else ""))
    L.append("")
    L.append("runtime %s" % meta.get("runtime_s"))
    return "\n".join(L) + "\n"


def cmd_prereg(args):
    md5 = core_md5()
    fams = {}
    for fam in rr.FAMILIES:
        fams[fam] = {J: {"features": [(f, rf.direction(f)) for f in judge_feats(J, fam)] if JUDGES[J]["kind"] == "z"
                         else JUDGES[J]["kind"], "bounded": JUDGES[J]["bounded"], "targets": JUDGES[J]["targets"],
                         "window": JUDGES[J]["win"]}
                     for J in JUDGES if fam in JUDGES[J]["fams"]}
    ear = {"source": "the user's ear verdicts on SA3 small-sfx Unity/listening pages and the blind 9-model bake-off",
           "C1": {f: [list(p) for p in v] for f, v in C1_PAIRS.items()}, "C1_correct_needed": TH["C1_correct"],
           "E": E_TEXT, "C3": C3_SET}
    dump(ear, EAR)
    pre = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "version": "realism judge suite v1.1", "md5": md5,
           "feats_md5": rf.feats_md5(), "chain": rr.CHAIN_VERSION, "thresholds": TH, "candidates": fams,
           "pools": {p: {k: v for k, v in s.items()} for p, s in rr.POOLS.items()},
           "keys": rr.KEYS, "union_pool": "every valid real pool except reload, J4 only",
           "pool_rules": {"min_clips": rr.MIN_CLIPS, "min_sources": rr.MIN_SOURCES,
                          "quality": "drop clipped_frac > 1e-3 or rolloff99 < 11 kHz when >= 20 clips / 4 sources "
                                     "remain (not on guns)", "footstep_crops": {
                              "prev_s": rr.CROP_PREV_S, "next_s": rr.CROP_NEXT_S, "pre_s": rr.CROP_PRE_S,
                              "cut_before_next_s": rr.CROP_CUT_S, "max_per_recording": rr.CROP_MAX}},
           "damage": {"d1_gains": rr.D1_GAINS, "d2_taus_ms": rr.D2_TAUS, "d3": "12 kHz order 8 zero-phase",
                      "d5_snr_db": rr.D5_SNRS, "d25": "tau 3 ms then SNR 20 dB", "d4": "anchor raw vs same_l, same_s",
                      "d6": "8 copies of one real clip, +-2 ms shift, +-0.5 dB", "d8": "30 ms fade ending at 0.5 s",
                      "per_key": rr.DAMAGE_N},
           "nuisance": rr.NUISANCE, "flaws": rr.FLAWS, "flaw_seeds": rr.FLAW_SEEDS,
           "C1_pairs": {f: [list(p) for p in v] for f, v in C1_PAIRS.items()}, "E_tests": E_TEXT, "C3": C3_SET,
           "P1": "mean lat_comb increase under d3 is smaller than under d2 at tau = 3 ms",
           "phase2": {"slots": TH["slots"], "V6_within_key": TH["V6_within_key"],
                      "V6_within_model": TH["V6_within_model"], "page": "12 keys x 9 models x 2 seeds + 3 real per pool"
                                                                        " key + 12 repeats, W1.0 judge format, "
                                                                        "BS.1770 loudness matched"},
           "known_before_freezing": KNOWN_BEFORE, "implementation_interpretations": INTERPRETATIONS,
           "runs_before_freeze": sorted(d for d in os.listdir(OUT) if os.path.isfile(os.path.join(OUT, d, "results.json")))
           if os.path.isdir(OUT) else [],
           "dev_runs_before_freeze": sorted(d for d in os.listdir(os.path.join(OUT, "dev"))
                                            if os.path.isfile(os.path.join(OUT, "dev", d, "results.json")))
           if os.path.isdir(os.path.join(OUT, "dev")) else []}
    dump(pre, PREREG)
    print("wrote %s and %s" % (ve.rel(PREREG), ve.rel(EAR)))
    print(json.dumps(md5, indent=1))
    return 0


def parse(argv):
    ap = argparse.ArgumentParser(description="SARGE realism judge suite v1.1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "score", "prereg"):
        p = sub.add_parser(name)
        p.add_argument("--route", action="append", default=[])
        p.add_argument("--bakeoff", default=None)
        p.add_argument("--models", nargs="+", default=None)
        p.add_argument("--raw", action="append", default=[])
        p.add_argument("--baseline", default=None)
        p.add_argument("--keys", nargs="+", default=None)
        p.add_argument("--seeds", type=int, default=8)
        p.add_argument("--refs", default="heldout", choices=["heldout", "heldout+refs"])
        p.add_argument("--judges", nargs="+", default=None)
        p.add_argument("--embed", default="panns", choices=["panns", "none"])
        p.add_argument("--identity", nargs="+", default=[])
        p.add_argument("--water-identity", default=None)
        p.add_argument("--boot", type=int, default=2000)
        p.add_argument("--rng", type=int, default=20261014)
        p.add_argument("--real-only", action="store_true")
        p.add_argument("--dev", action="store_true")
        p.add_argument("--force-select", nargs="+", default=None)
        p.add_argument("--out-dir", default=OUT)
        p.add_argument("--tag", default=None)
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--budget-scale", type=float, default=1.0)
    args = ap.parse_args(argv)
    if args.out_dir and not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(ROOT, args.out_dir)
    if args.tag is None:
        args.tag = "real_dev" if args.real_only else "run_%s" % time.strftime("%Y%m%d_%H%M")
    if args.judges:
        for J in list(JUDGES):
            if J not in args.judges:
                JUDGES.pop(J)
    return args


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    if args.cmd == "prereg":
        return cmd_prereg(args)
    if args.cmd == "validate":
        run(args, "real" if args.real_only else "full")
        return 0
    if args.cmd == "score":
        run(args, "score")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
