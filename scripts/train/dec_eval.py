import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
BENCH = os.path.join(SCRIPTS, "bench")
for _p in (BENCH, SCRIPTS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap
import run_sidecar

import argparse
import gc
import glob
import json
import time
from collections import Counter, defaultdict

import numpy as np
from scipy.io import wavfile

import dec_common as dc
import torch

NBOOT = 2000
SEEDS = (0, 1, 2)
GROUP_METRICS = {
    "hfk_win": "kurt", "hfk_frame": "kurt", "spike_rate": "kurt",
    "mrstft": "dist", "mel_dist": "dist", "mel2k": "dist", "lsd_0-1": "dist", "lsd_1-4": "dist", "lsd_4-8": "dist",
    "lsd_8-16": "dist", "lsd_16-22": "dist", "rise_err_ms": "dist", "crest_err_db": "dist",
    "si_sdr": "diff", "hf_env_corr": "diff", "hf_floor_db": "diff", "hf_energy_db": "diff", "gap_hiss_db": "diff",
    "pre_dec_db": "diff",
}


def eval_dir(tag):
    return os.path.join(dc.EVAL, tag)


def load_index():
    return dc.load_json(os.path.join(dc.CACHE, "index.json"))


def clip_audio(split, ents):
    mm = np.memmap(os.path.join(dc.CACHE, "%s.f32" % split), dtype=np.float32, mode="r")
    return [np.asarray(mm[e["off"]:e["off"] + e["n"]]) for e in ents]


def write_wav(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wavfile.write(path, dc.SR, np.asarray(x, np.float32))


def gen_kwargs(tier="default", ckpt=None, compile_=True):
    kw = dict(run_sidecar.TIERS[tier], fast_load=True)
    if not compile_:
        for k in run_sidecar.COMPILED:
            if k not in ("graph_decode", "decode_prefix"):
                kw.pop(k, None)
    if ckpt:
        kw["decoder_ckpt"] = dc.at_root(ckpt)
    return kw


def bank_items():
    ev = [(et, mat, "normal", v) for et, mat in dc.EVAL_KEYS for v in range(20)]
    ev += [("footstep", s, it, v) for s in dc.FOOT_SURF for it in ("soft", "hard") for v in range(10)]
    an = [(et, mat, it, v) for et, mat in dc.EVAL_KEYS if "%s|%s" % (et, mat) not in dc.ANCHOR_HELD
          for it in ("soft", "normal", "hard") for v in range(1000, 1040)]
    av = [(et, mat, "normal", v) for k in dc.ANCHOR_HELD for et, mat in [k.split("|")] for v in range(1000, 1032)]
    return {"eval": ev, "anchor": an, "anchor_val": av}


def onset_time(m, db=-20.0):
    e = dc.env_ms(m, 5.0, 1.0)
    if not len(e) or e.max() <= 0:
        return float("nan")
    return float(np.argmax(e >= e.max() * 10 ** (db / 20.0)) * 0.001)


def latent_stats(zg, ze):
    def st(z):
        z = z.astype(np.float64)
        ch_mean = z.mean(axis=(0, 2))
        ch_std = z.std(axis=(0, 2))
        c = z - ch_mean[None, :, None]
        ac = (c[:, :, 1:] * c[:, :, :-1]).sum(axis=(0, 2)) / np.maximum((c * c).sum(axis=(0, 2)), 1e-12)
        fd = float(np.mean(np.sum(np.diff(z, axis=2) ** 2, axis=1)))
        nrm = np.linalg.norm(z, axis=1).ravel()
        return ch_mean, ch_std, ac, fd, nrm
    gm, gs, ga, gf, gn = st(zg)
    em, es, ea, ef, en = st(ze)
    r = gs / np.maximum(es, 1e-9)
    pc = [5, 25, 50, 75, 95]
    return {"items": {"dit": int(zg.shape[0]), "encoded": int(ze.shape[0])}, "frames": int(zg.shape[2]),
            "std_ratio_pct": dict(zip(map(str, pc), np.percentile(r, pc).round(4).tolist())),
            "mean_shift_over_std_pct": dict(zip(map(str, pc), np.percentile((gm - em) / np.maximum(es, 1e-9), pc)
                                                .round(4).tolist())),
            "lag1_autocorr": {"dit": float(ga.mean()), "encoded": float(ea.mean())},
            "frame_diff_energy_ratio": gf / max(ef, 1e-12),
            "norm_pct": {"dit": dict(zip(map(str, pc), np.percentile(gn, pc).round(3).tolist())),
                         "encoded": dict(zip(map(str, pc), np.percentile(en, pc).round(3).tolist()))}}


def cmd_record(args):
    from sarge.generators.stable_audio3 import StableAudio3Generator
    from sarge.static_guard import is_static
    from sarge.types import CacheKey
    import dec_train as dt
    os.makedirs(dc.GEN, exist_ok=True)
    with dc.gpu_guard("record"):
        kw = dict(run_sidecar.TIERS["default"], fast_load=True, static_guard=False)
        gen = StableAudio3Generator(**kw)
        t0 = time.perf_counter()
        gen.ensure_ready()
        print("generator ready %.1f s" % (time.perf_counter() - t0), flush=True)
        pt = gen._model.model.pretransform
        inner = pt.decode
        cap = {}

        def rec(z, chunked=None):
            out = inner(z, chunked=chunked)
            cap["z"] = z.detach().half().cpu()
            cap["y"] = out.detach().float().cpu()
            return out

        pt.decode = rec
        n1 = int(gen.gen_duration_s * dc.SR)
        summary = {}
        allpk, allon, allsm = [], [], []
        banks = bank_items()
        for name, items in banks.items():
            if args.limit:
                items = items[:args.limit]
            zs, keys, pk, sm, on, stc = [], [], [], [], [], []
            t1 = time.perf_counter()
            for i, (et, mat, it, v) in enumerate(items):
                clip = gen.generate(CacheKey(et, mat, it), variant=v)
                z = cap["z"][0].numpy()
                y = cap["y"][0][:, :n1].numpy().astype(np.float64)
                m = y.mean(0)
                s = 0.5 * (y[0] - y[1])
                zs.append(z)
                keys.append("%s|%s|%s|%d" % (et, mat, it, v))
                pk.append(float(np.max(np.abs(y))))
                sm.append(float(10 * np.log10((np.mean(s * s) + 1e-20) / (np.mean(m * m) + 1e-20))))
                on.append(onset_time(m))
                stc.append(bool(is_static(clip.samples, dc.SR)[0]))
                if (i + 1) % 200 == 0:
                    print("record %s %d/%d %.1f s" % (name, i + 1, len(items), time.perf_counter() - t1), flush=True)
            np.savez(os.path.join(dc.GEN, "%s.npz" % name), z=np.stack(zs).astype(np.float16), keys=np.array(keys),
                     peak=np.array(pk), side_mid_db=np.array(sm), onset_s=np.array(on), static=np.array(stc))
            summary[name] = {"latents": len(zs), "shape": list(np.stack(zs).shape), "static": int(sum(stc)),
                             "peak_dbfs_pct": np.percentile(20 * np.log10(np.maximum(pk, 1e-9)), [5, 50, 95]).round(2).tolist(),
                             "side_mid_db_median": float(np.median(sm)), "onset_s_pct": np.nanpercentile(on, [5, 50, 95]).round(3).tolist(),
                             "seconds": round(time.perf_counter() - t1, 1)}
            print("record %s %s" % (name, json.dumps(summary[name])), flush=True)
            if name != "anchor_val":
                allpk += pk
                allon += on
                allsm += sm
        stats_fast = gen.fast_stats()
        del gen
        gc.collect()
        torch.cuda.empty_cache()
        pdb = 20 * np.log10(np.maximum(np.array(allpk), 1e-9))
        lo, hi = np.percentile(pdb, [5, 95])
        ons = np.array(allon)
        olo, ohi = np.nanpercentile(ons, [5, 95])
        sug = {"gain_db": [round(float(lo), 2), round(float(min(hi, 3.0)), 2)],
               "onset_s": [round(float(np.clip(olo, 0.0, 0.6)), 3), round(float(np.clip(max(ohi, olo + 0.01), 0.0, 0.6)), 3)],
               "pseudo_stereo": bool(np.median(allsm) > -20.0), "side_mid_db_median": float(np.median(allsm))}
        ae = dc.load_ae("cuda")
        index = load_index()
        data = dt.Data(index, "train")
        smp = dt.Sampler(data, 1234, (-20.0, -1.0), tuple(sug["onset_s"]), 0.0, False)
        rng = np.random.default_rng(99)
        n_enc = min(args.encode, 512)
        xs = []
        for _ in range(n_enc):
            m = smp.mixes[int(rng.choice(len(smp.mixes), p=smp.mix_p))]
            seg, _mask, p0 = smp.draw(smp.pick(m))
            tgt = float(np.array(allpk)[rng.integers(len(allpk))])
            seg = seg * np.float32(tgt / max(p0, 1e-9))
            xs.append(np.stack([seg, seg]))
        ze = []
        with torch.no_grad():
            for i in range(0, len(xs), 16):
                ze.append(ae.encode(torch.from_numpy(np.stack(xs[i:i + 16])).cuda()).float().cpu().numpy())
        ze = np.concatenate(ze)[..., :dc.SHORT_FRAMES]
        zg = np.concatenate([np.load(os.path.join(dc.GEN, "%s.npz" % n))["z"] for n in ("eval", "anchor")])
        zg = zg[..., :dc.SHORT_FRAMES].astype(np.float32)
        ls = latent_stats(zg, ze)
        ls["suggest"] = sug
        dc.dump_json(ls, os.path.join(dc.GEN, "latent_stats.json"))
        dc.dump_json({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "snapshot": dc.SNAP_ID, "tier": "default",
                      "tier_kwargs": kw, "torch": torch.__version__, "banks": summary, "suggest": sug,
                      "fast_stats": stats_fast, "key_format": "event|material|intensity|variant",
                      "z": "[N, 256, 22] fp16 latents at the outermost pretransform.decode; production decodes [..., :13]"},
                     os.path.join(dc.GEN, "index.json"))
        print("LATENT_STATS %s" % json.dumps(ls), flush=True)
        print("SUGGEST %s" % json.dumps(sug), flush=True)
    return 0


def det_batch(auds):
    return np.stack([dc.det_crop(a)[0] for a in auds])


def transient_val(index, limit=None):
    ents = [e for e in index["val"] if e["mix"] in dc.TRANSIENT and not e["flat"]]
    ents = sorted(ents, key=lambda e: dc.md5_text(e["id"]))
    return ents[:limit] if limit else ents


def cmd_teacher_probe(args):
    index = load_index()
    ents = transient_val(index, args.limit)
    auds = clip_audio("val", ents)
    x_all = det_batch(auds)
    out_rows = []
    with dc.gpu_guard("teacher-probe"):
        aes = dc.load_ae("cuda")
        ael = dc.load_ae("cuda", dc.SNAP_L)
        rs_s = float(aes.bottleneck.running_std)
        rs_l = float(ael.bottleneck.running_std)
        bs = 4
        for i in range(0, len(x_all), bs):
            x = torch.from_numpy(x_all[i:i + bs]).cuda()
            with torch.no_grad():
                zs = aes.encode(x)
                outs = defaultdict(list)
                for s in (0, 1):
                    torch.manual_seed(100 + s)
                    zl = ael.encode(x)
                    outs["s_rt"].append(dc.mid(dc.seeded_decode(aes, aes.decoder, zs, s)).double().cpu().numpy())
                    outs["l_zs"].append(dc.mid(dc.seeded_decode(ael, ael.decoder, zs, s)).double().cpu().numpy())
                    outs["l_zs_rescaled"].append(dc.mid(dc.seeded_decode(ael, ael.decoder, zs * rs_s / rs_l, s))
                                                 .double().cpu().numpy())
                    outs["l_rt"].append(dc.mid(dc.seeded_decode(ael, ael.decoder, zl, s)).double().cpu().numpy())
            for j in range(x.shape[0]):
                ref = x_all[i + j, 0].astype(np.float64)
                h = dc.hp4k(ref)
                win = dc.window(ref, dc.SR, 0.6, 0.03)
                idx = dc.active_frames(h)
                kw, kf = dc.kurt(h[win[0]:win[1]]), dc.hfk_frame(h, idx)
                row = {"id": ents[i + j]["id"], "bucket": ents[i + j]["bucket"]}
                for k, v in outs.items():
                    lw, lf, ml = [], [], []
                    for y in v:
                        d = y[j]
                        hd = dc.hp4k(d)
                        lw.append(abs(float(dc.log_ratio(dc.kurt(hd[win[0]:win[1]]), kw))))
                        lf.append(abs(float(dc.log_ratio(dc.hfk_frame(hd, idx), kf))))
                        ml.append(dc.mel_dist(ref, d, 2000.0))
                    row[k] = {"abs_log_hfk_win": float(np.nanmean(lw)), "abs_log_hfk_frame": float(np.nanmean(lf)),
                              "mel2k": float(np.mean(ml))}
                out_rows.append(row)
            print("probe %d/%d" % (min(i + bs, len(x_all)), len(x_all)), flush=True)
    med = {k: {m: float(np.nanmedian([r[k][m] for r in out_rows])) for m in ("abs_log_hfk_win", "abs_log_hfk_frame", "mel2k")}
           for k in ("s_rt", "l_zs", "l_zs_rescaled", "l_rt")}
    pairing = None
    for k in ("l_zs", "l_zs_rescaled"):
        if med[k]["mel2k"] <= 1.2 * med["s_rt"]["mel2k"]:
            pairing = k
            break
    passed = bool(pairing and all(med[pairing][m] <= 0.5 * med["s_rt"][m] for m in ("abs_log_hfk_win", "abs_log_hfk_frame")))
    rep = {"clips": len(out_rows), "medians": med, "running_std": {"same_s": rs_s, "same_l": rs_l},
           "pairing": {"l_zs": "direct", "l_zs_rescaled": "rescaled", None: None}[pairing], "teacher_on": passed,
           "rule": "pairing holds when SAME-L(z_S) mel distance below 2 kHz <= 1.2 x the SAME-S round trip (direct, "
                   "then z_S * rs_S / rs_L); pass when the paired median |log k-ratio| <= 0.5 x the SAME-S round trip on "
                   "hfk_win and hfk_frame", "rows": out_rows}
    dc.dump_json(rep, os.path.join(dc.EVAL, "teacher_probe.json"))
    print("TEACHER_PROBE %s" % json.dumps({k: v for k, v in rep.items() if k != "rows"}), flush=True)
    return 0


def cmd_teacher_cache(args):
    tp = dc.load_json(os.path.join(dc.EVAL, "teacher_probe.json"))
    if not tp["teacher_on"] and not args.force:
        raise SystemExit("teacher-probe did not pass; pass --force to cache anyway")
    d = np.load(os.path.join(dc.GEN, "anchor.npz"))
    z = d["z"][..., :dc.SHORT_FRAMES].astype(np.float32)
    if args.limit:
        z = z[:args.limit]
    n = len(z)
    path = os.path.join(dc.GEN, "samel.f16")
    mm = np.memmap(path + ".tmp", dtype=np.float16, mode="w+", shape=(n, 2, 2, dc.SHORT_SEG))
    d_l, d_sl = [], []
    with dc.gpu_guard("teacher-cache"):
        aes = dc.load_ae("cuda")
        ael = dc.load_ae("cuda", dc.SNAP_L)
        sc = float(aes.bottleneck.running_std) / float(ael.bottleneck.running_std) if tp["pairing"] == "rescaled" else 1.0
        bs = 8
        with torch.no_grad():
            for b, i in enumerate(range(0, n, bs)):
                zb = torch.from_numpy(z[i:i + bs]).cuda()
                ys = []
                for s in (0, 1):
                    y = dc.seeded_decode(ael, ael.decoder, zb * sc, 1000 * s + b)[..., :dc.SHORT_SEG]
                    ys.append(y)
                    mm[i:i + bs, s] = y.float().cpu().numpy().astype(np.float16)
                if b % 8 == 0:
                    yb = dc.seeded_decode(aes, aes.decoder, zb, b)[..., :dc.SHORT_SEG]
                    d_l.append(dc.mrstft(dc.mid(ys[0]), dc.mid(ys[1]), per_res=True).cpu().numpy())
                    d_sl.append(dc.mrstft(dc.mid(yb), dc.mid(ys[0]), per_res=True).cpu().numpy())
                if (b + 1) % 50 == 0:
                    print("teacher-cache %d/%d" % (min(i + bs, n), n), flush=True)
    mm.flush()
    del mm
    os.replace(path + ".tmp", path)
    dl, dsl = np.mean(d_l, 0), np.mean(d_sl, 0)
    res = [int(f) for f, a, b in zip(dc.FFTS, dl, dsl) if a < 0.5 * b]
    dc.dump_json({"resolutions": res, "ffts": list(dc.FFTS), "samel_self": dl.tolist(), "same_s_vs_samel": dsl.tolist(),
                  "pairing": tp["pairing"], "latents": n, "shape": [n, 2, 2, dc.SHORT_SEG]},
                 os.path.join(dc.GEN, "teacher_res.json"))
    print("TEACHER_RES %s self %s vs %s" % (res, dl.round(3).tolist(), dsl.round(3).tolist()), flush=True)
    return 0


def wmedian(v, W):
    o = np.argsort(v)
    vs = v[o]
    cw = np.cumsum(W[:, o], 1)
    idx = (cw >= cw[:, -1:] / 2.0).argmax(1)
    return vs[idx]


def boot_weights(clusters, nboot=NBOOT, seed=0):
    names = sorted(set(clusters))
    pos = {c: i for i, c in enumerate(names)}
    cl = np.array([pos[c] for c in clusters])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(names), (nboot, len(names)))
    cnt = np.zeros((nboot, len(names)))
    np.add.at(cnt, (np.arange(nboot)[:, None], draw), 1.0)
    return cnt[:, cl], len(names)


def ci(a):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return [None, None]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def group_stats(rows, have_new):
    out = {"clips": len(rows), "sources": len({r["cluster"] for r in rows})}
    if not rows:
        return out
    for m, kind in GROUP_METRICS.items():
        b = np.array([r["base"].get(m, np.nan) for r in rows], np.float64)
        n = np.array([r["new"].get(m, np.nan) for r in rows], np.float64) if have_new else None
        ref = np.array([r["base"].get(m + "_ref", np.nan) for r in rows], np.float64)
        if kind == "kurt":
            lb = dc.log_ratio(b, ref)
            ln = dc.log_ratio(n, ref) if have_new else None
            ok = np.isfinite(lb) & (np.isfinite(ln) if have_new else True)
        else:
            ok = np.isfinite(b) & (np.isfinite(n) if have_new else True)
        if ok.sum() < 2:
            continue
        cl = [r["cluster"] for r, k in zip(rows, ok) if k]
        W, k_src = boot_weights(cl)
        e = {"n": int(ok.sum())}
        if kind == "kurt":
            ab, sb = np.abs(lb[ok]), lb[ok]
            e["base_med_abs_log"] = float(np.median(ab))
            e["base_med_log"] = float(np.median(sb))
            e["ref_median"] = float(np.median(ref[ok]))
            e["base_median"] = float(np.median(b[ok]))
            if have_new:
                an, sn = np.abs(ln[ok]), ln[ok]
                e["new_med_abs_log"] = float(np.median(an))
                e["new_med_log"] = float(np.median(sn))
                e["new_median"] = float(np.median(n[ok]))
                e["reduction"] = 1.0 - e["new_med_abs_log"] / max(e["base_med_abs_log"], 1e-12)
                e["diff_ci"] = ci(wmedian(an, W) - wmedian(ab, W))
        else:
            bb = b[ok]
            e["base"] = float(np.mean(bb))
            if have_new:
                nn = n[ok]
                e["new"] = float(np.mean(nn))
                mb, mn = (W @ bb) / W.sum(1), (W @ nn) / W.sum(1)
                if kind == "dist":
                    e["ratio"] = e["new"] / max(e["base"], 1e-12)
                    e["ratio_ci"] = ci(mn / np.maximum(mb, 1e-12))
                else:
                    e["diff"] = e["new"] - e["base"]
                    e["diff_ci"] = ci(mn - mb)
        out[m] = e
    return out


def groups_of(rows):
    g = {"all": lambda r: True,
         "TRANSIENT": lambda r: dc.mix_of(r["bucket"]) in dc.TRANSIENT,
         "TRANSIENT_clean": lambda r: dc.mix_of(r["bucket"]) in dc.TRANSIENT and not r["flat"],
         "TRANSIENT_flat": lambda r: dc.mix_of(r["bucket"]) in dc.TRANSIENT and r["flat"],
         "SMOOTH": lambda r: dc.mix_of(r["bucket"]) in dc.SMOOTH,
         "flat": lambda r: r["flat"], "clean": lambda r: not r["flat"],
         "hear": lambda r: r["dataset"] == "hear", "gun_clean": lambda r: r["subset"] == "clean"}
    for b in sorted({(r["bucket"], r["subset"]) for r in rows}):
        g["bucket:%s|%s" % b] = (lambda bb: (lambda r: (r["bucket"], r["subset"]) == bb))(b)
    for b in sorted({r["bucket"] for r in rows}):
        g["bucket:%s" % b] = (lambda bb: (lambda r: r["bucket"] == bb))(b)
    return g


def summarize_rows(rows, have_new):
    return {name: group_stats([r for r in rows if f(r)], have_new) for name, f in groups_of(rows).items()}


def mean_metrics(ms):
    keys = {k for m in ms for k, v in m.items() if isinstance(v, (int, float))}
    return {k: float(np.nanmean([m[k] for m in ms if isinstance(m.get(k), (int, float))])) for k in keys}


def synthetic_tests(ae, decs, n_frames=40):
    rng = np.random.default_rng(7)
    n = n_frames * dc.DS
    items = []
    for i in range(6):
        c, f = dc.make_chirp(rng, n)
        items.append(("chirp", c, f))
    for i in range(2):
        w = rng.standard_normal(n)
        items.append(("white", (0.1 * w / w.std()).astype(np.float32), None))
        W = np.fft.rfft(rng.standard_normal(n))
        fr = np.fft.rfftfreq(n, 1 / dc.SR)
        W[1:] /= np.sqrt(fr[1:])
        W[0] = 0
        p = np.fft.irfft(W, n)
        items.append(("pink", (0.1 * p / p.std()).astype(np.float32), None))
    items.append(("silence", np.zeros(n, np.float32), None))
    x = torch.from_numpy(np.stack([np.stack([a, a]) for _k, a, _f in items])).cuda()
    out = {}
    with torch.no_grad():
        z = ae.encode(x)
        for name, dec in decs.items():
            res = defaultdict(list)
            for s in SEEDS:
                y = dc.mid(dc.seeded_decode(ae, dec, z, 50 + s)).double().cpu().numpy()
                for (kind, a, f), d in zip(items, y):
                    if kind == "chirp":
                        res["chirp_alias_db"].append(dc.chirp_alias_db(d, f))
                    elif kind == "silence":
                        res["silence_hiss8k_db"].append(dc.energy_db(dc.hp8k(d)))
                        res["silence_comb_db"].append(dc.comb_172(d) if np.any(d) else float("nan"))
                    else:
                        res["noise_comb_db"].append(dc.comb_172(d))
                        res["noise_flutter_db"].append(dc.flutter_538(d))
            out[name] = {k: float(np.nanmean(v)) for k, v in res.items()}
    out["ref"] = {"noise_comb_db": float(np.mean([dc.comb_172(a) for k, a, _f in items if k in ("white", "pink")])),
                  "noise_flutter_db": float(np.mean([dc.flutter_538(a) for k, a, _f in items if k in ("white", "pink")])),
                  "chirp_alias_db": float(np.mean([dc.chirp_alias_db(a, f) for k, a, f in items if k == "chirp"]))}
    return out


def cmd_real(args):
    index = load_index()
    ents = index["eval"]
    if args.limit:
        ents = sorted(ents, key=lambda e: dc.md5_text(e["id"]))[:args.limit]
    auds = clip_audio("eval", ents)
    crops = det_batch(auds)
    d = eval_dir(args.tag)
    os.makedirs(d, exist_ok=True)
    t0 = time.perf_counter()
    with dc.gpu_guard("real:%s" % args.tag):
        ae = dc.load_ae("cuda")
        decs = {"base": ae.decoder}
        if args.ckpt:
            decs["new"] = dc.load_decoder(dc.fresh_decoder("cuda"), dc.at_root(args.ckpt)).eval()
        ael = dc.load_ae("cuda", dc.SNAP_L) if args.samel else None
        print("models loaded %.1f s; %d clips" % (time.perf_counter() - t0, len(ents)), flush=True)
        per = [defaultdict(list) for _ in ents]
        keep = {}
        bs = 8
        with torch.no_grad():
            for b, i in enumerate(range(0, len(ents), bs)):
                x = torch.from_numpy(crops[i:i + bs]).cuda()
                z = ae.encode(x)
                xm = dc.mid(x)
                ys = {}
                for s in SEEDS:
                    for name, dec in decs.items():
                        ys[(name, s)] = dc.mid(dc.seeded_decode(ae, dec, z, 10000 * s + b))
                    if ael is not None:
                        torch.manual_seed(20000 * s + b)
                        zl = ael.encode(x)
                        ys[("samel", s)] = dc.mid(dc.seeded_decode(ael, ael.decoder, zl, 30000 * s + b))
                for (name, s), y in ys.items():
                    yn = y.double().cpu().numpy()
                    for j in range(y.shape[0]):
                        m = dc.clip_pair_metrics(crops[i + j, 0], yn[j])
                        m["mrstft"] = float(dc.mrstft(y[j:j + 1], xm[j:j + 1]))
                        per[i + j][name].append(m)
                        if s == 0 and ents[i + j].get("half") == "A":
                            keep[(i + j, name)] = yn[j].astype(np.float32)
                if (b + 1) % 10 == 0:
                    print("real %d/%d %.1f s" % (min(i + bs, len(ents)), len(ents), time.perf_counter() - t0), flush=True)
        syn = synthetic_tests(ae, decs)
        del ae, decs, ael
        gc.collect()
        torch.cuda.empty_cache()
    rows = []
    for e, p in zip(ents, per):
        r = {"id": e["id"], "bucket": e["bucket"], "subset": e.get("subset"), "flat": e["flat"], "cluster": e["cluster"],
             "half": e.get("half"), "dataset": e["dataset"]}
        for name, ms in p.items():
            r[name] = mean_metrics(ms)
        if "new" not in r:
            r["new"] = {}
        rows.append(r)
    have_new = bool(args.ckpt)
    res = {}
    for h in ("A", "B"):
        hr = [r for r in rows if r["half"] == h]
        res[h] = {"groups": summarize_rows(hr, have_new), "rows": hr}
        if args.samel:
            sr = [dict(r, new=r.get("samel", {})) for r in hr]
            res[h]["samel_groups"] = summarize_rows(sr, True)
    meta = {"tag": args.tag, "ckpt": args.ckpt, "samel": bool(args.samel), "clips": len(rows), "seeds": list(SEEDS),
            "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.perf_counter() - t0, 1),
            "decoder_meta": dc.decoder_meta(dc.at_root(args.ckpt)) if args.ckpt else None,
            "crop": "det_crop: 14 frames from 0.10 s before the loudest 5 ms, peak -6 dBFS, dual mono",
            "precision": "fp32 decoders, bottleneck eval mode", "nboot": NBOOT}
    dc.dump_json({"meta": meta, "half": "A", **res["A"], "synthetic": syn}, os.path.join(d, "metrics_real.json"))
    dc.dump_json({"meta": meta, "half": "B", **res["B"]}, os.path.join(d, "metrics_real_B.json"))
    listen_real(d, ents, crops, keep, rows)
    g = res["A"]["groups"]
    for name in ("all", "TRANSIENT_clean", "SMOOTH"):
        e = g.get(name, {})
        print("A %-16s clips %3s hfk_win %s hfk_frame %s mrstft %s" % (
            name, e.get("clips"), json.dumps({k: e.get("hfk_win", {}).get(k) for k in ("base_med_abs_log", "new_med_abs_log", "diff_ci")}),
            json.dumps({k: e.get("hfk_frame", {}).get(k) for k in ("base_med_abs_log", "new_med_abs_log")}),
            json.dumps({k: e.get("mrstft", {}).get(k) for k in ("base", "new", "ratio")})), flush=True)
    print("SYNTHETIC %s" % json.dumps(syn), flush=True)
    print("wrote %s (%.1f s)" % (os.path.join(d, "metrics_real.json"), time.perf_counter() - t0), flush=True)
    return 0


def listen_real(d, ents, crops, keep, rows):
    pos = {e["id"]: i for i, e in enumerate(ents)}
    a_rows = [r for r in rows if r["half"] == "A"]
    by = defaultdict(list)
    for r in a_rows:
        by[r["bucket"]].append(r)
    for b, rs in by.items():
        pick = sorted(rs, key=lambda r: dc.md5_text(r["id"]))[:4]
        if rs and rs[0]["new"]:
            gain = [(abs(float(dc.log_ratio(r["base"]["hfk_win"], r["base"]["hfk_win_ref"]))) -
                     abs(float(dc.log_ratio(r["new"]["hfk_win"], r["new"]["hfk_win_ref"]))), r["id"]) for r in rs
                    if np.isfinite(r["base"].get("hfk_win", np.nan)) and np.isfinite(r["new"].get("hfk_win", np.nan))]
            gain.sort()
            ids = {r["id"] for r in pick} | {i for _g, i in gain[-3:]} | {i for _g, i in gain[:3]}
            pick = [r for r in rs if r["id"] in ids]
        for r in pick:
            i = pos[r["id"]]
            base = os.path.join(d, "listen", "real", b)
            write_wav(os.path.join(base, "%s_ref.wav" % r["id"]), crops[i, 0])
            for name in ("base", "new", "samel"):
                if (i, name) in keep:
                    write_wav(os.path.join(base, "%s_%s.wav" % (r["id"], name)), keep[(i, name)])


def finish(mono, key, gen_like):
    from sarge.generators.stable_audio import _natural
    from sarge.loudness import normalize_peak, peak_for
    m = _natural(mono, dc.SR) if gen_like == "natural" else mono
    return normalize_peak(m, peak_for(key.split("|")[2])).astype(np.float32)


def route_clip(mono, intensity):
    from sarge.generators.stable_audio import _extract
    from sarge.loudness import normalize_loudness, target_for
    return normalize_loudness(_extract(np.asarray(mono, np.float32), dc.SR, 0.6), target_rms=target_for(intensity))


def real_bucket_medians(index):
    ents = [e for e in index["eval"] if e.get("half") == "A" and not e["flat"]]
    auds = clip_audio("eval", ents)
    by = defaultdict(lambda: defaultdict(list))
    for e, a in zip(ents, auds):
        c, _s = dc.det_crop(a)
        m = dc.self_metrics(c[0])
        for k in ("hfk_win", "hfk_frame", "spike_rate"):
            by[e["bucket"]][k].append(m[k])
    return {b: {k: float(np.nanmedian(v)) for k, v in d.items()} | {"clips": len(d["hfk_win"])} for b, d in by.items()}


def cmd_gen(args):
    d = eval_dir(args.tag)
    os.makedirs(d, exist_ok=True)
    bank = np.load(os.path.join(dc.GEN, "eval.npz"))
    keys = [str(k) for k in bank["keys"]]
    z_all = bank["z"][..., :dc.SHORT_FRAMES]
    static_base = bank["static"]
    if args.limit:
        keys, z_all, static_base = keys[:args.limit], z_all[:args.limit], static_base[:args.limit]
    index = load_index()
    medians = real_bucket_medians(index)
    t0 = time.perf_counter()
    from sarge.static_guard import is_static
    n1 = dc.SR
    outs = {"base": {}, "new": {}}
    pairs = defaultdict(lambda: defaultdict(list))
    base0 = {}
    route_audio = {"base": {}, "new": {}}
    listen = {}
    acts = {"base": 0.0, "new": 0.0}
    with dc.gpu_guard("gen:%s" % args.tag):
        ae32 = dc.load_ae("cuda")
        ae16 = dc.load_ae("cuda").half()
        decs16 = {"base": ae16.decoder, "new": dc.load_decoder(dc.fresh_decoder("cuda"), dc.at_root(args.ckpt)).half().eval()}
        decs32 = {"base": ae32.decoder, "new": dc.load_decoder(dc.fresh_decoder("cuda"), dc.at_root(args.ckpt)).eval()}
        hooks = []
        for name, dec in decs16.items():
            blk = dec.layers[3]
            for m in list(blk.transformers) + [blk.mapping]:
                hooks.append(m.register_forward_hook(
                    lambda mod, inp, out, name=name: acts.__setitem__(name, max(acts[name], float(out.detach().float().abs().max())))))
        bs = 16
        with torch.no_grad():
            for b, i in enumerate(range(0, len(keys), bs)):
                zb = torch.from_numpy(z_all[i:i + bs]).cuda()
                for s in SEEDS:
                    monos = {}
                    for name in ("base", "new"):
                        y16 = dc.seeded_decode(ae16, decs16[name], zb.half(), 40000 * s + b).float()
                        y = y16.clamp(-1, 1)[..., :n1].double().cpu().numpy()
                        extra = {}
                        if s == 0:
                            y32 = dc.seeded_decode(ae32, decs32[name], zb.float(), 40000 * s + b)
                            z2 = ae32.encode(y16.clamp(-1, 1))[..., :dc.SHORT_FRAMES]
                            err = (z2 - zb.float()).flatten(1).norm(dim=1) / zb.float().flatten(1).norm(dim=1)
                            extra = {j: {"fp16_gap": float(dc.mrstft(dc.mid(y16[j:j + 1]), dc.mid(y32[j:j + 1]))),
                                         "cycle_err": float(err[j])} for j in range(zb.shape[0])}
                        monos[name] = []
                        for j in range(y.shape[0]):
                            k = keys[i + j]
                            mono = y[j].mean(0)
                            fin = finish(mono, k, "natural")
                            met = dc.self_metrics(fin)
                            met["nonfinite"] = int(not np.isfinite(y16[j].cpu().numpy()).all())
                            sd = 0.5 * (y[j][0] - y[j][1])
                            met["side_mid_db"] = float(10 * np.log10((np.mean(sd * sd) + 1e-20) / (np.mean(mono * mono) + 1e-20)))
                            met["static"] = int(bool(is_static(fin, dc.SR)[0]))
                            met.update(extra.get(j, {}))
                            outs[name][(i + j, s)] = met
                            monos[name].append(mono)
                            if s == 0:
                                et, mat, it, v = k.split("|")
                                if it == "normal" and int(v) < 20 and (et, mat) in dc.VOCAB_KEYS:
                                    route_audio[name][(et, mat, int(v))] = route_clip(mono, it)
                                if int(v) < 5 and it == "normal":
                                    listen[(k, name)] = fin
                    for j in range(zb.shape[0]):
                        kk = "|".join(keys[i + j].split("|")[:2])
                        a, bb = monos["base"][j], monos["new"][j]
                        pairs[kk]["mel2k_new_base"].append(dc.mel_dist(a, bb, 2000.0))
                        ea = dc.band_env(torch.from_numpy(a[None]).float())
                        eb = dc.band_env(torch.from_numpy(bb[None]).float())
                        pairs[kk]["band_env_l1"].append(float((ea - eb).abs().mean()))
                        if s == 0:
                            base0[i + j] = a.astype(np.float32)
                        elif s == 1:
                            a0 = base0.pop(i + j).astype(np.float64)
                            pairs[kk]["mel2k_seed_floor"].append(dc.mel_dist(a0, a, 2000.0))
                            e0 = dc.band_env(torch.from_numpy(a0[None]).float())
                            pairs[kk]["band_env_seed_floor"].append(float((e0 - ea).abs().mean()))
                print("gen %d/%d %.1f s" % (min(i + bs, len(keys)), len(keys), time.perf_counter() - t0), flush=True)
        for h in hooks:
            h.remove()
        del ae16, ae32, decs16, decs32
        gc.collect()
        torch.cuda.empty_cache()
    per_key = defaultdict(lambda: {"base": defaultdict(list), "new": defaultdict(list)})
    for name in ("base", "new"):
        for (i, s), m in outs[name].items():
            kk = "|".join(keys[i].split("|")[:2])
            for f, v in m.items():
                if isinstance(v, (int, float)):
                    per_key[kk][name][f].append(v)
    keys_out = {}
    for kk, v in per_key.items():
        row = {name: {f: float(np.nanmean(x)) for f, x in v[name].items()} for name in ("base", "new")}
        row["pair"] = {f: float(np.nanmean(x)) for f, x in pairs[kk].items()}
        eb = dc.TRANSIENT_KEYS.get(kk)
        if eb and eb in medians:
            rm = medians[eb]
            row["real"] = rm
            row["toward_real"] = {f: abs(np.log(row["base"][f] / rm[f])) - abs(np.log(row["new"][f] / rm[f]))
                                  for f in ("hfk_win", "hfk_frame") if row["base"].get(f, 0) > 0 and row["new"].get(f, 0) > 0}
        keys_out[kk] = row
    tot = {name: {"nonfinite": int(sum(m.get("nonfinite", 0) for m in outs[name].values())),
                  "static": int(sum(m.get("static", 0) for (i, s), m in outs[name].items() if s == 0)),
                  "max_fp16_activation": acts[name],
                  "fp16_gap": float(np.nanmean([m["fp16_gap"] for m in outs[name].values() if "fp16_gap" in m])),
                  "cycle_err": float(np.nanmean([m["cycle_err"] for m in outs[name].values() if "cycle_err" in m])),
                  "hf_energy_db": float(np.nanmean([m["hf_energy_db"] for m in outs[name].values()]))}
           for name in ("base", "new")}
    tot["record_static"] = int(np.sum(static_base))
    for name in ("base", "new"):
        rd = os.path.join(d, "route_%s" % name)
        os.makedirs(rd, exist_ok=True)
        for (et, mat, v), a in route_audio[name].items():
            write_wav(os.path.join(rd, "%s-%s_%02d.wav" % (et, mat, v)), a)
        man = dc.load_json(dc.MANIFEST)
        tv = [r for r in man["records"] if r["set"] in ("train", "val")]
        info = {"clusters": sorted({r["cluster"] for r in tv}) if name == "new" else [],
                "sources": sorted({r["path"] for r in tv}) if name == "new" else [],
                "steps": 3, "note": "SAME-S decoder %s on the recorded DiT eval latents (13-frame prefix, eager fp16, seed 0), "
                                    "0.6 s _extract window, rms target_for(normal)" % ("fine-tuned %s" % args.ckpt if name == "new" else "base")}
        bad = [c for c in info["clusters"] if not dc.CLUSTER_RE.match(c)]
        if bad:
            raise SystemExit("route clusters not in manifest form: %s" % bad[:5])
        dc.dump_json(info, os.path.join(rd, "route.json"))
    for (k, name), a in listen.items():
        et, mat, it, v = k.split("|")
        write_wav(os.path.join(d, "listen", "gen", "%s_%s" % (et, mat), "v%02d_%s.wav" % (int(v), name)), a)
    rep = {"meta": {"tag": args.tag, "ckpt": args.ckpt, "latents": len(keys), "seeds": list(SEEDS),
                    "built": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.perf_counter() - t0, 1),
                    "decode": "13-frame prefix, eager fp16, bottleneck eval mode, clamp, first 1.0 s, _natural + peak per intensity"},
           "totals": tot, "keys": keys_out, "real_medians_halfA": medians}
    dc.dump_json(rep, os.path.join(d, "metrics_gen.json"))
    print("GEN totals %s" % json.dumps(tot), flush=True)
    for kk in dc.TRANSIENT_KEYS:
        if kk in keys_out:
            print("  %-16s %s" % (kk, json.dumps(keys_out[kk].get("toward_real"))), flush=True)
    return 0


def cmd_latency(args):
    from sarge.generators.stable_audio3 import StableAudio3Generator
    from sarge.types import CacheKey
    d = eval_dir(args.tag)
    os.makedirs(d, exist_ok=True)
    with dc.gpu_guard("latency:%s" % args.tag):
        kw = gen_kwargs("default", args.ckpt)
        kw["static_guard"] = False
        gen = StableAudio3Generator(**kw)
        t0 = time.perf_counter()
        gen.ensure_ready()
        ready = time.perf_counter() - t0
        keys = [CacheKey("footstep", "gravel", "normal"), CacheKey("shoot", "rifle", "normal"),
                CacheKey("break", "window", "normal"), CacheKey("click", "button", "normal")]
        for k in keys:
            for v in range(3):
                gen.generate(k, variant=100 + v)
        fast = gen._fast
        events = []
        target = fast.decode_graph if fast is not None and fast.decode_graph is not None else None
        if target is not None:
            def timed(z, chunked=None, **kw2):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                out = target(z, chunked=chunked, **kw2)
                e.record()
                events.append((s, e))
                return out
            if fast.prefix is not None:
                fast.prefix.orig = timed
            else:
                gen._model.model.pretransform.decode = timed
        e2e = []
        for i in range(args.calls):
            k = keys[i % len(keys)]
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            gen.generate(k, variant=i)
            torch.cuda.synchronize()
            e2e.append((time.perf_counter() - t1) * 1000.0)
        torch.cuda.synchronize()
        dms = [s.elapsed_time(e) for s, e in events]
        rep = {"tag": args.tag, "ckpt": args.ckpt, "calls": args.calls, "ready_s": round(ready, 1),
               "decode_ms_median": float(np.median(dms)) if dms else None, "decode_ms": dms,
               "e2e_ms_median": float(np.median(e2e)), "e2e_ms": e2e, "fast_stats": gen.fast_stats(),
               "decoder": os.path.basename(kw.get("decoder_ckpt", "")) or "base"}
    dc.dump_json(rep, os.path.join(d, "latency.json"))
    print("LATENCY %s decode median %s ms e2e median %.2f ms stats %s" % (
        args.tag, rep["decode_ms_median"], rep["e2e_ms_median"], json.dumps(rep["fast_stats"])), flush=True)
    return 0


def snapshots_identical():
    from safetensors import safe_open
    a = os.path.join(dc.SNAP_S, "model.safetensors")
    b = os.path.join(dc.SNAP_S_ENGINE, "model.safetensors")
    if not os.path.isfile(b):
        return {"identical": None, "reason": "engine snapshot missing: %s" % b}
    if os.path.getsize(a) != os.path.getsize(b):
        return {"identical": False, "reason": "sizes differ"}
    n = 0
    with safe_open(a, framework="pt") as fa, safe_open(b, framework="pt") as fb:
        ka = sorted(k for k in fa.keys() if k.startswith("pretransform.model."))
        kb = sorted(k for k in fb.keys() if k.startswith("pretransform.model."))
        if ka != kb:
            return {"identical": False, "reason": "pretransform keys differ"}
        for k in ka:
            if not torch.equal(fa.get_tensor(k), fb.get_tensor(k)):
                return {"identical": False, "reason": "tensor %s differs" % k}
            n += 1
    ca = dc.load_json(os.path.join(dc.SNAP_S, "model_config.json"))
    cb = dc.load_json(os.path.join(dc.SNAP_S_ENGINE, "model_config.json"))
    return {"identical": ca == cb, "pretransform_tensors_equal": n, "config_equal": ca == cb}


def cmd_hook_check(args):
    from sarge.generators.stable_audio3 import StableAudio3Generator
    from safetensors.torch import load_file
    from sarge.types import CacheKey
    run = os.path.join(dc.RUNS, args.run)
    base = os.path.join(run, "decoder_base.safetensors")
    ft = os.path.join(run, args.ckpt_name)
    for p in (base, ft):
        if not os.path.isfile(p):
            raise SystemExit("hook-check: missing %s" % p)
    snap = snapshots_identical()
    print("SNAPSHOTS %s" % json.dumps(snap), flush=True)
    keys = [CacheKey("footstep", "gravel", "normal"), CacheKey("shoot", "rifle", "normal")]
    kw = dict(fast=True, graph_decode=True, decode_prefix=True, fast_load=True, static_guard=False,
              sampler_type="euler", steps=3, skip_first_step=True, trim="natural", level="peak")
    rep = {"snapshots": snap, "kwargs": kw}
    with dc.gpu_guard("hook-check:%s" % args.run):
        outs = {}
        for tag, ck in (("none", None), ("base", base), ("ft", ft)):
            g = StableAudio3Generator(**kw, decoder_ckpt=ck)
            t0 = time.perf_counter()
            g.ensure_ready()
            outs[tag] = [g.generate(k, variant=3).samples.copy() for k in keys]
            st = g.fast_stats()
            rep[tag] = {"ready_s": round(time.perf_counter() - t0, 1), "fast_stats": st,
                        "finite": bool(all(np.isfinite(o).all() for o in outs[tag])),
                        "decoder_ckpt": g.decoder_ckpt}
            if tag == "ft":
                sd = load_file(ft)
                live = g._model.same.model.decoder.state_dict()
                k0 = "layers.3.mapping.weight_v"
                rep["ft"]["weight_equal_fp16"] = bool(torch.equal(live[k0].cpu(), sd[k0].half()))
                rep["ft"]["weight_dtype"] = str(live[k0].dtype)
                rep["ft"]["differs_from_base"] = bool(any(not np.array_equal(a, b) for a, b in zip(outs["ft"], outs["none"])))
            del g
            gc.collect()
            torch.cuda.empty_cache()
    rep["base_bit_identical"] = bool(all(np.array_equal(a, b) for a, b in zip(outs["none"], outs["base"])))
    ok = (rep["base_bit_identical"] and rep["ft"]["finite"] and rep["ft"]["fast_stats"].get("decode_graphs", 0) >= 1
          and rep["ft"]["weight_equal_fp16"] and rep["ft"]["fast_stats"].get("decode_failures", 1) == 0
          and snap.get("identical") is not False)
    rep["pass"] = bool(ok)
    dc.dump_json(rep, os.path.join(run, "hook_check.json"))
    print("HOOK_CHECK pass=%s base_bit_identical=%s ft finite=%s decode_graphs=%s decode_failures=%s weight_equal_fp16=%s "
          "differs_from_base=%s" % (ok, rep["base_bit_identical"], rep["ft"]["finite"],
                                    rep["ft"]["fast_stats"].get("decode_graphs"), rep["ft"]["fast_stats"].get("decode_failures"),
                                    rep["ft"]["weight_equal_fp16"], rep["ft"].get("differs_from_base")), flush=True)
    return 0 if ok else 1


def _get(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def judge_real(g, label):
    out = {}
    tc = g.get("TRANSIENT_clean", {})
    hw, hf, sr = tc.get("hfk_win", {}), tc.get("hfk_frame", {}), tc.get("spike_rate", {})
    p1 = []
    if "new_med_abs_log" not in hw:
        out["P1"] = {"pass": None, "why": "no new decoder in %s" % label}
    else:
        p1.append(("hfk_win reduction >= 25%", hw["reduction"] >= 0.25))
        p1.append(("hfk_win diff CI excludes 0", hw["diff_ci"][1] is not None and hw["diff_ci"][1] < 0))
        p1.append(("hfk_frame reduction >= 15%", hf.get("reduction", -1) >= 0.15))
        p1.append(("hfk_frame diff CI excludes 0", _get(hf, "diff_ci", 1) is not None and hf["diff_ci"][1] < 0))
        p1.append(("spike_rate closer", sr.get("new_med_abs_log", 9) < sr.get("base_med_abs_log", 0)))
        for b in sorted(k for k in g if k.startswith("bucket:") and "|" not in k):
            bb = b[7:]
            if dc.mix_of(bb) not in dc.TRANSIENT or bb == "step_metal":
                continue
            for f in ("hfk_win", "hfk_frame"):
                e = g[b].get(f, {})
                if "new_med_log" in e:
                    p1.append(("%s %s toward 0" % (bb, f), abs(e["new_med_log"]) < abs(e["base_med_log"])))
                    p1.append(("%s %s no overshoot > +0.2" % (bb, f), e["new_med_log"] <= 0.2))
        ec, fl = tc.get("hf_env_corr", {}), tc.get("hf_floor_db", {})
        p1.append(("guard hf_env_corr >= base - 0.02", ec.get("diff", -1) >= -0.02))
        p1.append(("guard hf_floor_db within 2 dB", abs(fl.get("diff", 9)) <= 2.0))
        out["P1"] = {"pass": all(v for _k, v in p1), "checks": p1}
    al = g.get("all", {})
    p2 = []
    for m in ("mrstft", "mel_dist"):
        if "ratio" in al.get(m, {}):
            p2.append(("%s <= 1.02 x base" % m, al[m]["ratio"] <= 1.02))
    if "diff" in al.get("si_sdr", {}):
        p2.append(("si_sdr >= base - 0.3", al["si_sdr"]["diff"] >= -0.3))
    for m in ("lsd_4-8", "lsd_8-16"):
        if "ratio" in al.get(m, {}):
            p2.append(("%s <= 1.03 x base" % m, al[m]["ratio"] <= 1.03))
    for m in ("rise_err_ms", "crest_err_db"):
        if "ratio_ci" in al.get(m, {}):
            p2.append(("%s upper CI <= 1.05" % m, al[m]["ratio_ci"][1] is not None and al[m]["ratio_ci"][1] <= 1.05))
    out["P2"] = {"pass": all(v for _k, v in p2) if p2 else None, "checks": p2}
    p3 = []
    for b, e in g.items():
        if not b.startswith("bucket:") or "|" in b or b == "bucket:step_metal":
            continue
        if "ratio" in e.get("mrstft", {}):
            p3.append(("%s mrstft <= 1.05" % b, e["mrstft"]["ratio"] <= 1.05))
        hw2 = e.get("hfk_win", {})
        if "new_med_log" in hw2:
            p3.append(("%s |hfk_win log| <= |base| + 0.1" % b, abs(hw2["new_med_log"]) <= abs(hw2["base_med_log"]) + 0.1))
            if dc.mix_of(b[7:]) in dc.SMOOTH:
                for f in ("hfk_win", "hfk_frame"):
                    e2 = e.get(f, {})
                    if "new_med_log" in e2:
                        p3.append(("%s %s within 0.15 of base" % (b, f), abs(e2["new_med_log"] - e2["base_med_log"]) <= 0.15))
    for s in ("flat", "clean", "hear", "gun_clean"):
        if "ratio" in g.get(s, {}).get("mrstft", {}):
            p3.append(("%s mrstft <= 1.05" % s, g[s]["mrstft"]["ratio"] <= 1.05))
    out["P3"] = {"pass": all(v for _k, v in p3) if p3 else None, "checks": p3}
    return out


def judge_p4(real):
    syn = real.get("synthetic", {})
    b, n = syn.get("base", {}), syn.get("new", {})
    if not n:
        return {"pass": None, "why": "no new decoder"}
    al = real["groups"].get("all", {})
    c = [("silence hiss > 8 kHz <= base + 1 dB", n["silence_hiss8k_db"] <= b["silence_hiss8k_db"] + 1.0),
         ("noise comb <= base + 3 dB", n["noise_comb_db"] <= b["noise_comb_db"] + 3.0),
         ("chirp alias <= base + 1 dB", n["chirp_alias_db"] <= b["chirp_alias_db"] + 1.0),
         ("flutter <= base + 3 dB", n["noise_flutter_db"] <= b["noise_flutter_db"] + 3.0)]
    if np.isfinite(n.get("silence_comb_db", np.nan)) and np.isfinite(b.get("silence_comb_db", np.nan)):
        c.append(("silence comb <= base + 3 dB", n["silence_comb_db"] <= b["silence_comb_db"] + 3.0))
    if "diff" in al.get("gap_hiss_db", {}):
        c.append(("clip gap hiss > 8 kHz <= base + 1 dB", al["gap_hiss_db"]["diff"] <= 1.0))
    if "diff" in al.get("pre_dec_db", {}):
        c.append(("pre-echo <= base + 1 dB", al["pre_dec_db"]["diff"] <= 1.0))
    return {"pass": all(v for _k, v in c), "checks": c}


def judge_p5(gen):
    if not gen:
        return {"pass": None, "why": "metrics_gen.json missing"}
    keys = gen["keys"]
    c = []
    moves = [keys[k].get("toward_real", {}) for k in dc.TRANSIENT_KEYS if k in keys]
    closer = sum(1 for m in moves if m and all(v > 0 for v in m.values()))
    away = [v for m in moves for v in m.values() if v < -0.1]
    c.append(("closer to real on >= 4 of 6 transient keys (%d)" % closer, closer >= 4))
    c.append(("no key moves away by > 0.1", not away))
    for k, r in keys.items():
        p = r["pair"]
        if p.get("mel2k_seed_floor"):
            c.append(("%s mel2k new-base <= 1.5 x seed floor" % k, p["mel2k_new_base"] <= 1.5 * p["mel2k_seed_floor"]))
            c.append(("%s band_env <= 1.5 x seed floor" % k, p["band_env_l1"] <= 1.5 * p["band_env_seed_floor"]))
        c.append(("%s hf energy within 1 dB" % k, abs(r["new"]["hf_energy_db"] - r["base"]["hf_energy_db"]) <= 1.0))
    t = gen["totals"]
    c.append(("cycle error <= 1.05 x base", t["new"]["cycle_err"] <= 1.05 * t["base"]["cycle_err"]))
    c.append(("no NaN/inf", t["new"]["nonfinite"] == 0))
    c.append(("max fp16 activation < 30000", t["new"]["max_fp16_activation"] < 30000))
    c.append(("static <= base + 1", t["new"]["static"] <= t["base"]["static"] + 1))
    c.append(("fp16 gap <= 1.5 x base", t["new"]["fp16_gap"] <= 1.5 * t["base"]["fp16_gap"]))
    return {"pass": all(v for _k, v in c), "checks": c}


def judge_p6(vocab):
    if not vocab:
        return {"pass": None, "why": "vocab.json missing"}
    summ = vocab["summary"]
    c, cb, cn = [], [], []
    mat_b, mat_n = [], []
    for k, row in summ.items():
        b = row["settings"].get("dec_base", {})
        n = row["settings"].get("dec_new", {})
        if b.get("cons") is None or n.get("cons") is None:
            continue
        cb.append(b["cons"])
        cn.append(n["cons"])
        drop = b["cons"] - n["cons"]
        sig = n.get("cons_ci") and b.get("cons_ci") and n["cons_ci"][1] < b["cons_ci"][0]
        c.append(("%s no significant drop > 0.05" % k, not (drop > 0.05 and sig)))
        if b.get("material") is not None and n.get("material") is not None:
            mat_b.append(b["material"])
            mat_n.append(n["material"])
    if cb:
        c.append(("mean consistency >= base - 0.01", np.mean(cn) >= np.mean(cb) - 0.01))
    if mat_b:
        c.append(("footstep material new - base >= -0.03 (means; vocab_eval gives no paired CI)",
                  np.mean(mat_n) - np.mean(mat_b) >= -0.03))
    return {"pass": all(v for _k, v in c) if c else None, "checks": c}


def judge_p7(lat, hook):
    c = []
    if lat.get("base") and lat.get("new"):
        db, dn = np.median(lat["base"]["decode"]), np.median(lat["new"]["decode"])
        eb, en = np.median(lat["base"]["e2e"]), np.median(lat["new"]["e2e"])
        c.append(("graph decode within 5%% (%.3f vs %.3f ms)" % (dn, db), abs(dn / db - 1) <= 0.05))
        c.append(("end to end within 3%% (%.2f vs %.2f ms)" % (en, eb), abs(en / eb - 1) <= 0.03))
        c.append(("decode_failures == 0", all(f == 0 for f in lat["new"]["failures"])))
    else:
        return {"pass": None, "why": "latency runs missing"}
    if hook:
        c.append(("hook-check pass", bool(hook.get("pass"))))
    return {"pass": all(v for _k, v in c), "checks": c}


def judge_val(vrow, base):
    if not vrow:
        return {"pass": None, "why": "no val.jsonl"}
    c = [("val selection score improves (%.4f vs base %.4f)" % (vrow["score"], base["score"]), vrow["score"] < base["score"]),
         ("val hfk_win reduction >= 25%", vrow["hfk_win_med_abs_log"] <= 0.75 * base["hfk_win_med_abs_log"]),
         ("val hfk_frame reduction >= 15%", vrow["hfk_frame_med_abs_log"] <= 0.85 * base["hfk_frame_med_abs_log"]),
         ("val eligible", bool(vrow["eligible"]))]
    return {"pass": all(v for _k, v in c), "checks": c}


def cmd_verdict(args):
    d = eval_dir(args.tag)
    rd = lambda p: dc.load_json(p) if os.path.isfile(p) else None
    real = rd(os.path.join(d, "metrics_real.json"))
    gen = rd(os.path.join(d, "metrics_gen.json"))
    vocab = rd(os.path.join(d, "vocab.json"))
    lat = {"base": defaultdict(list), "new": defaultdict(list)}
    for t in args.latency:
        p = os.path.join(eval_dir(t), "latency.json")
        if os.path.isfile(p):
            j = dc.load_json(p)
            k = "new" if j.get("ckpt") else "base"
            lat[k]["decode"].append(j["decode_ms_median"])
            lat[k]["e2e"].append(j["e2e_ms_median"])
            lat[k]["failures"].append(j["fast_stats"].get("decode_failures", 0))
    run = args.run or args.tag
    vrows = []
    vp = os.path.join(dc.RUNS, run, "val.jsonl")
    if os.path.isfile(vp):
        vrows = [json.loads(l) for l in open(vp, encoding="utf-8") if l.strip()]
    base_v = next((v for v in vrows if v["tag"] == "base"), None)
    best_v = min((v for v in vrows if v["tag"] != "base" and v.get("eligible")), key=lambda v: v["score"], default=None)
    hook = rd(os.path.join(dc.RUNS, run, "hook_check.json"))
    out = {"tag": args.tag, "final": bool(args.final), "val": judge_val(best_v, base_v) if base_v else {"pass": None}}
    if real:
        out["A"] = judge_real(real["groups"], "half A")
        out["P4"] = judge_p4(real)
    out["P5"] = judge_p5(gen)
    out["P6"] = judge_p6(vocab)
    out["P7"] = judge_p7(lat, hook)
    if args.final:
        rb = rd(os.path.join(d, "metrics_real_B.json"))
        out["B"] = judge_real(rb["groups"], "half B") if rb else {"pass": None, "why": "metrics_real_B.json missing"}
    parts = [out["val"].get("pass")] + [v.get("pass") for k in ("A", "B") if k in out for v in out[k].values()] + \
            [out[k].get("pass") for k in ("P4", "P5", "P6", "P7") if k in out]
    out["pass"] = bool(parts and all(p is True for p in parts))
    out["missing"] = sum(p is None for p in parts)
    dc.dump_json(out, os.path.join(d, "verdict.json"))
    print("VERDICT %s pass=%s missing=%d" % (args.tag, out["pass"], out["missing"]), flush=True)
    for k in ("val", "A", "B", "P4", "P5", "P6", "P7"):
        if k in out:
            v = out[k]
            if k in ("A", "B"):
                print("  %s %s" % (k, {p: e.get("pass") for p, e in v.items()}), flush=True)
            else:
                print("  %s %s" % (k, v.get("pass")), flush=True)
    return 0


def parse(argv):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--encode", type=int, default=512)
    p = sub.add_parser("teacher-probe")
    p.add_argument("--limit", type=int, default=None)
    c = sub.add_parser("teacher-cache")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--force", action="store_true")
    re_ = sub.add_parser("real")
    re_.add_argument("--tag", required=True)
    re_.add_argument("--ckpt", default=None)
    re_.add_argument("--samel", action="store_true")
    re_.add_argument("--limit", type=int, default=None)
    g = sub.add_parser("gen")
    g.add_argument("--tag", required=True)
    g.add_argument("--ckpt", required=True)
    g.add_argument("--limit", type=int, default=None)
    l = sub.add_parser("latency")
    l.add_argument("--tag", required=True)
    l.add_argument("--ckpt", default=None)
    l.add_argument("--calls", type=int, default=60)
    h = sub.add_parser("hook-check")
    h.add_argument("--run", required=True)
    h.add_argument("--ckpt-name", default="decoder_ema_last.safetensors")
    v = sub.add_parser("verdict")
    v.add_argument("--tag", required=True)
    v.add_argument("--run", default=None)
    v.add_argument("--final", action="store_true")
    v.add_argument("--latency", nargs="*", default=[])
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    fn = {"record": cmd_record, "teacher-probe": cmd_teacher_probe, "teacher-cache": cmd_teacher_cache,
          "real": cmd_real, "gen": cmd_gen, "latency": cmd_latency, "hook-check": cmd_hook_check,
          "verdict": cmd_verdict}[args.cmd]
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
