import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
BENCH = os.path.join(SCRIPTS, "bench")
for _p in (BENCH, SCRIPTS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import deque

import numpy as np
import torch

import dec_common as dc
import dec_disc as dd

SMOKE = {"batch": 4, "anchor_batch": 2, "steps": 40, "d_warmup": 10, "val_every": 20, "val_clips": 24,
         "state_every": 10, "export_every": 20, "d_acc_crops": 16, "watch_crops": 8, "anchor_val": 16}
FULL = {"batch": 16, "anchor_batch": 6, "steps": 12000, "d_warmup": 2000, "val_every": 1000, "val_clips": 0,
        "state_every": 500, "export_every": 1000, "d_acc_crops": 256, "watch_crops": 64, "anchor_val": 0}
LAST3 = ("layers.3.transformers.3.", "layers.3.transformers.4.", "layers.3.transformers.5.", "layers.3.mapping.",
         "layers.3.new_tokens")


def parse(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-at", type=int, default=None)
    for k in FULL:
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=None)
    ap.add_argument("--d-warmup-extra", type=int, default=2000)
    ap.add_argument("--d-warmup-acc", type=float, default=0.75)
    ap.add_argument("--chirps", type=int, default=1)
    ap.add_argument("--short-prob", type=float, default=0.15)
    ap.add_argument("--d-micro", type=int, default=4)
    ap.add_argument("--g-micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-min", type=float, default=3e-6)
    ap.add_argument("--lr-warmup", type=int, default=500)
    ap.add_argument("--d-lr", type=float, default=2e-4)
    ap.add_argument("--d-prob", type=float, default=1.0)
    ap.add_argument("--gp", type=float, default=0.0)
    ap.add_argument("--w-mid", type=float, default=1.0)
    ap.add_argument("--w-side", type=float, default=0.1)
    ap.add_argument("--w-hf", type=float, default=0.0)
    ap.add_argument("--w-adv", type=float, default=0.1)
    ap.add_argument("--w-fm", type=float, default=5.0)
    ap.add_argument("--w-anchor", type=float, default=0.25)
    ap.add_argument("--w-anchor-hf", type=float, default=0.05)
    ap.add_argument("--w-teacher", type=float, default=0.25)
    ap.add_argument("--w-gen-adv", type=float, default=0.0)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--bn-noise", choices=["train", "eval"], default="train")
    ap.add_argument("--scope", choices=["full", "last3"], default="full")
    ap.add_argument("--disc", default="mpd,mbd")
    ap.add_argument("--mbd-windows", default="2048,1024,512,256,128")
    ap.add_argument("--gain-db", default=None)
    ap.add_argument("--onset-range", default=None)
    ap.add_argument("--pseudo-stereo", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--mix-prob", type=float, default=0.15)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--watch-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min-stop-step", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vram-budget-gb", type=float, default=9.0)
    args = ap.parse_args(argv)
    base = SMOKE if args.smoke else FULL
    for k, v in base.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.smoke:
        args.d_warmup_extra = min(args.d_warmup_extra, 10)
        args.watch_every = min(args.watch_every, 10)
        args.lr_warmup = min(args.lr_warmup, 10)
        args.min_stop_step = 0
    return args


def pair(s, default):
    if s is None:
        return default
    a, b = (float(v) for v in str(s).split(","))
    return (a, b)


class Data:
    def __init__(self, index, split):
        self.ents = index[split]
        self.mm = np.memmap(os.path.join(dc.CACHE, "%s.f32" % split), dtype=np.float32, mode="r")

    def clip(self, i):
        e = self.ents[i]
        return self.mm[e["off"]:e["off"] + e["n"]]


class Sampler:
    def __init__(self, data, seed, gain, onset, mix_prob, stereo):
        self.d = data
        self.tabs = dc.bucket_tables(data.ents)
        self.mixes = [m for m in self.tabs if dc.SHARES.get(m, 0) > 0]
        p = np.array([dc.SHARES[m] for m in self.mixes], np.float64)
        self.mix_p = p / p.sum()
        self.rng = np.random.default_rng(seed)
        self.gain = gain
        self.onset = onset
        self.mix_prob = mix_prob
        self.stereo = stereo
        mx = [i for i, e in enumerate(data.ents) if e["mix"] in ("texture", "human_steady")]
        w = np.array([max(data.ents[i]["weight"], 1e-6) for i in mx])
        self.mixers = (np.array(mx), w / w.sum()) if mx else None

    def state(self):
        return self.rng.bit_generator.state

    def set_state(self, st):
        self.rng.bit_generator.state = st

    def hash(self):
        return hashlib.md5(json.dumps(self.state(), sort_keys=True, default=str).encode()).hexdigest()[:12]

    def pick(self, m):
        t = self.tabs[m]
        return int(t["idx"][self.rng.choice(len(t["idx"]), p=t["p"])])

    def crop(self, i):
        e = self.d.ents[i]
        x = self.d.clip(i)
        n = x.shape[0]
        r = self.rng
        if r.uniform() < 0.7 and e["onsets"]:
            on = int(e["onsets"][r.integers(len(e["onsets"]))])
            s = max(0, on - int(r.uniform(*self.onset) * dc.SR))
        else:
            s = int(r.integers(0, max(1, n - dc.SEG + 1)))
        seg = np.array(x[s:s + dc.SEG], dtype=np.float32)
        mask = np.ones(dc.SEG, np.float32)
        if seg.shape[0] < dc.SEG:
            fo = min(int(0.01 * dc.SR), seg.shape[0] // 2)
            if fo > 0:
                seg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
            seg = np.pad(seg, (0, dc.SEG - seg.shape[0]))
        for a, b in e["clip_runs"]:
            a2, b2 = max(0, a - s), min(dc.SEG, b - s)
            if b2 > a2:
                mask[a2:b2] = 0.0
        return seg, mask

    def draw(self, i):
        for _ in range(8):
            seg, mask = self.crop(i)
            pk = float(np.max(np.abs(seg)))
            if pk >= 10 ** (-70 / 20.0) or self.rng.uniform() < 0.03:
                return seg, mask, pk
        return seg, mask, pk

    def batch(self, b, chirps):
        r = self.rng
        n_real = max(0, b - chirps)
        mixes = r.choice(len(self.mixes), size=n_real, p=self.mix_p)
        xs, ms, ws, ids = [], [], [], []
        for mi in mixes:
            i = self.pick(self.mixes[mi])
            e = self.d.ents[i]
            seg, mask, pk = self.draw(i)
            silent = pk < 10 ** (-70 / 20.0)
            if not silent:
                seg = seg * np.float32(10 ** (r.uniform(*self.gain) / 20.0) / pk)
            if r.uniform() < 0.5:
                seg = -seg
            if not silent and self.mixers is not None and r.uniform() < self.mix_prob:
                j = int(self.mixers[0][r.choice(len(self.mixers[0]), p=self.mixers[1])])
                s2, _m2, p2 = self.draw(j)
                if p2 >= 10 ** (-70 / 20.0):
                    seg = seg + s2 * np.float32(np.max(np.abs(seg)) * 10 ** (-r.uniform(18.0, 30.0) / 20.0) / p2)
            left, right = seg, seg
            if self.stereo and r.uniform() < 0.15:
                dly = int(r.integers(0, int(0.0003 * dc.SR) + 1))
                right = np.pad(seg, (dly, 0))[:dc.SEG] * np.float32(10 ** (r.uniform(-1.0, 1.0) / 20.0))
            peak = float(np.max(np.abs(seg)))
            adv = bool(e["adv"]) and not silent and peak >= 10 ** (-50 / 20.0)
            xs.append(np.stack([left, right]))
            ms.append(mask)
            ws.append(1.0 if adv else 0.0)
            ids.append(e["id"])
        for _ in range(chirps):
            ch, _f = dc.make_chirp(r, dc.SEG)
            xs.append(np.stack([ch, ch]))
            ms.append(np.ones(dc.SEG, np.float32))
            ws.append(0.0)
            ids.append("chirp")
        return np.stack(xs), np.stack(ms), np.array(ws, np.float32), ids

    def real_crop(self, m):
        m = m if m in self.tabs else self.mixes[int(self.rng.choice(len(self.mixes), p=self.mix_p))]
        seg, mask, pk = self.draw(self.pick(m))
        if pk >= 10 ** (-70 / 20.0):
            seg = seg * np.float32(10 ** (self.rng.uniform(*self.gain) / 20.0) / pk)
        return seg


class ValSet:
    def __init__(self, data, n):
        ents = data.ents
        order = sorted(range(len(ents)), key=lambda i: dc.md5_text(ents[i]["id"]))
        if n:
            order = order[:n]
        self.ents = [ents[i] for i in order]
        crops, refs = [], []
        for i in order:
            c, _s = dc.det_crop(np.asarray(data.clip(i)))
            crops.append(c)
        self.x = torch.from_numpy(np.stack(crops))
        self.ref = []
        for c, e in zip(crops, self.ents):
            m = c[0].astype(np.float64)
            h = dc.hp4k(m)
            win = dc.window(m, dc.SR, 0.6, 0.03)
            idx = dc.active_frames(h)
            self.ref.append({"m": m, "h": h, "win": win, "idx": idx, "kw": dc.kurt(h[win[0]:win[1]]),
                             "kf": dc.hfk_frame(h, idx), "tc": (e["mix"] in dc.TRANSIENT) and not e["flat"]})
        self.z = None

    def encode(self, ae, dev, bs=16):
        zs = []
        with torch.no_grad():
            for i in range(0, len(self.x), bs):
                zs.append(ae.encode(self.x[i:i + bs].to(dev)).float())
        self.z = torch.cat(zs)


def val_decode(ae, dec, z, seed, bs=16):
    out = []
    with torch.no_grad():
        for k, i in enumerate(range(0, len(z), bs)):
            out.append(dc.seeded_decode(ae, dec, z[i:i + bs], seed * 1000 + k))
    return torch.cat(out)


def run_val(ae, dec, ref_dec, vs, anchor_val, seeds=(0, 1)):
    mode = ae.bottleneck.training
    rows = [dict() for _ in vs.ents]
    mrs = []
    try:
        ae.bottleneck.train(False)
        for s in seeds:
            parts = []
            with torch.no_grad():
                for k, i in enumerate(range(0, len(vs.z), 16)):
                    ym = dc.mid(dc.seeded_decode(ae, dec, vs.z[i:i + 16], s * 1000 + k))
                    xm = dc.mid(vs.x[i:i + 16].to(ym.device))
                    mrs.append(float(dc.mrstft(ym, xm)))
                    parts.append(ym.double().cpu().numpy())
            yn = np.concatenate(parts)
            for j, (r, row) in enumerate(zip(vs.ref, rows)):
                d = yn[j]
                hd = dc.hp4k(d)
                lw = abs(float(dc.log_ratio(dc.kurt(hd[r["win"][0]:r["win"][1]]), r["kw"])))
                lf = abs(float(dc.log_ratio(dc.hfk_frame(hd, r["idx"]), r["kf"])))
                for k, v in (("lw", lw), ("lf", lf), ("si_sdr", dc.si_sdr(r["m"], d)),
                             ("floor", dc.hf_floor_db(hd, r["idx"])), ("env", dc.hf_env_corr(r["h"], hd))):
                    row.setdefault(k, []).append(v)
        drift = floor = float("nan")
        if anchor_val is not None and len(anchor_val):
            za = anchor_val
            b0 = dc.mid(val_decode(ae, ref_dec, za, 7)).double().cpu().numpy()
            b1 = dc.mid(val_decode(ae, ref_dec, za, 8)).double().cpu().numpy()
            n0 = dc.mid(val_decode(ae, dec, za, 7)).double().cpu().numpy()
            floor = float(np.mean([dc.mel_dist(a[:dc.SHORT_SEG], b[:dc.SHORT_SEG], 2000.0) for a, b in zip(b0, b1)]))
            drift = float(np.mean([dc.mel_dist(a[:dc.SHORT_SEG], b[:dc.SHORT_SEG], 2000.0) for a, b in zip(b0, n0)]))
    finally:
        ae.bottleneck.train(mode)
    agg = {k: np.array([np.nanmean(r[k]) for r in rows]) for k in rows[0]}
    tc = np.array([r["tc"] for r in vs.ref])
    med_w = float(np.nanmedian(agg["lw"][tc])) if tc.any() else float("nan")
    med_f = float(np.nanmedian(agg["lf"][tc])) if tc.any() else float("nan")
    return {"score": float(np.nanmean([med_w, med_f])), "hfk_win_med_abs_log": med_w, "hfk_frame_med_abs_log": med_f,
            "mrstft": float(np.mean(mrs)), "si_sdr": float(np.nanmean(agg["si_sdr"])),
            "hf_floor_db": float(np.nanmean(agg["floor"])), "hf_env_corr": float(np.nanmean(agg["env"])),
            "anchor_drift_mel2k": drift, "anchor_seed_floor_mel2k": floor, "clips": len(rows),
            "transient_clean": int(tc.sum())}


def eligible(v, base):
    why = []
    if v["mrstft"] > 1.02 * base["mrstft"]:
        why.append("mrstft")
    if v["si_sdr"] < base["si_sdr"] - 0.3:
        why.append("si_sdr")
    if np.isfinite(v["anchor_drift_mel2k"]) and v["anchor_drift_mel2k"] > 1.5 * base["anchor_seed_floor_mel2k"]:
        why.append("anchor_drift")
    if abs(v["hf_floor_db"] - base["hf_floor_db"]) > 2.0:
        why.append("hf_floor")
    if v["hf_env_corr"] < base["hf_env_corr"] - 0.02:
        why.append("hf_env_corr")
    return not why, why


def lr_at(main, args):
    if main < args.lr_warmup:
        return args.lr * (main + 1) / max(1, args.lr_warmup)
    start = int(0.7 * args.steps)
    if main < start:
        return args.lr
    t = min(1.0, (main - start) / max(1, args.steps - start))
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * t))


def checksum(m):
    with torch.no_grad():
        return float(sum(p.double().sum() for p in m.parameters()))


def mem_gb():
    return round(torch.cuda.max_memory_reserved() / 1e9, 3), round(torch.cuda.max_memory_allocated() / 1e9, 3)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.dev = "cuda"
        self.run = os.path.join(dc.RUNS, args.name)
        self.man = dc.load_json(dc.MANIFEST)
        self.audit = dc.audit(self.man)
        print("audit %s" % json.dumps(self.audit), flush=True)
        self.index = dc.load_json(os.path.join(dc.CACHE, "index.json"))
        if self.index["meta"]["manifest_md5"] != dc.file_md5(dc.MANIFEST):
            raise SystemExit("cache_v1 was built from another manifest; rerun dec_data.py cache")
        gi = os.path.join(dc.GEN, "index.json")
        self.gen_index = dc.load_json(gi) if os.path.isfile(gi) else {}
        sug = self.gen_index.get("suggest", {})
        self.gain = pair(args.gain_db, tuple(sug.get("gain_db", (-20.0, -1.0))))
        self.onset = pair(args.onset_range, tuple(sug.get("onset_s", (0.02, 0.35))))
        self.stereo = {"on": True, "off": False}.get(args.pseudo_stereo, bool(sug.get("pseudo_stereo", False)))
        self.train = Data(self.index, "train")
        self.val = Data(self.index, "val")
        self.sampler = Sampler(self.train, args.seed, self.gain, self.onset, args.mix_prob, self.stereo)
        self.aug_rng = np.random.default_rng(args.seed + 1)
        self.step_rng = np.random.default_rng(args.seed + 2)
        torch.manual_seed(args.seed)

    def load_models(self):
        a = self.args
        t0 = time.perf_counter()
        self.ae = dc.load_ae(self.dev)
        self.dec = self.ae.decoder
        self.dec.requires_grad_(False)
        for k, p in self.dec.named_parameters():
            p.requires_grad_(a.scope == "full" or k.startswith(LAST3))
        self.ref = dc.fresh_decoder(self.dev).requires_grad_(False)
        self.ema = dc.fresh_decoder(self.dev).requires_grad_(False)
        self.ae.bottleneck.train(a.bn_noise == "train")
        self.train_params = [p for p in self.dec.parameters() if p.requires_grad]
        self.ema_params = [p for (k, p), (k2, q) in zip(self.ema.named_parameters(), self.dec.named_parameters())
                           if q.requires_grad]
        self.dec_params = [p for p in self.dec.parameters() if p.requires_grad]
        kinds = tuple(k.strip() for k in a.disc.split(",") if k.strip())
        wins = tuple(int(v) for v in a.mbd_windows.split(",") if v.strip())
        self.disc = dd.Disc(kinds, wins).to(self.dev)
        self.opt_g = torch.optim.AdamW(self.train_params, lr=a.lr, betas=(0.8, 0.99), weight_decay=0.0)
        self.opt_d = torch.optim.AdamW(self.disc.parameters(), lr=a.d_lr, betas=(0.8, 0.99), weight_decay=0.0)
        print("models loaded %.1f s: decoder %.2fM trainable %.2fM, disc %.2fM (%s, mbd %s), bn_noise %s" % (
            time.perf_counter() - t0, dd.n_params(self.dec) / 1e6, sum(p.numel() for p in self.train_params) / 1e6,
            dd.n_params(self.disc) / 1e6, ",".join(kinds), list(wins), a.bn_noise), flush=True)
        an = os.path.join(dc.GEN, "anchor.npz")
        av = os.path.join(dc.GEN, "anchor_val.npz")
        need = a.w_anchor > 0 or a.w_anchor_hf > 0 or a.w_teacher > 0 and a.teacher or a.w_gen_adv > 0
        if need and not os.path.isfile(an):
            raise SystemExit("missing %s: run dec_eval.py record first" % an)
        self.anchor = None
        self.anchor_keys = []
        if os.path.isfile(an):
            d = np.load(an)
            self.anchor = torch.from_numpy(d["z"][..., :dc.SHORT_FRAMES].astype(np.float32))
            self.anchor_keys = [str(k) for k in d["keys"]]
        self.anchor_val = None
        if os.path.isfile(av):
            d = np.load(av)
            za = d["z"][..., :dc.SHORT_FRAMES].astype(np.float32)
            if a.anchor_val:
                za = za[:a.anchor_val]
            self.anchor_val = torch.from_numpy(za).to(self.dev)
        self.teacher = None
        if a.teacher:
            tp = dc.at_root(a.teacher)
            tr = dc.load_json(os.path.join(os.path.dirname(tp), "teacher_res.json"))
            self.teacher = np.memmap(tp, dtype=np.float16, mode="r").reshape(-1, 2, 2, dc.SHORT_SEG)
            self.teacher_ffts = tuple(int(n) for n in tr["resolutions"])
            print("teacher %s resolutions %s" % (tp, self.teacher_ffts), flush=True)

    def fresh_state(self):
        return {"step": 0, "in_warmup": self.args.d_warmup > 0, "warmup_target": self.args.d_warmup,
                "warmup_done": 0 if self.args.d_warmup > 0 else 0, "best": None, "best_step": None, "bad_vals": 0,
                "val_hist": [], "gp_gamma": float(self.args.gp), "gp_auto": None, "dsteps": 0, "nan_run": 0,
                "gap_run": 0, "dlow_run": 0, "base_val": None, "lr": 0.0, "stopped": None}

    def main_step(self):
        s = self.st
        return 0 if s["in_warmup"] else s["step"] - s["warmup_done"]

    def save_state(self, why):
        s = self.st
        state = {"st": s, "dec": self.dec.state_dict(), "ema": self.ema.state_dict(), "disc": self.disc.state_dict(),
                 "opt_g": self.opt_g.state_dict(), "opt_d": self.opt_d.state_dict(), "sampler": self.sampler.state(),
                 "aug_rng": self.aug_rng.bit_generator.state, "step_rng": self.step_rng.bit_generator.state,
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                 "args": vars(self.args)}
        p = os.path.join(self.run, "state.pt")
        self.saved_step = s["step"]
        torch.save(state, p + ".tmp")
        os.replace(p + ".tmp", p)
        print("STATE saved (%s) step %d main %d lr %.3g sampler %s ema %.6f -> %s (%.2f GB)" % (
            why, s["step"], self.main_step(), s["lr"], self.sampler.hash(), checksum(self.ema), p,
            os.path.getsize(p) / 1e9), flush=True)

    def load_state(self):
        p = os.path.join(self.run, "state.pt")
        state = torch.load(p, map_location=self.dev, weights_only=False)
        self.dec.load_state_dict(state["dec"], strict=True)
        self.ema.load_state_dict(state["ema"], strict=True)
        self.disc.load_state_dict(state["disc"], strict=True)
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.sampler.set_state(state["sampler"])
        self.aug_rng.bit_generator.state = state["aug_rng"]
        self.step_rng.bit_generator.state = state["step_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
        self.st = state["st"]
        print("RESUMED step %d main %d lr %.3g sampler %s ema %.6f warmup %s best %s" % (
            self.st["step"], self.main_step(), self.st["lr"], self.sampler.hash(), checksum(self.ema),
            self.st["in_warmup"], self.st["best"]), flush=True)

    def meta(self, extra=None):
        m = {"run": self.args.name, "args": vars(self.args), "manifest_md5": dc.file_md5(dc.MANIFEST),
             "snapshot": dc.SNAP_ID, "step": self.st["step"], "main_step": self.main_step(),
             "licenses": {k: dc.LICENSES[k] for k in ("foleyset", "c3gd")},
             "attribution": self.man["meta"]["attribution"]}
        m.update(extra or {})
        return m

    def write_route(self):
        tv = [r for r in self.man["records"] if r["set"] in ("train", "val")]
        dc.dump_json({"clusters": sorted({r["cluster"] for r in tv}), "sources": sorted({r["path"] for r in tv}),
                      "note": "SAME-S decoder fine-tune %s: trained on the train set and selected on the val set of "
                              "%s" % (self.args.name, os.path.relpath(dc.MANIFEST, dc.ROOT)),
                      "manifest_md5": dc.file_md5(dc.MANIFEST)}, os.path.join(self.run, "route.json"))

    def startup_checks(self):
        a = self.args
        bp = os.path.join(self.run, "decoder_base.safetensors")
        dc.export_decoder(self.dec, bp, self.meta({"kind": "base"}))
        dc.load_decoder(self.ref, bp)
        x, mask, w, _ids = self.sampler.batch(min(2, a.batch), 0)
        x = torch.from_numpy(x).to(self.dev)
        with torch.no_grad():
            z = self.ae.encode(x)
            mode = self.ae.bottleneck.training
            self.ae.bottleneck.train(False)
            torch.manual_seed(123)
            y_ref = self.ae.decode(z)
            self.ae.bottleneck.train(mode)
            y_tr = dc.seeded_decode(self.ae, self.dec, z, 123)
        d_eq = float((y_ref - y_tr).abs().max())
        if d_eq > 1e-4:
            raise SystemExit("startup: decode_train differs from ae.decode by %.3g" % d_eq)
        y = dc.decode_train(self.ae, self.dec, z)
        loss = dc.mrstft(dc.mid(y), dc.mid(x))
        loss.backward()
        missing = [k for k, p in self.dec.named_parameters() if p.requires_grad and p.grad is None]
        enc_grad = [k for k, p in self.ae.encoder.named_parameters() if p.requires_grad or p.grad is not None]
        self.opt_g.zero_grad(set_to_none=True)
        if missing or enc_grad:
            raise SystemExit("startup: params without grad %s, encoder params with grad %s" % (missing[:5], enc_grad[:5]))
        anc = float("nan")
        if self.anchor is not None:
            zg = self.anchor[:max(1, a.anchor_batch)].to(self.dev)
            yg, rg = dc.twin_decode(self.ae, self.dec, self.ref, zg)
            with torch.no_grad():
                anc = float(dc.mrstft(dc.mid(yg), dc.mid(rg), fmax=4000.0) +
                            0.2 * dc.mrstft(dc.side(yg), dc.side(rg), den=dc.mid(rg)))
                anc_hf = float((dc.band_env(dc.mid(yg)) - dc.band_env(dc.mid(rg))).abs().mean())
            if anc != 0.0 or anc_hf != 0.0:
                raise SystemExit("startup: anchor loss at step 0 is %.3g / %.3g, expected exactly 0" % (anc, anc_hf))
        if self.ae.bottleneck.training != (a.bn_noise == "train"):
            raise SystemExit("startup: bottleneck mode %s" % self.ae.bottleneck.training)
        print("STARTUP OK: decode_train vs ae.decode max diff %.3g, all %d trainable params get grads, encoder none, "
              "decoder_base reloads strict, anchor loss at step 0 = %s, bottleneck train=%s" % (
                  d_eq, len(self.train_params), anc, self.ae.bottleneck.training), flush=True)

    def build_val(self):
        a = self.args
        t0 = time.perf_counter()
        self.vs = ValSet(self.val, a.val_clips)
        self.vs.encode(self.ae, self.dev)
        crops = ValSet(self.val, a.d_acc_crops)
        crops.encode(self.ae, self.dev)
        self.acc_x, self.acc_z = crops.x, crops.z
        n = min(a.watch_crops, len(crops.x))
        self.watch_x, self.watch_z = crops.x[:n], crops.z[:n]
        print("val set %d clips (%d unclipped transient), D-acc crops %d, watch crops %d (%.1f s)" % (
            len(self.vs.ents), sum(r["tc"] for r in self.vs.ref), len(self.acc_x), n, time.perf_counter() - t0),
            flush=True)

    def d_acc(self, x, z, dec, bs=16):
        accs = []
        mode = self.ae.bottleneck.training
        try:
            self.ae.bottleneck.train(False)
            with torch.no_grad():
                for k, i in enumerate(range(0, len(x), bs)):
                    xb = dc.mid(x[i:i + bs].to(self.dev))
                    yb = dc.mid(dc.seeded_decode(self.ae, dec, z[i:i + bs], 500 + k))
                    sc = dd.disc_scale(xb)
                    accs.append(dd.pair_acc(self.disc(xb, sc), self.disc(yb, sc)))
        finally:
            self.ae.bottleneck.train(mode)
        return float(torch.cat(accs).mean())

    def validate(self, tag):
        t0 = time.perf_counter()
        v = run_val(self.ae, self.ema, self.ref, self.vs, self.anchor_val)
        if self.ae.bottleneck.training != (self.args.bn_noise == "train"):
            raise SystemExit("bottleneck mode not restored after validation")
        v["step"] = self.st["step"]
        v["main_step"] = self.main_step()
        v["tag"] = tag
        v["seconds"] = round(time.perf_counter() - t0, 1)
        if self.st["base_val"] is None:
            self.st["base_val"] = dict(v)
        ok, why = eligible(v, self.st["base_val"])
        v["eligible"] = ok
        v["ineligible"] = why
        improved = False
        if tag != "base" and ok and (self.st["best"] is None or v["score"] < self.st["best"]):
            self.st["best"] = v["score"]
            self.st["best_step"] = self.st["step"]
            self.st["bad_vals"] = 0
            improved = True
            dc.export_decoder(self.ema, os.path.join(self.run, "decoder_ema_best.safetensors"),
                              self.meta({"kind": "ema_best", "val": v, "base_val": self.st["base_val"]}))
        elif tag != "base":
            self.st["bad_vals"] += 1
        v["improved"] = improved
        self.st["val_hist"].append(v)
        with open(os.path.join(self.run, "val.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(v, default=dc._jsonable) + "\n")
        print("VAL %s step %d main %d score %.4f (base %.4f) mrstft %.4f si_sdr %.2f floor %.1f env %.3f drift %.4f/%.4f "
              "eligible %s %s%s (%.1f s)" % (
                  tag, v["step"], v["main_step"], v["score"], self.st["base_val"]["score"], v["mrstft"], v["si_sdr"],
                  v["hf_floor_db"], v["hf_env_corr"], v["anchor_drift_mel2k"], v["anchor_seed_floor_mel2k"], ok, why,
                  " BEST" if improved else "", v["seconds"]), flush=True)
        return v

    def rec_terms(self, yl, xt, mt, masked):
        a = self.args
        ym, xm = dc.mid(yl), dc.mid(xt)
        fm_ = mt if masked else None
        t = {"mr_mid": dc.mrstft(ym, xm, frame_mask=fm_), "mr_side": dc.mrstft(dc.side(yl), dc.side(xt), den=xm)}
        if a.w_hf > 0:
            t["hf"] = dc.mrstft(ym, xm, ffts=dc.HF_FFTS, fmin=4000.0, frame_mask=fm_)
        return t

    def adv_terms(self, yl, xt, w, wsum, sel):
        ym, xm = dc.mid(yl[sel]), dc.mid(xt[sel])
        xa, ya = dd.paired_aug(xm, ym, self.aug_rng)
        sc = dd.disc_scale(xm)
        with torch.no_grad():
            out_r = self.disc(xa, sc)
        out_f = self.disc(ya, sc)
        return {"adv": dd.g_adv(out_r, out_f, w[sel], wsum), "fm": dd.fm_loss(out_r, out_f, w[sel], wsum)}

    def weights(self):
        a = self.args
        return {"mr_mid": a.w_mid, "mr_side": a.w_side, "hf": a.w_hf, "adv": a.w_adv, "fm": a.w_fm,
                "anchor_lo": a.w_anchor, "anchor_hf": a.w_anchor_hf, "teacher": a.w_teacher, "gen_adv": a.w_gen_adv}

    def anchor_terms(self, zg, sel_idx):
        a = self.args
        yg, rg = dc.twin_decode(self.ae, self.dec, self.ref, zg)
        yg, rg = yg[..., :dc.SHORT_SEG], rg[..., :dc.SHORT_SEG]
        t = {}
        if a.w_anchor > 0:
            t["anchor_lo"] = dc.mrstft(dc.mid(yg), dc.mid(rg), fmax=4000.0) + \
                0.2 * dc.mrstft(dc.side(yg), dc.side(rg), den=dc.mid(rg))
        if a.w_anchor_hf > 0:
            t["anchor_hf"] = (dc.band_env(dc.mid(yg)) - dc.band_env(dc.mid(rg))).abs().mean()
        if self.teacher is not None and a.w_teacher > 0:
            sd = self.step_rng.integers(0, 2, len(sel_idx))
            tg = np.stack([self.teacher[i, s].astype(np.float32).mean(0) for i, s in zip(sel_idx, sd)])
            t["teacher"] = dc.mrstft(dc.mid(yg), torch.from_numpy(tg).to(self.dev), ffts=self.teacher_ffts)
        return t, yg

    def gen_pairs(self, sel_idx):
        reals = []
        for i in sel_idx:
            m = dc.KEY_BUCKET.get("|".join(self.anchor_keys[i].split("|")[:2]) if self.anchor_keys else "", None)
            reals.append(self.sampler.real_crop(m)[:dc.SHORT_SEG])
        return torch.from_numpy(np.stack(reals)).to(self.dev)

    def watch_acts(self):
        hooks, acts = [], {}
        blk = self.dec.layers[3]
        for i, m in enumerate(blk.transformers):
            hooks.append(m.register_forward_hook(lambda mod, inp, out, i=i: acts.__setitem__(
                "block%d" % i, max(acts.get("block%d" % i, 0.0), float(out.detach().abs().max())))))
        hooks.append(blk.mapping.register_forward_hook(lambda mod, inp, out: acts.__setitem__(
            "mapping", max(acts.get("mapping", 0.0), float(out.detach().abs().max())))))
        return hooks, acts

    def step(self):
        a, s = self.args, self.st
        dev = self.dev
        warm = s["in_warmup"]
        main = self.main_step()
        lr = 0.0 if warm else lr_at(main, a)
        for g in self.opt_g.param_groups:
            g["lr"] = lr
        s["lr"] = lr
        xb, mb, wb, _ids = self.sampler.batch(a.batch, a.chirps)
        x = torch.from_numpy(xb).to(dev, non_blocking=True)
        mask = torch.from_numpy(mb).to(dev, non_blocking=True)
        w = torch.from_numpy(wb).to(dev)
        short = self.step_rng.uniform() < a.short_prob
        do_d = self.step_rng.uniform() < a.d_prob or warm
        watch = (not warm) and (main % a.watch_every == 0)
        sel_idx = None
        if self.anchor is not None and not warm:
            sel_idx = self.step_rng.choice(len(self.anchor), size=min(a.anchor_batch, len(self.anchor)), replace=False)
        T = dc.SHORT_SEG if short else dc.SEG
        with torch.no_grad():
            z = self.ae.encode(x)
        zin = z[..., :dc.SHORT_FRAMES] if short else z
        xt, mt = x[..., :T], mask[:, :T]
        masked = bool((mt < 0.5).any())
        hooks, acts = self.watch_acts() if watch else ([], {})
        if warm:
            with torch.no_grad():
                y = dc.decode_train(self.ae, self.dec, zin)[..., :T]
        else:
            y = dc.decode_train(self.ae, self.dec, zin)[..., :T]
        for h in hooks:
            h.remove()
        log = {"short": short}
        xm = dc.mid(xt)
        if do_d:
            self.disc.requires_grad_(True)
            xa, ya = dd.paired_aug(xm, dc.mid(y).detach(), self.aug_rng)
            sc = dd.disc_scale(xm)
            self.opt_d.zero_grad(set_to_none=True)
            dl, acc = dd.d_step(self.disc, xa, ya, w, sc, a.d_micro)
            gpv = None
            if s["gp_gamma"] > 0 and s["dsteps"] % 4 == 0:
                gpv = dd.gp_backward(self.disc, xa, ya, sc, s["gp_gamma"])
            dn = float(torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 10.0))
            if math.isfinite(dl) and math.isfinite(dn):
                self.opt_d.step()
            s["dsteps"] += 1
            log.update({"d_loss": dl, "d_acc": float(acc.mean()), "d_gn": dn, "gp": gpv})
            ph = "warmup" if warm else "main"
            if not self.mem_marks.get("d"):
                self.mem_marks["d"] = mem_gb()
                print("MEM after first %s D step: reserved %.2f GB, allocated %.2f GB" % ((ph,) + self.mem_marks["d"]),
                      flush=True)
                self.check_budget("first %s D step" % ph)
            if gpv is not None and not self.mem_marks.get("gp"):
                self.mem_marks["gp"] = mem_gb()
                print("MEM after first %s gp step: reserved %.2f GB, allocated %.2f GB" % ((ph,) + self.mem_marks["gp"]),
                      flush=True)
                self.check_budget("first %s gp step" % ph)
        if warm:
            return log
        self.disc.requires_grad_(False)
        W = self.weights()
        yl = y.detach().requires_grad_(True)
        wsum = w.sum().clamp(min=1e-6)
        tv = {}
        bal2 = {}

        def run_terms(t):
            tot = 0.0
            for k, v in t.items():
                tot = tot + W[k] * v
                tv[k] = tv.get(k, 0.0) + float(v.detach())
                if watch and W[k] > 0:
                    g, = torch.autograd.grad(W[k] * v, yl, retain_graph=True)
                    bal2[k] = bal2.get(k, 0.0) + float(g.square().sum())
            tot.backward()

        run_terms(self.rec_terms(yl, xt, mt, masked))
        if a.w_adv > 0 or a.w_fm > 0:
            for sel in torch.arange(y.shape[0], device=dev).chunk(max(1, min(a.g_micro, y.shape[0]))):
                run_terms(self.adv_terms(yl, xt, w, wsum, sel))
        bal = {k: math.sqrt(v) for k, v in bal2.items()}
        gy = yl.grad
        surrogate = (y * gy).sum()
        ta = {}
        gen_d = None
        if sel_idx is not None:
            zg = self.anchor[sel_idx].to(dev, non_blocking=True)
            ta, yg = self.anchor_terms(zg, sel_idx)
            if a.w_gen_adv > 0:
                real = self.gen_pairs(sel_idx)
                sc = dd.disc_scale(real)
                with torch.no_grad():
                    out_r = self.disc(real, sc)
                out_f = self.disc(dc.mid(yg), sc)
                ones = torch.ones(len(sel_idx), device=dev)
                ta["gen_adv"] = dd.g_adv(out_r, out_f, ones, ones.sum())
                if do_d:
                    gen_d = (real, dc.mid(yg).detach(), sc, ones)
            if watch:
                for k, v in ta.items():
                    if W[k] > 0:
                        g, = torch.autograd.grad(W[k] * v, yg, retain_graph=True)
                        bal[k] = float(g.norm())
            for k, v in ta.items():
                surrogate = surrogate + W[k] * v
                tv[k] = float(v.detach())
        self.opt_g.zero_grad(set_to_none=True)
        surrogate.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(self.train_params, 5.0))
        total = sum(W[k] * v for k, v in tv.items())
        if math.isfinite(gn) and math.isfinite(total):
            self.opt_g.step()
            with torch.no_grad():
                torch._foreach_lerp_(self.ema_params, self.dec_params, 1.0 - a.ema)
            s["nan_run"] = 0
            if gen_d is not None:
                real, fake, sc, ones = gen_d
                self.disc.requires_grad_(True)
                self.opt_d.zero_grad(set_to_none=True)
                gl = a.w_gen_adv / max(a.w_adv, 1e-6) * dd.d_loss(self.disc(real, sc), self.disc(fake, sc), ones, ones.sum())
                gl.backward()
                torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 10.0)
                self.opt_d.step()
                self.disc.requires_grad_(False)
                log["gen_d_loss"] = float(gl.detach())
        else:
            self.opt_g.zero_grad(set_to_none=True)
            s["nan_run"] += 1
            print("NONFINITE step %d (run %d): total %s gn %s" % (s["step"] + 1, s["nan_run"], total, gn), flush=True)
            if s["nan_run"] >= 20:
                raise SystemExit("20 non-finite steps in a row")
        if s["step"] == 0 or self.bn_check:
            if self.ae.bottleneck.training != (a.bn_noise == "train"):
                raise SystemExit("bottleneck mode changed during training")
        log.update({k: v for k, v in tv.items()})
        log.update({"g_loss": total, "g_gn": gn, "lr": lr})
        if watch:
            log["acts"] = acts
            if any(v > 30000 for v in acts.values()):
                print("WARNING activation above 30000: %s" % acts, flush=True)
            if bal:
                rec = math.sqrt(sum(bal.get(k, 0.0) ** 2 for k in ("mr_mid", "mr_side", "hf")))
                gan = math.sqrt(sum(bal.get(k, 0.0) ** 2 for k in ("adv", "fm")))
                bal["gan_over_rec"] = gan / max(rec, 1e-12)
                log["grad_y"] = bal
        if not self.mem_marks.get("g"):
            self.mem_marks["g"] = mem_gb()
            print("MEM after first G step: reserved %.2f GB, allocated %.2f GB" % self.mem_marks["g"], flush=True)
            self.check_budget("first G step")
        return log

    def check_budget(self, where):
        r, _al = mem_gb()
        if r > self.args.vram_budget_gb:
            self.save_state("over vram budget")
            raise SystemExit("max_memory_reserved %.2f GB after %s is above the %.1f GB budget; fall back to --g-micro 2, "
                             "then --mbd-windows 2048,1024,512,256, then --batch 12" % (r, where, self.args.vram_budget_gb))

    def end_warmup_check(self):
        a, s = self.args, self.st
        acc = self.d_acc(self.acc_x, self.acc_z, self.dec)
        extra = s["warmup_target"] - a.d_warmup
        if acc < a.d_warmup_acc and extra < a.d_warmup_extra:
            s["warmup_target"] += 500 if not a.smoke else 5
            print("D WARMUP acc %.3f < %.2f on %d val crops: extend to %d" % (acc, a.d_warmup_acc, len(self.acc_x),
                                                                             s["warmup_target"]), flush=True)
            return
        s["in_warmup"] = False
        s["warmup_done"] = s["step"]
        self.mem_marks.pop("d", None)
        self.mem_marks.pop("gp", None)
        print("D WARMUP done at step %d: pairwise acc %.3f on %d val crops" % (s["step"], acc, len(self.acc_x)),
              flush=True)

    def overfit_watch(self, log):
        a, s = self.args, self.st
        tr = float(np.mean(self.recent_acc)) if self.recent_acc else float("nan")
        va = self.d_acc(self.watch_x, self.watch_z, self.dec)
        gap = tr - va
        n_need = max(1, 1000 // a.watch_every)
        s["gap_run"] = s["gap_run"] + 1 if gap > 0.15 else 0
        dlow = np.mean(self.recent_dl) < 0.1 if self.recent_dl else False
        s["dlow_run"] = s["dlow_run"] + 1 if dlow else 0
        if s["gp_gamma"] <= 0 and (s["gap_run"] >= n_need or s["dlow_run"] >= n_need):
            s["gp_gamma"] = 1.0
            s["gp_auto"] = {"step": s["step"], "gap_run": s["gap_run"], "dlow_run": s["dlow_run"]}
            print("GP ON (gamma 1.0) at step %d: gap run %d, low D-loss run %d" % (s["step"], s["gap_run"], s["dlow_run"]),
                  flush=True)
        log["d_acc_train"] = tr
        log["d_acc_val"] = va

    def train_loop(self):
        a, s = self.args, self.st
        jl = open(os.path.join(self.run, "train.jsonl"), "a", encoding="utf-8")
        win = []
        self.recent_acc = deque(maxlen=max(1, a.watch_every))
        self.recent_dl = deque(maxlen=max(1, a.watch_every))
        t_last = time.perf_counter()
        n_last = s["step"]
        total_steps = lambda: s["warmup_target"] + a.steps
        try:
            while True:
                if not s["in_warmup"] and self.main_step() >= a.steps:
                    s["stopped"] = "steps"
                    break
                log = self.step()
                s["step"] += 1
                if "d_acc" in log:
                    self.recent_acc.append(log["d_acc"])
                    self.recent_dl.append(log["d_loss"])
                win.append(log)
                if s["in_warmup"] and s["step"] >= s["warmup_target"]:
                    self.end_warmup_check()
                main = self.main_step()
                if not s["in_warmup"] and main > 0 and main % a.watch_every == 0:
                    self.overfit_watch(log)
                if s["step"] % a.log_every == 0:
                    now = time.perf_counter()
                    sps = (s["step"] - n_last) / max(now - t_last, 1e-9)
                    t_last, n_last = now, s["step"]
                    agg = {}
                    for k in {k for l in win for k in l}:
                        vals = [l[k] for l in win if isinstance(l.get(k), (int, float)) and l.get(k) is not None]
                        if vals:
                            agg[k] = float(np.mean(vals))
                    extra = {k: l[k] for l in win for k in ("acts", "grad_y", "d_acc_train", "d_acc_val") if k in l}
                    rec = {"step": s["step"], "main": main, "warmup": s["in_warmup"], **agg, **extra,
                           "steps_per_s": sps, "mem_reserved_gb": mem_gb()[0],
                           "eta_h": (total_steps() - s["step"]) / max(sps, 1e-9) / 3600}
                    jl.write(json.dumps(rec, default=dc._jsonable) + "\n")
                    jl.flush()
                    print("step %d main %d %s lr %.2e g %.4f mr %.4f adv %.3f fm %.3f anc %.4f/%.4f d %.3f acc %.2f "
                          "gn %.2f | %.2f it/s eta %.2f h mem %.2f GB" % (
                              s["step"], main, "warm" if s["in_warmup"] else "main", agg.get("lr", 0.0),
                              agg.get("g_loss", float("nan")), agg.get("mr_mid", float("nan")),
                              agg.get("adv", float("nan")), agg.get("fm", float("nan")),
                              agg.get("anchor_lo", float("nan")), agg.get("anchor_hf", float("nan")),
                              agg.get("d_loss", float("nan")), agg.get("d_acc", float("nan")),
                              agg.get("g_gn", float("nan")), sps, rec["eta_h"], rec["mem_reserved_gb"]), flush=True)
                    win = []
                    self.check_budget("step %d" % s["step"])
                    if not self.mem_marks.get("log"):
                        self.mem_marks["log"] = mem_gb()
                        print("MEM after first logging step: reserved %.2f GB, allocated %.2f GB" % self.mem_marks["log"],
                              flush=True)
                        self.check_budget("first logging step")
                if not s["in_warmup"] and main > 0 and main % a.val_every == 0:
                    self.validate("val")
                if not s["in_warmup"] and main > 0 and main % a.export_every == 0:
                    dc.export_decoder(self.ema, os.path.join(self.run, "dec_ema_s%05d.safetensors" % s["step"]),
                                      self.meta({"kind": "ema"}))
                if s["step"] % a.state_every == 0:
                    self.save_state("every %d" % a.state_every)
                if (not s["in_warmup"] and main >= a.min_stop_step and s["bad_vals"] >= a.patience):
                    s["stopped"] = "early stop after %d validations without improvement" % s["bad_vals"]
                    print("EARLY STOP at step %d" % s["step"], flush=True)
                    break
                stop_file = os.path.join(self.run, "STOP")
                if os.path.isfile(stop_file):
                    if self.saved_step != s["step"]:
                        self.save_state("STOP file")
                    os.remove(stop_file)
                    print("STOP file seen: state saved at step %d, exiting" % s["step"], flush=True)
                    return "stop_file"
                if a.stop_at is not None and s["step"] >= a.stop_at:
                    if self.saved_step != s["step"]:
                        self.save_state("--stop-at %d" % a.stop_at)
                    print("STOP-AT %d reached" % a.stop_at, flush=True)
                    return "stop_at"
        finally:
            jl.close()
        dc.export_decoder(self.ema, os.path.join(self.run, "decoder_ema_last.safetensors"),
                          self.meta({"kind": "ema_last", "best_step": s["best_step"], "best": s["best"]}))
        if self.saved_step != s["step"]:
            self.save_state("final")
        print("DONE %s at step %d (main %d): best score %s at step %s" % (s["stopped"], s["step"], self.main_step(),
                                                                         s["best"], s["best_step"]), flush=True)
        return "done"

    def go(self):
        a = self.args
        sp = os.path.join(self.run, "state.pt")
        if a.smoke and not a.resume and os.path.isdir(self.run):
            shutil.rmtree(self.run)
        if os.path.isfile(sp) and not a.resume:
            raise SystemExit("%s exists; pass --resume or choose another --name" % sp)
        if a.resume and not os.path.isfile(sp):
            raise SystemExit("--resume: no %s" % sp)
        os.makedirs(self.run, exist_ok=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
        self.mem_marks = {}
        self.saved_step = None
        self.bn_check = True
        self.load_models()
        self.build_val()
        dc.dump_json(vars(a), os.path.join(self.run, "args.json"))
        self.write_route()
        print("data: gain %s dBFS, onset range %s s, pseudo-stereo %s, train clips %d, buckets %s" % (
            self.gain, self.onset, self.stereo, len(self.train.ents),
            {m: len(t["idx"]) for m, t in self.sampler.tabs.items()}), flush=True)
        if a.resume:
            self.load_state()
        else:
            self.st = self.fresh_state()
            self.startup_checks()
            self.validate("base")
        return self.train_loop()


def main(argv=None):
    args = parse(argv)
    with dc.gpu_guard("train:%s" % args.name):
        t = Trainer(args)
        r = t.go()
    print("exit %s" % r, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
