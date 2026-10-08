import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(HERE), "bench")
for _p in (BENCH, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bootstrap

import argparse
import gc
import json
import math
import random
import time
from collections import defaultdict
from functools import partial

import numpy as np
import torch

import encode_latents as el

ROOT = el.ROOT
RUNS = os.path.join("data", "lora", "runs")
HELDOUT_MANIFEST = os.path.join("data", "refs_heldout", "manifest.json")
TARGETS = {
    "self": ["self_attn.to_qkv", "self_attn.to_out"],
    "cross": ["cross_attn.to_q", "cross_attn.to_kv", "cross_attn.to_out"],
    "ff": ["ff.ff.0.proj", "ff.ff.2"],
}
VAL_T = [0.1, 0.3, 0.5, 0.7, 0.9]
INPAINT = "0.1,0.8,0.1"
BUCKETS = 10


def rel(p):
    return el.rel(p)


def seed_all(seed, deterministic):
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def free_gpu(tag=""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("gpu after %s: %.0f MB allocated, %.0f MB reserved" % (
            tag or "free", torch.cuda.memory_allocated() / 2 ** 20, torch.cuda.memory_reserved() / 2 ** 20), flush=True)


def load_sam(args):
    sam = el.load_model(half=args.precision == "fp16")
    if args.precision == "bf16":
        sam.model.model.to(torch.bfloat16)
    return sam


def lora_config(args):
    inc = []
    for k in args.targets:
        for leaf in TARGETS[k]:
            p = "transformer.layers.[%s].%s" % (args.layers, leaf) if args.layers else leaf
            if p not in inc:
                inc.append(p)
    return {"rank": int(args.rank), "alpha": float(args.alpha), "adapter_type": "lora", "include": inc,
            "exclude": None}


def attach(dit, cfg):
    from stable_audio_3.models.lora import LoRAParametrization, add_lora, get_lora_layers
    kw = {"rank": int(cfg["rank"]), "lora_alpha": float(cfg.get("alpha", cfg["rank"])),
          "adapter_type": cfg.get("adapter_type", "lora")}
    conf = {torch.nn.Linear: {"weight": partial(LoRAParametrization.from_linear, **kw)},
            torch.nn.Conv1d: {"weight": partial(LoRAParametrization.from_conv1d, **kw)}}
    add_lora(dit, conf, include=cfg.get("include"), exclude=cfg.get("exclude"))
    layers = get_lora_layers(dit)
    if not layers:
        raise SystemExit("no layer matched the LoRA include patterns %s" % cfg.get("include"))
    return layers


def lora_params(dit):
    from stable_audio_3.models.lora import get_lora_params
    return list(get_lora_params(dit))


def export(dit, cfg, path):
    from stable_audio_3.models.lora import get_lora_state_dict, save_lora_safetensors
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_lora_safetensors(get_lora_state_dict(dit), cfg, path)


def merge_into(dit, path, strength=1.0):
    from torch.nn.utils import parametrize
    from stable_audio_3.models.lora import load_lora_checkpoint, name_is_lora, set_lora_strength
    p = el.at_root(path)
    sd, cfg = load_lora_checkpoint(p)
    if cfg.get("adapter_type", "lora") != "lora":
        raise SystemExit("merge supports plain LoRA, %s has adapter_type %s" % (rel(p), cfg.get("adapter_type")))
    attach(dit, cfg)
    want = {k for k in dit.state_dict() if name_is_lora(k)}
    res = dit.load_state_dict(sd, strict=False)
    missing = sorted(want - set(sd))
    if missing or res.unexpected_keys:
        raise SystemExit("LoRA %s does not fit the model: %d missing, %d unexpected (first %s)" % (
            rel(p), len(missing), len(res.unexpected_keys), (missing or list(res.unexpected_keys))[:2]))
    set_lora_strength(dit, float(strength))
    n = 0
    with torch.no_grad():
        for _name, mod in list(dit.named_modules()):
            pl = getattr(getattr(mod, "parametrizations", None), "weight", None)
            if pl is None:
                continue
            merged = mod.weight.detach().clone()
            ptr = pl.original.data_ptr()
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)
            mod.weight.copy_(merged)
            if mod.weight.data_ptr() != ptr:
                raise SystemExit("merging moved the storage of a weight")
            n += 1
    return {"path": rel(p), "layers": n, "rank": cfg["rank"], "alpha": cfg.get("alpha"), "strength": float(strength),
            "sarge": cfg.get("sarge", {})}


def install_hook(path, strength=1.0):
    from stable_audio_3 import model as sa3m
    cls = sa3m.StableAudioModel
    if "_sarge_orig" not in cls.__dict__:
        cls._sarge_orig = staticmethod(cls.from_pretrained)
    orig = cls._sarge_orig
    if path is None:
        cls.from_pretrained = staticmethod(orig)
        return None

    def from_pretrained(model_name, device=None, model_half=True):
        sam = orig(model_name, device=device, model_half=model_half)
        sam.lora = merge_into(sam.model.model, path, strength)
        print("lora merged into %s: %s" % (model_name, json.dumps({k: v for k, v in sam.lora.items() if k != "sarge"})),
              flush=True)
        return sam

    cls.from_pretrained = staticmethod(from_pretrained)
    return from_pretrained


def lora_meta(path):
    from safetensors import safe_open
    with safe_open(el.at_root(path), framework="pt") as f:
        md = f.metadata() or {}
    return json.loads(md.get("lora_config", "{}"))


def lora_sources(path):
    cache = lora_meta(path).get("sarge", {}).get("cache")
    ip = os.path.join(el.at_root(cache), "index.json") if cache else None
    if not ip or not os.path.isfile(ip):
        raise SystemExit("LoRA %s names no readable training cache (%s), so its sources are unknown" % (path, cache))
    idx = json.load(open(ip, encoding="utf-8"))
    audio = idx["meta"].get("audio", "real")
    clips = idx["clips"]
    return {"cache": cache, "audio": audio, "clips": len(clips),
            "clusters": sorted({c["cluster"] for c in clips if c.get("cluster")}),
            "sources": sorted({c["src"] for c in clips if audio == "real" and c.get("src")})}


def check_leaks(path, argv):
    import vocab_eval
    va = vocab_eval.parse(list(argv))
    mp = os.path.join(vocab_eval.ROOT, va.manifest)
    man = json.load(open(mp, encoding="utf-8"))
    src = lora_sources(path)
    leak = vocab_eval.route_leaks(src, man)
    n = {k: len(v) for k, v in leak.items()}
    if any(n.values()):
        if man["meta"].get("heldout"):
            raise SystemExit("LoRA %s trained on sources that the held-out references of %s hold: %s" % (
                rel(el.at_root(path)), rel(mp), json.dumps({k: v[:3] for k, v in leak.items() if v})))
        print("LoRA %s shares sources with %s: %s" % (rel(el.at_root(path)), rel(mp), json.dumps(n)), flush=True)
    else:
        print("LoRA %s: %d training clusters from %s, none of them in %s" % (
            rel(el.at_root(path)), len(src["clusters"]), src["cache"], rel(mp)), flush=True)
    return leak


def load_cache(path, style, device):
    from safetensors.torch import load_file
    d = el.at_root(path)
    idx = json.load(open(os.path.join(d, "index.json"), encoding="utf-8"))
    tp = os.path.join(d, "text_%s.safetensors" % style)
    if not os.path.isfile(tp):
        raise SystemExit("%s has no captions in style %s (styles %s)" % (rel(d), style, idx["meta"].get("styles")))
    lat = load_file(os.path.join(d, "latents.safetensors"))
    tx = load_file(tp)
    clips = idx["clips"]
    if lat["latents"].shape[0] != len(clips) or tx["index"].shape[0] != len(clips):
        raise SystemExit("%s: %d clips in index.json, %d latents, %d caption rows" % (
            rel(d), len(clips), lat["latents"].shape[0], tx["index"].shape[0]))
    pools = {}
    for p in ("train", "val"):
        cs = [c for c in clips if c["pool"] == p]
        pools[p] = {"rows": np.array([c["row"] for c in cs], dtype=np.int64),
                    "w": np.array([c.get("weight", 1.0) for c in cs], dtype=np.float64)}
    return {"dir": rel(d), "meta": idx["meta"], "clips": clips, "lat": lat["latents"].to(device),
            "content": lat["content"].to(device).long(), "tok": tx["tokens"].to(device),
            "off": tx["offsets"].tolist(), "start": tx["starts"].tolist(), "uidx": tx["index"].tolist(),
            "pad": tx["pad"].to(device), "sec": tx["seconds"].to(device), "width": int(tx["width"][0]),
            "pools": pools}


def text_batch(C, rows, dtype):
    B, W, D = len(rows), C["width"], C["pad"].shape[0]
    emb = C["pad"].to(dtype).view(1, 1, D).expand(B, W, D).clone()
    mask = torch.zeros((B, W), dtype=torch.bool, device=emb.device)
    for b, r in enumerate(rows):
        u = C["uidx"][int(r)]
        s, e, st = C["off"][u], C["off"][u + 1], C["start"][u]
        emb[b, st:st + e - s] = C["tok"][s:e].to(dtype)
        mask[b, st:st + e - s] = True
    sec = C["sec"].to(dtype).view(1, 1, D).expand(B, 1, D).contiguous()
    return {"prompt": [emb, mask], "seconds_total": [sec, torch.ones((B, 1), device=emb.device)]}


def augment(pad, scale_s, sr, ds):
    if scale_s <= 0:
        return pad
    B, T = pad.shape
    ext = torch.empty(B, device=pad.device).exponential_(lambd=1.0 / (scale_s * sr / ds)).long()
    lens = torch.clamp(pad.sum(dim=-1) + ext, max=T)
    return torch.arange(T, device=pad.device).unsqueeze(0) < lens.unsqueeze(1)


def masked_loss(mse, mask, attn_masked, w_pad, ctx_mask=None):
    C = mse.shape[1]
    sig = torch.where(mask.unsqueeze(1), mse, 0.0)
    sig_n = mask.sum(dim=1) * C
    padv = torch.where(~mask.unsqueeze(1), mse, 0.0)
    pad_n = (~mask).sum(dim=1) * C
    if attn_masked:
        per = sig.sum(dim=(1, 2)) / (sig_n + 1e-8)
    else:
        per = (sig.sum(dim=(1, 2)) + w_pad * padv.sum(dim=(1, 2))) / (sig_n + w_pad * pad_n + 1e-8)
    loss = per.mean()
    ctx = torch.zeros((), device=mse.device)
    if ctx_mask is not None and attn_masked and w_pad > 0:
        n = ctx_mask.sum(dim=1) * C
        if n.sum() > 0:
            ctx = (torch.where(ctx_mask.unsqueeze(1), mse, 0.0).sum(dim=(1, 2)) / (n + 1e-8)).mean()
            loss = loss + ctx * w_pad
    return loss, per.detach(), ctx.detach()


def prod_schedule(w, steps, eff, T):
    from stable_audio_3.inference.sampling import build_schedule
    s = build_schedule(steps=steps, sigma_max=1.0, dist_shift=w.sampling_dist_shift,
                       effective_seq_len=torch.tensor([float(eff)]), fallback_seq_len=T, include_endpoint=True)
    s = s[0] if s.dim() == 2 else s
    return [float(v) for v in s[:-1]]


def sample_t(B, args, w, eff, ctx):
    dev = ctx["dev"]
    if args.t_sampler == "package":
        from stable_audio_3.inference.sampling import truncated_logistic_normal_rescaled
        t = (1 - truncated_logistic_normal_rescaled(B)).to(dev)
        if w.dist_shift is not None:
            t = w.dist_shift.shift(t, eff)
    else:
        base = torch.tensor(ctx["sched"], dtype=torch.float32)[torch.randint(0, len(ctx["sched"]), (B,))]
        t = torch.sigmoid(torch.logit(base.clamp(1e-4, 1 - 1e-4)) + args.t_jitter * torch.randn(B)).to(dev)
    if args.p_one_shot > 0:
        t = torch.where(torch.rand_like(t) < args.p_one_shot, torch.ones_like(t), t)
    return t.float()


def train_loss(sam, C, rows, args, ctx):
    from stable_audio_3.models.inpainting import random_inpaint_mask
    w = sam.model
    dev = ctx["dev"]
    ix = torch.as_tensor(rows, device=dev)
    x = C["lat"][ix].float()
    B, _ch, T = x.shape
    pad = torch.arange(T, device=dev).unsqueeze(0) < C["content"][ix].unsqueeze(1)
    eff = torch.full((B,), float(ctx["eff"]), device=dev)
    t = sample_t(B, args, w, eff, ctx)
    a, s = (1 - t)[:, None, None], t[:, None, None]
    noise = torch.randn_like(x)
    noised = x * a + noise * s
    target = noise - x
    aug = augment(pad, args.silence_ext, ctx["sr"], ctx["ds"])
    loss_mask = aug.to(torch.bool)
    ctx_mask = None
    if ctx["inpaint"]:
        masked, imask = random_inpaint_mask(x, padding_masks=aug, mask_padding=w.mask_padding_attention,
                                            mask_type_probabilities=list(ctx["inpaint"]))
        loss_mask = loss_mask & ~imask.squeeze(1).to(torch.bool)
        ctx_mask = imask.squeeze(1).to(torch.bool) & aug.to(torch.bool)
    else:
        masked, imask = torch.zeros_like(x), torch.zeros((B, 1, T), device=dev)
    cond = text_batch(C, rows, ctx["dtype"])
    cond["inpaint_mask"] = [imask]
    cond["inpaint_masked_input"] = [masked]
    kw = {"padding_mask": aug} if w.mask_padding_attention else {}
    out = w(noised, t, cond=cond, cfg_dropout_prob=args.cfg_dropout, use_checkpointing=args.grad_ckpt, **kw)
    mse = (out.float() - target) ** 2
    loss, per, ctxl = masked_loss(mse, loss_mask, w.mask_padding_attention, args.mask_loss_weight, ctx_mask)
    return loss, per, ctxl, t.detach()


@torch.no_grad()
def val_loss(sam, C, args, ctx):
    rows = C["pools"]["val"]["rows"]
    if not len(rows):
        return None
    w = sam.model
    dev = ctx["dev"]
    g = torch.Generator(device="cpu").manual_seed(int(args.seed) + 7919)
    tot = defaultdict(float)
    n = 0
    for i in range(0, len(rows), args.batch):
        r = rows[i:i + args.batch]
        ix = torch.as_tensor(r, device=dev)
        x = C["lat"][ix].float()
        B, _ch, T = x.shape
        pad = torch.arange(T, device=dev).unsqueeze(0) < C["content"][ix].unsqueeze(1)
        cond = text_batch(C, r, ctx["dtype"])
        cond["inpaint_mask"] = [torch.zeros((B, 1, T), device=dev)]
        cond["inpaint_masked_input"] = [torch.zeros_like(x)]
        kw = {"padding_mask": pad} if w.mask_padding_attention else {}
        for tv in ctx["val_t"]:
            noise = torch.randn(x.shape, generator=g).to(dev)
            t = torch.full((B,), tv, device=dev)
            out = w(x * (1 - tv) + noise * tv, t, cond=cond, cfg_dropout_prob=0.0, use_checkpointing=False, **kw)
            loss, _per, _c = masked_loss((out.float() - (noise - x)) ** 2, pad, w.mask_padding_attention,
                                         args.mask_loss_weight)
            tot["%.3f" % tv] += float(loss) * B
        n += B
    res = {k: v / n for k, v in tot.items()}
    res["package_mean"] = float(np.mean([res["%.3f" % tv] for tv in VAL_T]))
    res["production_mean"] = float(np.mean([res["%.3f" % tv] for tv in ctx["sched"]]))
    return res


def save_state(path, dit, opt, scaler, step, rng):
    from stable_audio_3.models.lora import get_lora_state_dict
    torch.save({"lora": {k: v.detach().float().cpu() for k, v in get_lora_state_dict(dit).items()},
                "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
                "rng": {"python": random.getstate(), "numpy": rng.bit_generator.state,
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}, path)


def load_state(path, dit, opt, scaler, rng):
    st = torch.load(path, map_location="cpu", weights_only=False)
    res = dit.load_state_dict(st["lora"], strict=False)
    if res.unexpected_keys:
        raise SystemExit("resume state has keys the model lacks: %s" % res.unexpected_keys[:2])
    opt.load_state_dict(st["opt"])
    scaler.load_state_dict(st["scaler"])
    random.setstate(st["rng"]["python"])
    rng.bit_generator.state = st["rng"]["numpy"]
    torch.set_rng_state(st["rng"]["torch"])
    if st["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["rng"]["cuda"])
    return int(st["step"])


def fmt_s(s):
    s = int(max(0, s))
    return "%d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def run_dir(args):
    return el.at_root(args.out) if args.out else el.at_root(os.path.join(RUNS, args.name))


def train(args):
    seed_all(args.seed, args.deterministic)
    run = run_dir(args)
    state_p = os.path.join(run, "state.pt")
    if os.path.isfile(state_p) and not args.resume:
        raise SystemExit("%s already holds a run; pass --resume or another --name" % rel(run))
    os.makedirs(run, exist_ok=True)
    t_load = time.perf_counter()
    sam = load_sam(args)
    w = sam.model
    dit = w.model
    dev = str(sam.device)
    dtype = next(dit.parameters()).dtype
    if dev.startswith("cuda") and w.pretransform is not None:
        w.pretransform.to("cpu")
    C = load_cache(args.cache, args.caption, dev)
    meta = C["meta"]
    if meta.get("model") != el.MODEL:
        raise SystemExit("cache %s was encoded with %s, training loads %s" % (C["dir"], meta.get("model"), el.MODEL))
    T = int(C["lat"].shape[-1])
    ds = int(w.pretransform.downsampling_ratio) if w.pretransform is not None else int(meta["downsampling_ratio"])
    sr = int(w.sample_rate)
    eff = int(math.ceil(int(meta["gen_s"] * sr) / ds))
    cfg = lora_config(args)
    cfg["sarge"] = {"cache": C["dir"], "audio": meta.get("audio", "real"), "caption": args.caption,
                    "steps": args.steps, "lr": args.lr, "batch": args.batch, "seed": args.seed,
                    "precision": args.precision, "t_sampler": args.t_sampler, "base": el.MODEL}
    layers = attach(dit, cfg)
    dit.train()
    params = lora_params(dit)
    for p in params:
        p.requires_grad_(True)
    n_lora = sum(p.numel() for p in params)
    n_base = sum(p.numel() for n_, p in dit.named_parameters() if "parametrizations" not in n_ or
                 n_.endswith(".original"))
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16" and dev.startswith("cuda"))
    sched = prod_schedule(w, args.prod_steps, eff, T)
    probs = [float(v) for v in args.inpaint.split(",")] if args.inpaint != "off" else None
    if probs and abs(sum(probs) - 1.0) > 1e-6:
        raise SystemExit("--inpaint probabilities sum to %.3f" % sum(probs))
    ctx = {"dev": dev, "dtype": dtype, "eff": eff, "sr": sr, "ds": ds, "sched": sched, "inpaint": probs,
           "val_t": sorted(set(VAL_T) | set(sched))}
    tr = C["pools"]["train"]
    if not len(tr["rows"]):
        raise SystemExit("cache %s has no train clips" % C["dir"])
    pw = tr["w"] / tr["w"].sum()
    rng = np.random.default_rng(args.seed)
    step = load_state(state_p, dit, opt, scaler, rng) if args.resume and os.path.isfile(state_p) else 0
    conf = {"args": vars(args), "lora": cfg, "cache_meta": {k: meta.get(k) for k in (
                "model", "gen_s", "pad_s", "preroll_s", "n_audio", "latent_frames", "audio", "balance", "sources")},
            "layers": len(layers), "lora_params": n_lora, "base_params": n_base, "precision": str(dtype),
            "production_schedule_t": sched, "effective_frames": eff, "latent_frames": T,
            "train_clips": int(len(tr["rows"])), "val_clips": int(len(C["pools"]["val"]["rows"])),
            "load_seconds": round(time.perf_counter() - t_load, 1)}
    json.dump(conf, open(os.path.join(run, "config.json"), "w", encoding="utf-8"), indent=1)
    print("lora: %d layers, %d params (%.2f%% of %d), rank %d alpha %g, targets %s" % (
        len(layers), n_lora, 100.0 * n_lora / max(n_base, 1), n_base, cfg["rank"], cfg["alpha"], cfg["include"]))
    print("cache %s caption %s: %d train clips, %d val clips, latents %s, effective %d of %d frames, base %s" % (
        C["dir"], args.caption, len(tr["rows"]), len(C["pools"]["val"]["rows"]), tuple(C["lat"].shape), eff, T,
        dtype))
    print("production %d-step schedule t = %s; t sampler %s; inpaint %s; cfg dropout %.2f; silence extension %.1f s"
          % (args.prod_steps, ", ".join("%.3f" % v for v in sched), args.t_sampler, args.inpaint, args.cfg_dropout,
             args.silence_ext), flush=True)
    logf = open(os.path.join(run, "log.jsonl"), "a", encoding="utf-8")
    if step == 0:
        v0 = val_loss(sam, C, args, ctx)
        if v0:
            logf.write(json.dumps({"step": 0, "val": v0}) + "\n")
            print("val step 0  package %.4f  production %.4f" % (v0["package_mean"], v0["production_mean"]), flush=True)
    win = defaultdict(list)
    bucket = defaultdict(list)
    bad = 0
    loss = per = None
    t_run = time.perf_counter()
    first = step
    while step < args.steps:
        step += 1
        rows = tr["rows"][rng.choice(len(tr["rows"]), size=args.batch, replace=True, p=pw)]
        lr = args.lr * min(1.0, step / max(1, args.warmup))
        for gp in opt.param_groups:
            gp["lr"] = lr
        t0 = time.perf_counter()
        loss, per, ctxl, tt = train_loss(sam, C, rows, args, ctx)
        if not torch.isfinite(loss):
            bad += 1
            opt.zero_grad(set_to_none=True)
            print("step %d: non-finite loss, skipped (%d so far)" % (step, bad), flush=True)
            if bad > args.max_bad:
                raise SystemExit("more than %d non-finite steps" % args.max_bad)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, args.clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        win["loss"].append(float(loss))
        win["ctx"].append(float(ctxl))
        win["gn"].append(float(gn))
        win["sec"].append(time.perf_counter() - t0)
        for tv, pv in zip(tt.cpu().tolist(), per.cpu().tolist()):
            bucket[min(BUCKETS - 1, int(tv * BUCKETS))].append(pv)
        if step % args.log_every == 0 or step == args.steps:
            sps = float(np.mean(win["sec"]))
            rec = {"step": step, "loss": float(np.mean(win["loss"])), "ctx": float(np.mean(win["ctx"])),
                   "grad_norm": float(np.mean(win["gn"])), "lr": lr, "scale": float(scaler.get_scale()),
                   "sec_per_step": sps, "loss_by_t": {"%.1f" % (b / BUCKETS): float(np.mean(v))
                                                      for b, v in sorted(bucket.items())}}
            if dev.startswith("cuda"):
                rec["max_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print("step %6d  loss %.4f  ctx %.4f  gn %.3f  lr %.2e  scale %g  %.3f s/step  eta %s%s" % (
                step, rec["loss"], rec["ctx"], rec["grad_norm"], lr, rec["scale"], sps,
                fmt_s(sps * (args.steps - step)), "  mem %.0f MB" % rec["max_mem_mb"] if "max_mem_mb" in rec else ""),
                flush=True)
            win.clear()
            bucket.clear()
        if args.val_every and step % args.val_every == 0:
            v = val_loss(sam, C, args, ctx)
            if v:
                logf.write(json.dumps({"step": step, "val": v}) + "\n")
                logf.flush()
                print("val step %d  package %.4f  production %.4f  %s" % (step, v["package_mean"],
                      v["production_mean"], "  ".join("t%s %.4f" % (k, x) for k, x in v.items() if k[0].isdigit())),
                      flush=True)
        if step % args.save_every == 0 or step == args.steps:
            export(dit, cfg, os.path.join(run, "lora_%06d.safetensors" % step))
            save_state(state_p, dit, opt, scaler, step, rng)
    final = os.path.join(run, "lora_final.safetensors")
    last = os.path.join(run, "lora_%06d.safetensors" % step)
    if not os.path.isfile(last):
        export(dit, cfg, last)
    export(dit, cfg, final)
    logf.close()
    took = time.perf_counter() - t_run
    print("trained steps %d-%d in %s (%.3f s/step), %d skipped" % (first + 1, step, fmt_s(took),
                                                                   took / max(1, step - first), bad))
    print("saved -> %s" % rel(final))
    del sam, w, dit, params, opt, scaler, C, loss, per
    free_gpu("training")
    return final


def check(args):
    sam = el.load_model(half=True)
    dit = sam.model.model
    dev = str(sam.device)
    dtype = next(dit.parameters()).dtype
    g = torch.Generator(device="cpu").manual_seed(0)
    T = 22
    x = torch.randn((1, sam.model.io_channels, T), generator=g).to(dev, dtype)
    t = torch.full((1,), 0.7, device=dev)
    kw = {"cross_attn_cond": torch.randn((1, 257, 768), generator=g).to(dev, dtype),
          "global_cond": torch.randn((1, 768), generator=g).to(dev, dtype),
          "local_add_cond": torch.zeros((1, sam.model.io_channels + 1, T), device=dev, dtype=dtype),
          "padding_mask": torch.ones((1, T), dtype=torch.bool, device=dev), "use_checkpointing": False}
    with torch.no_grad():
        y0 = dit(x, t, **kw).float()
        from stable_audio_3.models.lora import load_lora_checkpoint
        sd, cfg = load_lora_checkpoint(el.at_root(args.lora))
        attach(dit, cfg)
        dit.load_state_dict(sd, strict=False)
        from stable_audio_3.models.lora import set_lora_strength
        set_lora_strength(dit, args.strength)
        y1 = dit(x, t, **kw).float()
    from torch.nn.utils import parametrize
    with torch.no_grad():
        for _n, mod in list(dit.named_modules()):
            if getattr(getattr(mod, "parametrizations", None), "weight", None) is not None:
                merged = mod.weight.detach().clone()
                parametrize.remove_parametrizations(mod, "weight", leave_parametrized=False)
                mod.weight.copy_(merged)
        y2 = dit(x, t, **kw).float()
    import stable_audio_3.models.dit as dit_mod
    res = {"lora_effect_max_abs": float((y1 - y0).abs().max()), "merge_vs_parametrized_max_abs":
           float((y2 - y1).abs().max()), "output_scale": float(y0.abs().mean()), "dtype": str(dtype),
           "has_lora_after_merge": bool(dit_mod.has_lora(dit))}
    print(json.dumps(res, indent=1))
    return 0 if res["merge_vs_parametrized_max_abs"] <= 1e-2 * max(res["output_scale"], 1e-6) else 1


def has(argv, flag):
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def lora_path(args):
    return None if args.lora.lower() in ("none", "base") else args.lora


def tag_of(args):
    if args.tag:
        return args.tag
    if lora_path(args) is None:
        return "base"
    p = el.at_root(args.lora)
    return "%s_%s" % (os.path.basename(os.path.dirname(p)), os.path.splitext(os.path.basename(p))[0].replace(
        "lora_", ""))


def score(args, rest):
    tag = tag_of(args)
    argv = list(rest)
    if not has(argv, "--manifest") and not has(argv, "--refs"):
        if not os.path.isfile(el.at_root(HELDOUT_MANIFEST)) and not args.full_refs:
            raise SystemExit("%s is missing; build it with scripts/bench/vocab_refs.py --heldout, or pass "
                             "--full-refs to score against data/refs" % HELDOUT_MANIFEST)
        if not args.full_refs:
            argv += ["--manifest", HELDOUT_MANIFEST]
    if not has(argv, "--gen-dir"):
        argv += ["--gen-dir", os.path.join("data", "_gate", "vocab_lora", tag)]
    if not has(argv, "--out"):
        argv += ["--out", os.path.join(ROOT, "scripts", "train", "out_vocab_lora_%s.json" % tag)]
    if not has(argv, "--listen"):
        argv += ["--listen", os.path.join("data", "_gate", "listen", "vocab_lora_%s.wav" % tag)]
    if lora_path(args) is not None:
        check_leaks(lora_path(args), argv)
    install_hook(lora_path(args), args.strength)
    import vocab_eval
    print("vocab_eval %s" % " ".join(argv), flush=True)
    return vocab_eval.main(argv)


def gate(args, rest):
    tag = tag_of(args)
    argv = list(rest)
    if not has(argv, "--steps"):
        argv += ["--steps", "3"]
    if not has(argv, "--tag"):
        argv += ["--tag", "_lora_%s" % tag]
    if not has(argv, "--out"):
        argv += ["--out", os.path.join("scripts", "train", "out_gate_lora_%s.json" % tag)]
    install_hook(lora_path(args), args.strength)
    os.chdir(ROOT)
    import gate_quality
    sys.argv = ["gate_quality.py"] + argv
    print("gate_quality %s" % " ".join(argv), flush=True)
    gate_quality.main()
    return 0


def add_train_args(ap):
    ap.add_argument("--cache", default=el.OUT)
    ap.add_argument("--caption", default="sarge", choices=list(el.STYLES))
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--targets", nargs="+", default=["self", "cross", "ff"], choices=sorted(TARGETS))
    ap.add_argument("--layers", default=None)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--t-sampler", default="package", choices=["package", "production"])
    ap.add_argument("--t-jitter", type=float, default=0.3)
    ap.add_argument("--prod-steps", type=int, default=3)
    ap.add_argument("--p-one-shot", type=float, default=0.0)
    ap.add_argument("--cfg-dropout", type=float, default=0.1)
    ap.add_argument("--silence-ext", type=float, default=4.0)
    ap.add_argument("--mask-loss-weight", type=float, default=0.2)
    ap.add_argument("--inpaint", default=INPAINT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--max-bad", type=int, default=20)
    ap.add_argument("--resume", action="store_true")


def parse(argv):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description="LoRA on Stable Audio 3 small-sfx from cached latents, plain PyTorch, "
                                             "plus hooks that merge a LoRA into StableAudio3Generator for scoring.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_train_args(sub.add_parser("train"))
    for name in ("score", "gate", "check"):
        p = sub.add_parser(name)
        p.add_argument("--lora", required=True)
        p.add_argument("--strength", type=float, default=1.0)
        p.add_argument("--tag", default=None)
        if name == "score":
            p.add_argument("--full-refs", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "train" and not args.name and not args.out:
        args.name = "%s_%s_r%d_s%d" % (os.path.basename(os.path.normpath(args.cache)), args.caption, args.rank,
                                       args.seed)
    return args, rest


def main(argv=None):
    args, rest = parse(argv)
    if args.cmd == "train":
        if rest:
            raise SystemExit("train takes no arguments after --")
        train(args)
        return 0
    if args.cmd == "check":
        if lora_path(args) is None:
            raise SystemExit("check needs a LoRA file")
        return check(args)
    if args.cmd == "score":
        return score(args, rest)
    return gate(args, rest)


if __name__ == "__main__":
    raise SystemExit(main())
