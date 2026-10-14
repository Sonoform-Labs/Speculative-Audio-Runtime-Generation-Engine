import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

OUT = ROOT / "data" / "_gate" / "opt" / "woosh_buckets"
SERVER = ROOT / "scripts" / "woosh_server.py"
KEYS = [("break", "window"), ("bullet impact", "glass"), ("footstep", "gravel"), ("footstep", "stone"),
        ("footstep", "wood"), ("footstep", "metal"), ("footstep", "water"), ("splash", "water"), ("punch", "body")]
INTENSITIES = ("normal", "loud")
CONTEXT = "default"
SEEDS = 6
FRAMES = (101, 151, 201, 301, 501)
FLOOR_DB = -55.0
TAIL_S = 0.05
EDGE_S = 0.02
MAX_TRUNCATED = 0.05


def server(path):
    spec = importlib.util.spec_from_file_location("woosh_server", str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules["woosh_server"] = m
    spec.loader.exec_module(m)
    return m


def family(event):
    from sarge.amp import family_for
    return family_for(event) or "other"


def name(et, mat, inten, v):
    return "%s-%s-%s_%02d" % (et.replace(" ", "_"), mat, inten, v)


def med(xs):
    return float(np.median(np.asarray(xs, dtype=float))) if len(xs) else None


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if len(xs) else None


def dump(obj, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def load_json(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def tail(x, sr, floor_db=FLOOR_DB, tail_s=TAIL_S, edge_s=EDGE_S):
    a = np.asarray(x, dtype=np.float32)
    m = a if a.ndim == 1 else a.mean(axis=1)
    n = m.shape[0]
    win = max(1, int(0.01 * sr))
    if n <= win or not np.any(m):
        return {"truncated": False, "alive_end_s": 0.0, "trim_at_cut": False, "content_truncated": False,
                "edge_peak": False}
    c = np.empty(n + 1)
    c[0] = 0.0
    np.cumsum(m.astype(np.float64) ** 2, out=c[1:])
    env = np.sqrt(np.maximum(c[win:] - c[:-win], 0.0) / win)
    alive = np.flatnonzero(env >= float(env.max()) * 10.0 ** (floor_db / 20.0))
    end = int(alive[-1]) + win
    edge = n - int(edge_s * sr)
    body = alive[alive + win <= edge]
    return {"truncated": bool(end > n - int(tail_s * sr)), "alive_end_s": end / sr,
            "trim_at_cut": bool(end + int(0.03 * sr) >= n),
            "content_truncated": bool(len(body) and int(body[-1]) + win > edge - int(tail_s * sr)),
            "edge_peak": bool(int(np.argmax(env)) + win > edge)}


def choose(fractions, limit=MAX_TRUNCATED):
    for f in sorted(fractions):
        if fractions[f] <= limit:
            return f
    return None


def gpu_mb(torch):
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / 2 ** 20


def clip_rows(ws, gen, frames_of, reps, save_dir=None, ref_dir=None):
    import torch
    from sarge.generators.stable_audio import _natural
    from sarge.prompts import build
    from sarge.types import CacheKey, seed_for
    rows, saved = {}, {}
    for rep in range(1 + reps):
        for et, mat in KEYS:
            for inten in INTENSITIES:
                key = CacheKey(et, mat, inten, CONTEXT)
                p = build(et, mat, inten, CONTEXT)
                fr = frames_of(et)
                for v in range(SEEDS):
                    n = name(et, mat, inten, v)
                    miss0 = gen.cache.misses
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    x = gen(p, seed_for(key, v), frames=fr) if fr is not None else gen(p, seed_for(key, v))
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t) * 1000.0
                    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
                    miss = bool(gen.cache.misses > miss0)
                    if rep:
                        r = rows[n]
                        r["ms"].append(ms)
                        r["miss"].append(miss)
                        r["rep_max_abs"] = max(r["rep_max_abs"], float(np.max(np.abs(x - saved[n]))))
                        continue
                    t2 = time.perf_counter()
                    ws.finish(x, et, inten, True, True)
                    fin = (time.perf_counter() - t2) * 1000.0
                    nat = _natural(x[:, 0] if x.ndim == 2 else x, ws.SR).shape[0] / ws.SR
                    r = {"key": "%s|%s" % (et, mat), "intensity": inten, "seed": v, "family": family(et),
                         "frames": fr if fr is not None else gen.frames, "raw_s": x.shape[0] / ws.SR, "natural_s": nat,
                         "at_cut": bool(abs(nat - x.shape[0] / ws.SR) < 1e-9), "finish_ms": fin, "ms": [ms],
                         "miss": [miss], "rep_max_abs": 0.0, **tail(x, ws.SR)}
                    if ref_dir is not None:
                        ref = np.load(Path(ref_dir) / str(r["frames"]) / (n + ".npy"))
                        r["vs_sweep_max_abs"] = float(np.max(np.abs(ref.astype(np.float64) - x))) if (
                                ref.shape == x.shape) else "shape %s vs %s" % (ref.shape, x.shape)
                    if save_dir is not None:
                        np.save(Path(save_dir) / (n + ".npy"), x)
                    saved[n] = x
                    rows[n] = r
        print("pass %d done" % rep, flush=True)
    return list(rows.values())


def timing(rows):
    ms = [m for r in rows for m in r["ms"]]
    hit = [m for r in rows for m, s in zip(r["ms"], r["miss"]) if not s]
    miss = [m for r in rows for m, s in zip(r["ms"], r["miss"]) if s]
    fin = [r["finish_ms"] for r in rows]
    return {"median_ms": med(ms), "p90_ms": pct(ms, 90), "hit_median_ms": med(hit), "miss_median_ms": med(miss),
            "finish_median_ms": med(fin), "served_median_ms": med([r["ms"][0] + r["finish_ms"] for r in rows]),
            "calls": len(ms)}


def by_family(rows):
    out = {}
    for fam in sorted({r["family"] for r in rows}):
        rs = [r for r in rows if r["family"] == fam]
        out[fam] = {"n": len(rs), "truncated": sum(r["truncated"] for r in rs),
                    "truncated_frac": sum(r["truncated"] for r in rs) / len(rs),
                    "at_cut": sum(r["at_cut"] for r in rs), "trim_at_cut": sum(r["trim_at_cut"] for r in rs),
                    "content_truncated": sum(r["content_truncated"] for r in rs),
                    "edge_peak": sum(r["edge_peak"] for r in rs),
                    "natural_median_s": med([r["natural_s"] for r in rs]),
                    "natural_p90_s": pct([r["natural_s"] for r in rs], 90),
                    "natural_max_s": max(r["natural_s"] for r in rs), **timing(rs),
                    "keys": {k: {"truncated": sum(r["truncated"] for r in rs if r["key"] == k),
                                 "n": sum(1 for r in rs if r["key"] == k),
                                 "natural_median_s": med([r["natural_s"] for r in rs if r["key"] == k]),
                                 "natural_max_s": max(r["natural_s"] for r in rs if r["key"] == k)}
                             for k in sorted({r["key"] for r in rs})}}
    return out


def cmd_sweep(a):
    ws = server(a.server)
    import torch
    opts = dict(ws.OPTIONS, **ws.ADOPTED["dflow"])
    d = OUT / a.tag / str(a.frames)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    gen = ws.make_gen("dflow", a.frames, 4, "dopri5", **opts)
    load_s = time.perf_counter() - t0
    mb0 = gpu_mb(torch)
    t1 = time.perf_counter()
    gen("warmup, a soft thud", 0)
    warm_s = time.perf_counter() - t1
    mb1 = gpu_mb(torch)
    print("dflow %d frames: load %.1f s, warm-up %.2f s, reserved %.0f -> %.0f MB, graph error %s" % (
        a.frames, load_s, warm_s, mb0, mb1, gen.stats["graph_error"]), flush=True)
    rows = clip_rows(ws, gen, lambda et: None, a.reps, save_dir=d)
    fam = by_family(rows)
    res = {"frames": a.frames, "server": str(a.server), "options": opts, "load_s": load_s, "warm_s": warm_s,
           "reserved_mb_loaded": mb0, "reserved_mb_ready": mb1, "peak_reserved_mb": torch.cuda.max_memory_reserved() /
           2 ** 20, "stats": {k: v for k, v in gen.stats.items() if k != "nfe"}, "timing": timing(rows),
           "families": fam, "rep_max_abs": max(r["rep_max_abs"] for r in rows), "rows": rows,
           "torch": torch.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    dump(res, OUT / a.tag / ("%d.json" % a.frames))
    for f, v in fam.items():
        print("%4d %-7s truncated %2d/%2d  natural median %.2f p90 %.2f s  %.1f ms (hit %.1f, finish %.1f)" % (
            a.frames, f, v["truncated"], v["n"], v["natural_median_s"], v["natural_p90_s"], v["median_ms"],
            v["hit_median_ms"], v["finish_median_ms"]), flush=True)
    print("%d frames: median %.1f ms, rep max abs %g" % (a.frames, res["timing"]["median_ms"], res["rep_max_abs"]),
          flush=True)
    return 0


def cmd_choose(a):
    ws = server(a.server)
    sweeps = {f: load_json(OUT / a.tag / ("%d.json" % f)) for f in a.frames}
    fams = sorted({k for s in sweeps.values() for k in s["families"]})
    out = {"limit": MAX_TRUNCATED, "families": {}}
    for fam in fams:
        fs = {f: s["families"][fam] for f, s in sweeps.items()}
        frac = {f: v["truncated_frac"] for f, v in fs.items()}
        pick = choose(frac)
        keep = ws.FRAMES.get(fam, ws.FRAMES["other"])
        out["families"][fam] = {"frames": pick, "kept": keep if pick is None else None, "truncated_frac": frac,
                                "keys": sorted(fs[min(fs)]["keys"]),
                                "content_truncated": {f: v["content_truncated"] for f, v in fs.items()},
                                "edge_peak": {f: v["edge_peak"] for f, v in fs.items()},
                                "n": fs[min(fs)]["n"],
                                "median_ms": {f: v["median_ms"] for f, v in fs.items()},
                                "served_median_ms": {f: v["served_median_ms"] for f, v in fs.items()},
                                "natural_median_s": {f: v["natural_median_s"] for f, v in fs.items()}}
        print("%-7s %s -> %s" % (fam, "  ".join("%d:%.3f" % kv for kv in sorted(frac.items())),
                                 pick if pick is not None else "none at or under %.0f%%, keep %d" % (
                                     100 * MAX_TRUNCATED, keep)), flush=True)
    dump(out, OUT / ("choice_%s.json" % a.tag))
    return 0


def cmd_buckets(a):
    ws = server(a.server)
    import torch
    opts = dict(ws.OPTIONS, **ws.ADOPTED["dflow"])
    frames = ws.parse_frames(a.frames)
    t0 = time.perf_counter()
    gen = ws.make_gen("dflow", frames["other"], 4, "dopri5", **opts)
    load_s = time.perf_counter() - t0
    mb0 = gpu_mb(torch)
    warm = ws.prepare(gen, frames)
    mb1 = gpu_mb(torch)
    print("load %.1f s, buckets %s, reserved %.0f -> %.0f MB" % (load_s, json.dumps(warm), mb0, mb1), flush=True)
    rows = clip_rows(ws, gen, lambda et: ws.frames_for(et, frames), a.reps, ref_dir=OUT / a.ref)
    vs = [r["vs_sweep_max_abs"] for r in rows]
    res = {"frames_table": frames, "server": str(a.server), "options": opts, "load_s": load_s, "buckets": warm,
           "reserved_mb_loaded": mb0, "reserved_mb_ready": mb1,
           "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2 ** 20,
           "stats": {k: v for k, v in gen.stats.items() if k != "nfe"}, "timing": timing(rows),
           "families": by_family(rows), "rep_max_abs": max(r["rep_max_abs"] for r in rows),
           "vs_sweep": {"ref": a.ref, "n": len(vs), "equal": sum(1 for v in vs if v == 0.0),
                        "max_abs": max((v for v in vs if not isinstance(v, str)), default=None),
                        "shape_mismatch": [v for v in vs if isinstance(v, str)]},
           "rows": rows, "torch": torch.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    dump(res, OUT / ("buckets_%s.json" % a.tag))
    for f, v in res["families"].items():
        print("%-7s truncated %2d/%2d  natural median %.2f s  %.1f ms (hit %.1f, finish %.1f)" % (
            f, v["truncated"], v["n"], v["natural_median_s"], v["median_ms"], v["hit_median_ms"],
            v["finish_median_ms"]), flush=True)
    print("vs sweep %s, rep max abs %s" % (json.dumps(res["vs_sweep"]), res["rep_max_abs"]), flush=True)
    return 0


def parse(argv):
    ap = argparse.ArgumentParser(description="Woosh DFlow latent length per sound family: sweep, choose, check buckets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sweep")
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--tag", default="sweep")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--server", default=str(SERVER))
    p = sub.add_parser("choose")
    p.add_argument("--frames", type=int, nargs="+", default=list(FRAMES))
    p.add_argument("--tag", default="sweep")
    p.add_argument("--server", default=str(SERVER))
    p = sub.add_parser("buckets")
    p.add_argument("--frames", default=None)
    p.add_argument("--tag", default="new")
    p.add_argument("--ref", default="sweep")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--server", default=str(SERVER))
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(sys.argv[1:] if argv is None else list(argv))
    return {"sweep": cmd_sweep, "choose": cmd_choose, "buckets": cmd_buckets}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
