import _bootstrap

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import psutil

from sarge.sidecar.client import ProtocolError, connect

BENCH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BENCH))
PIN = os.path.join(os.path.dirname(ROOT), "sarge-research", "modal", "runpinned.py")
OUT = os.path.join(ROOT, "data", "_gate", "opt", "_e2e")
MEDIUM_FIN = os.path.join(ROOT, "data", "_gate", "opt", "medium", "_work", "fin", "m_final")
WOOSH_FIN = os.path.join(ROOT, "data", "_gate", "opt", "woosh", "_work", "fin", "w_final")
ARGS = ["--generator", "hybrid", "--realtime", "--stereo", "--amp", "--detailed-prompts"]
GUNS = [("shoot", "pistol"), ("shoot", "rifle"), ("reload", "rifle"), ("empty_click", "pistol"), ("equip", "rifle")]
REST = [("break", "window"), ("bullet impact", "glass"), ("footstep", "gravel"), ("footstep", "stone"),
        ("footstep", "water"), ("splash", "water"), ("footstep", "wood"), ("punch", "body")]
INTENSITIES = ("normal", "soft", "loud")
SECOND = [("shoot", "pistol", "normal", "indoor_small"), ("impact", "metal", "normal", "default"),
          ("shoot", "rifle", "normal", "tunnel"), ("footstep", "metal", "normal", "default")]
IDLE = [(0.0, "guns", ("shoot", "rifle", "soft", "indoor_small")), (1.0, "guns", ("shoot", "rifle", "loud", "indoor_small")),
        (3.0, "guns", ("reload", "rifle", "normal", "indoor_small")), (8.0, "guns", ("equip", "rifle", "normal", "indoor_small")),
        (3.0, "guns", ("shoot", "pistol", "soft", "tunnel")), (8.0, "guns", ("shoot", "pistol", "loud", "tunnel")),
        (0.0, "rest", ("footstep", "gravel", "normal", "indoor_small")),
        (3.0, "rest", ("footstep", "stone", "normal", "indoor_small")),
        (8.0, "rest", ("footstep", "wood", "normal", "indoor_small")),
        (3.0, "rest", ("break", "window", "normal", "tunnel")), (8.0, "rest", ("splash", "water", "normal", "tunnel"))]
SR = {"guns": 44100, "rest": 48000}


def smi():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,clocks.sm,pstate",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        used, total, sm, ps = (x.strip() for x in r.stdout.strip().splitlines()[0].split(","))
        return {"used_mib": int(used), "total_mib": int(total), "sm_mhz": int(sm), "pstate": ps}
    except Exception as e:
        return {"error": str(e)}


def proc_gpu(pids):
    pids = [int(p) for p in pids if p]
    if not pids:
        return {}
    cmd = ("$c = Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage','\\GPU Process Memory(*)\\Shared Usage' "
           "-EA SilentlyContinue; $c.CounterSamples | ForEach-Object { '{0}|{1}' -f $_.Path, $_.CookedValue }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
    except Exception as e:
        return {"error": str(e)}
    out = {str(p): {"dedicated_mib": 0.0, "shared_mib": 0.0} for p in pids}
    for line in r.stdout.splitlines():
        if "|" not in line:
            continue
        path, val = line.rsplit("|", 1)
        low = path.lower()
        for p in pids:
            if "(pid_%d_" % p in low:
                k = "dedicated_mib" if low.endswith("dedicated usage") else "shared_mib"
                out[str(p)][k] += float(val) / 2 ** 20
    return out


class Sampler(threading.Thread):
    def __init__(self, every=0.5):
        super().__init__(daemon=True)
        self.every, self.rows, self.stop = every, [], threading.Event()
        self.t0 = time.time()

    def run(self):
        while not self.stop.is_set():
            s = smi()
            if "used_mib" in s:
                self.rows.append((time.time() - self.t0, s["used_mib"]))
            self.stop.wait(self.every)

    def peak(self, a=0.0, b=1e18):
        xs = [u for t, u in self.rows if a <= t <= b]
        return max(xs) if xs else None


def tree(pid):
    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
    except psutil.Error:
        return {}
    out = {}
    for k in procs:
        try:
            cl = " ".join(k.cmdline())
        except psutil.Error:
            continue
        if "python" in k.name().lower():
            out[k.pid] = "woosh" if "woosh_server" in cl else "sidecar"
    return out


def woosh_by_port(port):
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
        except psutil.Error:
            continue
        if "woosh_server" in cl and ("--port %d" % port) in cl and "python" in (p.info["name"] or "").lower():
            out.append(p.pid)
    return out


def alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def by_role(pg, procs):
    out = {}
    for p, role in procs.items():
        g = pg.get(str(p)) or {}
        r = out.setdefault(role, {"dedicated_mib": 0.0, "shared_mib": 0.0, "pids": []})
        r["pids"].append(p)
        r["dedicated_mib"] += g.get("dedicated_mib", 0.0)
        r["shared_mib"] += g.get("shared_mib", 0.0)
    return out


def ref_for(side, et, mat):
    if side == "guns":
        p = os.path.join(MEDIUM_FIN, "%s-%s_00.npy" % (et, mat))
    else:
        p = os.path.join(WOOSH_FIN, "%s-%s_00.npy" % (et.replace(" ", "_"), mat))
    return np.load(p) if os.path.isfile(p) else None


def fire(c, side, et, mat, inten, ctx, rows, tag):
    t = time.perf_counter()
    try:
        body, s = c.event(et, mat, inten, ctx)
    except ProtocolError as e:
        row = {"pass": tag, "side": side, "key": "%s|%s|%s|%s" % (et, mat, inten, ctx),
               "ms": (time.perf_counter() - t) * 1000.0, "error": str(e)[:300], "channels": 0, "shape_ok": False,
               "sr_ok": False, "served": "error", "max_abs_vs_bench": None}
        rows.append(row)
        print(json.dumps(row), flush=True)
        return row
    ms = (time.perf_counter() - t) * 1000.0
    y = np.asarray(s, dtype=np.float32)
    ch = int(body.get("channels", 1))
    d = None
    if inten == "normal" and ctx == "default" and int(body.get("variant_id", -1)) == 0:
        ref = ref_for(side, et, mat)
        if ref is not None:
            ref = ref.reshape(-1, ch) if ref.ndim == 1 and ch > 1 else ref
            d = float(np.max(np.abs(ref.astype(np.float64) - y))) if ref.shape == y.shape else "shape %s vs %s" % (
                ref.shape, y.shape)
    row = {"pass": tag, "side": side, "key": "%s|%s|%s|%s" % (et, mat, inten, ctx), "ms": ms,
           "served": body.get("served"), "variant": int(body.get("variant_id", -1)), "sr": int(body["sample_rate"]),
           "sr_ok": int(body["sample_rate"]) == SR[side], "channels": ch, "shape_ok": y.ndim == 2 and y.shape[1] == ch,
           "duration_s": float(body["duration_s"]), "peak": float(np.max(np.abs(y))) if y.size else 0.0,
           "max_abs_vs_bench": d}
    rows.append(row)
    print(json.dumps(row), flush=True)
    return row


def med(xs):
    return float(np.median(xs)) if len(xs) else None


def summary(rows):
    first = [r for r in rows if r["pass"] == "first"]
    rep = [r for r in rows if r["pass"] == "repeat"]
    conn2 = [r for r in rows if r["pass"] == "conn2"]
    per_key = {}
    for r in first:
        et, mat, inten, _ctx = r["key"].split("|")
        per_key.setdefault("%s|%s" % (et, mat), {"side": r["side"]})[inten] = round(r["ms"], 1)
    out = {"very_first_ms": first[0]["ms"] if first else None, "very_first_key": first[0]["key"] if first else None}
    for side in ("guns", "rest"):
        f = [r["ms"] for r in first[1:] if r["side"] == side]
        out[side] = {"first_hit_median_ms": med(f), "first_hit_p90_ms": float(np.percentile(f, 90)) if f else None,
                     "first_hit_max_ms": max(f) if f else None, "n": len(f),
                     "repeat_median_ms": med([r["ms"] for r in rep if r["side"] == side]),
                     "conn2_first_ms": [round(r["ms"], 1) for r in conn2 if r["side"] == side],
                     "after_idle": [[r["idle_s"], round(r["ms"], 1), r.get("sm_mhz_before"), r.get("pstate_before")]
                                    for r in rows if r["pass"] == "idle" and r["side"] == side]}
    out["per_key_first_ms"] = per_key
    out["errors"] = sorted({"%s %s" % (r["key"], r["error"][:120]) for r in rows if r.get("error")})
    out["all_two_channels"] = all(r["channels"] == 2 and r["shape_ok"] for r in rows)
    out["all_sample_rates_ok"] = all(r["sr_ok"] for r in rows)
    out["served"] = sorted({"%s:%s" % (r["pass"], r["served"]) for r in rows})
    vs = [r["max_abs_vs_bench"] for r in rows if r["max_abs_vs_bench"] is not None]
    out["vs_bench"] = {"n": len(vs), "max_abs": max(v for v in vs if not isinstance(v, str)) if any(
        not isinstance(v, str) for v in vs) else None, "shape_mismatch": [v for v in vs if isinstance(v, str)]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="hybrid sidecar end to end: startup, first hits, repeats, vram, kill")
    ap.add_argument("--tag", default="hybrid")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--woosh-port", type=int, default=8771)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    for p in woosh_by_port(a.woosh_port):
        raise SystemExit("a woosh server is already running on port %d (pid %d)" % (a.woosh_port, p))
    log_path = os.path.join(OUT, "sidecar_%s.log" % a.tag)
    cmd = [sys.executable, "-u", PIN, os.path.join(ROOT, "scripts", "run_sidecar.py")] + ARGS + [
        "--port", str(a.port), "--woosh-port", str(a.woosh_port),
        "--cache-dir", os.path.join(ROOT, ".cache", "sfx_bake")] + list(a.extra)
    res = {"tag": a.tag, "cmd": cmd, "gpu_before": smi(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    sampler = Sampler()
    log = open(log_path, "w", encoding="utf-8")
    t0 = time.time()
    sampler.t0 = t0
    sampler.start()
    srv = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    woosh_up = ready = None
    procs = {}
    rows = []
    try:
        while time.time() - t0 < 900:
            if srv.poll() is not None:
                break
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                txt = fh.read()
            if woosh_up is None and " up in " in txt and "woosh" in txt:
                woosh_up = time.time() - t0
                procs = tree(srv.pid)
                res["gpu_woosh_up"] = smi()
            if "sidecar READY" in txt:
                ready = time.time() - t0
                break
            time.sleep(0.25)
        res["startup"] = {"woosh_up_s": woosh_up, "ready_s": ready, "ready_after_woosh_s": (ready - woosh_up) if (
            ready and woosh_up) else None, "peak_used_mib": sampler.peak()}
        print("woosh up %s s, READY %s s" % (woosh_up, ready), flush=True)
        if ready is None:
            raise SystemExit("sidecar never became READY, see %s" % log_path)
        procs = tree(srv.pid)
        res["pids"] = {str(p): r for p, r in procs.items()}
        time.sleep(2.0)
        res["gpu_ready"] = smi()
        res["proc_gpu_ready"] = by_role(proc_gpu(list(procs)), procs)
        sampler.every = 2.0
        c = connect(port=a.port, timeout=120.0)
        res["welcome"] = c.welcome
        t_req = time.time() - t0
        res["gpu_before_first"] = smi()
        n = max(len(GUNS), len(REST))
        for i in range(n):
            for side, keys in (("guns", GUNS), ("rest", REST)):
                if i < len(keys):
                    et, mat = keys[i]
                    for inten in INTENSITIES:
                        fire(c, side, et, mat, inten, "default", rows, "first")
        for side, keys in (("guns", GUNS), ("rest", REST)):
            for et, mat in keys:
                for inten in INTENSITIES:
                    fire(c, side, et, mat, inten, "default", rows, "repeat")
        c2 = connect(port=a.port, timeout=120.0)
        from sarge.amp import family_for
        for et, mat, inten, ctx in SECOND:
            fire(c2, "guns" if family_for(et) in ("blast", "handling") else "rest", et, mat, inten, ctx, rows, "conn2")
        for gap, side, (et, mat, inten, ctx) in IDLE:
            time.sleep(gap)
            pre = smi()
            r = fire(c2, side, et, mat, inten, ctx, rows, "idle")
            r["idle_s"], r["sm_mhz_before"], r["pstate_before"] = gap, pre.get("sm_mhz"), pre.get("pstate")
        res["stats"] = c2.stats()
        c2.close()
        c.close()
        res["gpu_after"] = smi()
        res["proc_gpu_after"] = by_role(proc_gpu(list(procs)), procs)
        res["peak_used_mib_requests"] = sampler.peak(t_req)
        res["summary"] = summary(rows)
        t_kill = time.perf_counter()
        srv.kill()
        srv.wait(timeout=30)
        wp = [p for p, r in procs.items() if r == "woosh"]
        deadline = time.perf_counter() + 15.0
        while time.perf_counter() < deadline and any(alive(p) for p in procs):
            time.sleep(0.1)
        left = {str(p): r for p, r in procs.items() if alive(p)}
        res["kill"] = {"how": "TerminateProcess on the launched python (what Unity Process.Kill does)",
                       "woosh_pids": wp, "left_after_15_s": left,
                       "woosh_orphaned": any(alive(p) for p in wp),
                       "all_gone_after_s": None if left else time.perf_counter() - t_kill}
        print("kill: left running %s" % left, flush=True)
    finally:
        if srv.poll() is None:
            srv.kill()
        for p in set(woosh_by_port(a.woosh_port)) | set(procs):
            try:
                q = psutil.Process(p)
                if "python" in q.name().lower() and any(s in " ".join(q.cmdline()) for s in (
                        "run_sidecar", "woosh_server")):
                    q.kill()
            except psutil.Error:
                pass
        sampler.stop.set()
        log.close()
        time.sleep(1.0)
        res["left_running"] = woosh_by_port(a.woosh_port)
        res["gpu_end"] = smi()
        res["vram_timeline"] = [[round(t, 1), u] for t, u in sampler.rows]
        res["rows"] = rows
        path = os.path.join(OUT, "e2e_%s.json" % a.tag)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=str)
        print("saved -> %s" % path, flush=True)
    s = res.get("summary", {})
    print(json.dumps({k: res.get(k) for k in ("startup", "gpu_before", "gpu_ready", "gpu_after", "proc_gpu_ready",
                                               "kill")}, default=str), flush=True)
    print(json.dumps({k: v for k, v in s.items() if k != "per_key_first_ms"}, default=str), flush=True)
    return 0 if ready is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
