import _bootstrap

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import threading
import time

import psutil

np = ProtocolError = connect = None

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
VARIANT_GUNS = [("shoot", "pistol", "normal", "default"), ("shoot", "rifle", "loud", "indoor_small"),
                ("reload", "rifle", "normal", "tunnel"), ("empty_click", "pistol", "soft", "default"),
                ("equip", "rifle", "normal", "outdoor_open"), ("shoot", "pistol", "loud", "indoor_large")]
VARIANT_REST = [("break", "window", "normal", "default"), ("bullet impact", "glass", "loud", "indoor_small"),
                ("footstep", "gravel", "soft", "default"), ("footstep", "stone", "normal", "tunnel"),
                ("footstep", "water", "normal", "default"), ("splash", "water", "loud", "outdoor_open"),
                ("footstep", "wood", "normal", "indoor_large"), ("punch", "body", "normal", "default")]
STAGE = re.compile(r"^(\[woosh\] )?\[startup\] ([\d.]+) s (.*?)(?: \(([\d.]+) s\))?$")
PRE_ROLL_S = 3.0
WATCH = ("System", "MsMpEng.exe", "SearchIndexer.exe", "MemCompression")
SAVER = None


def late_imports():
    global np, ProtocolError, connect
    import numpy as np
    from sarge.sidecar.client import ProtocolError, connect


def smi():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,clocks.sm,pstate,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        used, total, sm, ps, util = (x.strip() for x in r.stdout.strip().splitlines()[0].split(","))
        return {"used_mib": int(used), "total_mib": int(total), "sm_mhz": int(sm), "pstate": ps, "util": int(util)}
    except Exception as e:
        return {"error": str(e)}


def disk_letters():
    cmd = "Get-Partition | Where-Object DriveLetter | ForEach-Object { '{0}|{1}' -f $_.DiskNumber, $_.DriveLetter }"
    out = {}
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
        for line in r.stdout.splitlines():
            if "|" in line:
                n, d = line.strip().split("|")
                out.setdefault("PhysicalDrive" + n, []).append(d + ":")
    except Exception:
        pass
    return {k: "+".join(sorted(v)) for k, v in out.items()}


class Saver:
    def __init__(self, path):
        self.path = path
        self.index = []
        self.names = set()
        os.makedirs(path, exist_ok=True)

    def save(self, tag, et, mat, inten, ctx, variant, y, body, ms):
        slug = "_".join(str(x).replace(" ", "-") for x in (et, mat, inten, ctx))
        name = "%s__%s__v%d" % (tag, slug, variant)
        k = 2
        while name in self.names:
            name = "%s__%s__v%d__%d" % (tag, slug, variant, k)
            k += 1
        self.names.add(name)
        y = np.ascontiguousarray(y, dtype=np.float32)
        np.save(os.path.join(self.path, name + ".npy"), y)
        self.index.append({"file": name + ".npy", "pass": tag, "key": "%s|%s|%s|%s" % (et, mat, inten, ctx),
                           "variant": int(variant), "served": body.get("served"),
                           "sample_rate": int(body.get("sample_rate", 0)), "channels": int(body.get("channels", 1)),
                           "shape": list(y.shape), "sha1": hashlib.sha1(y.tobytes()).hexdigest(), "ms": round(ms, 2)})

    def close(self, extra=None):
        with open(os.path.join(self.path, "index.json"), "w", encoding="utf-8") as fh:
            json.dump({"clips": self.index, **(extra or {})}, fh, indent=1)


def settle(quiet_s=15.0, limit=0.05, max_s=900.0, every=1.0):
    procs = []
    for pid, name in watchlist().items():
        if name == "MsMpEng.exe":
            try:
                procs.append(psutil.Process(pid))
            except psutil.Error:
                pass

    def cpu():
        s = 0.0
        for p in procs:
            try:
                c = p.cpu_times()
                s += c.user + c.system
            except psutil.Error:
                pass
        return s

    if not procs:
        return {"skipped": "MsMpEng not found"}
    t0 = time.time()
    c0 = last = cpu()
    tl = time.time()
    quiet_since = t0
    rates = []
    while time.time() - t0 < max_s:
        time.sleep(every)
        c, now = cpu(), time.time()
        rate = (c - last) / max(1e-6, now - tl)
        rates.append(round(rate, 3))
        last, tl = c, now
        if rate >= limit:
            quiet_since = now
        elif now - quiet_since >= quiet_s:
            return {"seconds": round(now - t0, 1), "msmpeng_cpu_s": round(c - c0, 2), "quiet": True,
                    "rates": rates}
    return {"seconds": round(time.time() - t0, 1), "msmpeng_cpu_s": round(cpu() - c0, 2), "quiet": False,
            "rates": rates}


def affinity(procs):
    out = {}
    for pid, role in procs.items():
        try:
            out[str(pid)] = [role, sorted(psutil.Process(pid).cpu_affinity())]
        except (psutil.Error, AttributeError):
            pass
    return out


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


def watchlist():
    out = {}
    for p in psutil.process_iter(["name"]):
        if p.info["name"] in WATCH:
            out[p.pid] = p.info["name"]
    try:
        out[psutil.win_service_get("SysMain").pid()] = "SysMain"
    except Exception:
        pass
    return out


class Sampler(threading.Thread):
    def __init__(self, every=0.5, disks=None, watch=None):
        super().__init__(daemon=True)
        self.every, self.rows, self.stop = every, [], threading.Event()
        self.t0 = time.time()
        self.disks = disks or {}
        self.root = None
        self.procs = {}
        self.last = {}
        self.trace = []
        self.watch = {}
        self.foreign = {}
        self.ticks = 0
        for pid, name in (watch or {}).items():
            try:
                self.watch[pid] = (psutil.Process(pid), name)
            except psutil.Error:
                pass

    def _refresh(self):
        try:
            root = psutil.Process(self.root)
            found = [root] + root.children(recursive=True)
        except psutil.Error:
            return
        for p in found:
            if p.pid in self.procs:
                continue
            try:
                cl = " ".join(p.cmdline())
            except psutil.Error:
                continue
            self.procs[p.pid] = (p, "woosh" if "woosh_server" in cl else "sidecar")

    def _probe(self, t, s):
        row = {"t": round(t - self.t0, 3), "vram": s.get("used_mib"), "gpu": s.get("util")}
        try:
            row["disk"] = {self.disks.get(k, k): [v.read_bytes, v.write_bytes, v.read_count, v.read_time]
                           for k, v in psutil.disk_io_counters(perdisk=True).items()}
            ct = psutil.cpu_times()
            row["cpu_all"] = ct.user + ct.system
        except Exception:
            pass
        if self.root:
            self._refresh()
            for pid, (p, role) in self.procs.items():
                try:
                    c = p.cpu_times()
                    io = p.io_counters()
                    self.last[pid] = (role, c.user + c.system, io.read_bytes, io.other_bytes)
                except psutil.Error:
                    pass
            agg = {}
            for role, cpu, rb, ob in self.last.values():
                a = agg.setdefault(role, [0.0, 0, 0])
                a[0] += cpu
                a[1] += rb
                a[2] += ob
            row["proc"] = {k: [round(v[0], 3), v[1], v[2]] for k, v in agg.items()}
        others = {}
        for pid, (p, name) in self.watch.items():
            try:
                c = p.cpu_times()
                io = p.io_counters()
            except psutil.Error:
                continue
            o = others.setdefault(name, [0.0, 0, 0])
            o[0] += c.user + c.system
            o[1] += io.read_bytes
            o[2] += io.read_count
        row["others"] = {k: [round(v[0], 3), v[1], v[2]] for k, v in others.items()}
        self.ticks += 1
        if self.ticks % 4 == 1:
            row["foreign"] = self._foreign()
        return row

    def _foreign(self):
        me, parent = os.getpid(), os.getppid()
        out = {}
        for p in psutil.process_iter(["name", "ppid"]):
            if "python" not in (p.info["name"] or "").lower() or p.pid in self.procs or p.pid in (me, parent) or \
                    p.info["ppid"] == me:
                continue
            try:
                if p.pid not in self.foreign:
                    self.foreign[p.pid] = " ".join(p.cmdline())[:200]
                c = p.cpu_times()
                out[str(p.pid)] = round(c.user + c.system, 2)
            except psutil.Error:
                continue
        return out

    def run(self):
        while not self.stop.is_set():
            t = time.time()
            s = smi()
            if "used_mib" in s:
                self.rows.append((t - self.t0, s["used_mib"]))
            self.trace.append(self._probe(t, s))
            self.stop.wait(max(0.05, self.every - (time.time() - t)))

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


def stages(txt):
    out = []
    for line in txt.splitlines():
        m = STAGE.match(line.strip())
        if m:
            out.append([float(m.group(2)), float(m.group(4)) if m.group(4) else None,
                        "woosh" if m.group(1) else "sidecar", m.group(3)])
    return out


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
    if SAVER is not None:
        SAVER.save(tag, et, mat, inten, ctx, int(body.get("variant_id", -1)), y, body, ms)
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
    ap.add_argument("--save-dir", default=None,
                    help="save every served clip as .npy (float32 PCM as received) plus index.json")
    ap.add_argument("--evict", action="store_true",
                    help="drop SARGE's files from the file cache first (evict_cache.py) for a cold start")
    ap.add_argument("--settle", type=float, default=15.0,
                    help="after --evict, wait until Microsoft Defender (MsMpEng) has used under 0.05 of a core for "
                         "this many seconds, so its post-eviction rescans do not land in the measured startup (0 = off)")
    ap.add_argument("--sidecar-root", default=ROOT,
                    help="run scripts/run_sidecar.py from this checkout (e.g. a HEAD export) with this repo's venvs "
                         "and compile caches")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args(argv)
    global SAVER
    os.makedirs(OUT, exist_ok=True)
    for p in woosh_by_port(a.woosh_port):
        raise SystemExit("a woosh server is already running on port %d (pid %d)" % (a.woosh_port, p))
    log_path = os.path.join(OUT, "sidecar_%s.log" % a.tag)
    side_root = os.path.abspath(a.sidecar_root)
    cmd = [sys.executable, "-u", PIN, os.path.join(side_root, "scripts", "run_sidecar.py")] + ARGS + [
        "--port", str(a.port), "--woosh-port", str(a.woosh_port),
        "--cache-dir", os.path.join(ROOT, ".cache", "sfx_bake")] + list(a.extra)
    env = None
    if os.path.normcase(side_root) != os.path.normcase(ROOT):
        tv = importlib.metadata.version("torch").split("+")[0]
        env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=os.path.join(ROOT, ".cache", "inductor-" + tv),
                   TRITON_CACHE_DIR=os.path.join(ROOT, ".cache", "triton-" + tv))
        cmd += ["--woosh-python", os.path.join(ROOT, ".venv_woosh", "Scripts", "python.exe")]
    res = {"tag": a.tag, "cmd": cmd, "sidecar_root": side_root, "gpu_before": smi(),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if a.save_dir:
        SAVER = Saver(a.save_dir)
    if a.evict:
        import evict_cache
        res["evict"] = evict_cache.evict(evict_cache.default_roots() + ([side_root] if env is not None else []))
        print("evicted %d files, %.2f GB, %d failed, %.1f s" % (res["evict"]["files"], res["evict"]["gb"],
                                                                 res["evict"]["failed"], res["evict"]["seconds"]),
              flush=True)
        if a.settle > 0:
            res["settle"] = settle(quiet_s=a.settle)
            print("defender settle %s" % {k: v for k, v in res["settle"].items() if k != "rates"}, flush=True)
    try:
        vm = psutil.virtual_memory()
        res["ram_before"] = {"available_gb": round(vm.available / 1e9, 2), "total_gb": round(vm.total / 1e9, 2)}
    except Exception:
        pass
    res["disks"] = disk_letters()
    watch = watchlist()
    res["watch"] = {str(k): v for k, v in watch.items()}
    sampler = Sampler(disks=res["disks"], watch=watch)
    log = open(log_path, "w", encoding="utf-8")
    t0 = time.time() + PRE_ROLL_S
    sampler.t0 = t0
    sampler.start()
    time.sleep(max(0.0, t0 - time.time()))
    t0 = time.time()
    srv = subprocess.Popen(cmd, cwd=side_root, env=env, stdout=log, stderr=subprocess.STDOUT)
    sampler.root = srv.pid
    try:
        res["launch_offset_s"] = psutil.Process(srv.pid).create_time() - t0
    except psutil.Error:
        res["launch_offset_s"] = None
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
        late_imports()
        procs = tree(srv.pid)
        res["pids"] = {str(p): r for p, r in procs.items()}
        res["affinity"] = affinity(procs)
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
        res["stats_before_variants"] = c2.stats()
        t_var = time.time() - t0
        for side, keys in (("guns", VARIANT_GUNS), ("rest", VARIANT_REST)):
            for et, mat, inten, ctx in keys:
                fire(c2, side, et, mat, inten, ctx, rows, "variants")
                c2.prewarm([{"event_type": et, "material": mat, "intensity": inten, "context": ctx}])
                fire(c2, side, et, mat, inten, ctx, rows, "variants")
                fire(c2, side, et, mat, inten, ctx, rows, "variants")
        res["peak_used_mib_variants"] = sampler.peak(t_var)
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
        res["trace"] = sampler.trace
        res["foreign_python"] = {str(k): v for k, v in sampler.foreign.items()}
        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                res["stages"] = stages(fh.read())
        except OSError:
            res["stages"] = []
        res["rows"] = rows
        if SAVER is not None:
            SAVER.close({"tag": a.tag, "startup": res.get("startup"), "time": res.get("time")})
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
