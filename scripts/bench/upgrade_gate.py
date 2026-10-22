import _bootstrap

import argparse
import json
import os
import subprocess
import sys
import time

import e2e_hybrid as e2e

OUT = os.path.join(e2e.ROOT, "data", "_gate", "untried", "upgrade")
BOUNDS = {"startup_peak_mib": 10752, "ready_mib": 9830, "both_rendering_mib": 11571}
FILL_REST = [("footstep", "gravel"), ("footstep", "stone"), ("footstep", "wood"), ("break", "window"),
             ("bullet impact", "glass"), ("splash", "water"), ("punch", "body"), ("dog_bark", "dog"),
             ("door_creak", "wood"), ("car_horn", "car")]
FILL_GUNS = [("shoot", "pistol"), ("shoot", "rifle"), ("reload", "rifle")]
STRESS_CTX = ("indoor_small", "tunnel", "indoor_large", "outdoor_open")
GAPS = (0.3, 1.2, 0.5, 2.0, 0.8, 1.5)


def shot(c, side, et, mat, inten, ctx, rows, tag, t0):
    t = time.perf_counter()
    try:
        body, s = c.event(et, mat, inten, ctx)
    except e2e.ProtocolError as e:
        row = {"pass": tag, "side": side, "key": "%s|%s|%s|%s" % (et, mat, inten, ctx), "t": time.time() - t0,
               "ms": (time.perf_counter() - t) * 1000.0, "served": "error", "error": str(e)[:200], "rev": 0,
               "sha": None, "variant": -1}
        rows.append(row)
        print(json.dumps(row), flush=True)
        return row
    ms = (time.perf_counter() - t) * 1000.0
    y = e2e.np.asarray(s, dtype=e2e.np.float32)
    row = {"pass": tag, "side": side, "key": "%s|%s|%s|%s" % (et, mat, inten, ctx), "ms": ms, "t": time.time() - t0,
           "served": body.get("served"), "variant": int(body.get("variant_id", -1)),
           "rev": int(body.get("prompt_rev", 0) or 0), "sha": e2e.hashlib.sha1(y.tobytes()).hexdigest()[:16]}
    rows.append(row)
    print(json.dumps(row), flush=True)
    return row


def commit_free_gb():
    import ctypes
    from ctypes import wintypes

    class PI(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                    ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                    ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                    ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                    ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                    ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD),
                    ("ThreadCount", wintypes.DWORD)]

    p = PI()
    p.cb = ctypes.sizeof(PI)
    if not ctypes.WinDLL("psapi").GetPerformanceInfo(ctypes.byref(p), p.cb):
        return None
    return (p.CommitLimit - p.CommitTotal) * p.PageSize / 2 ** 30


def events(path):
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def med(xs):
    return float(e2e.np.median(xs)) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="gate V and first-hit latency with --woosh-upgrade flow: startup peak, "
                                             "READY, peak while the upgrade renders beside served renders, and "
                                             "served misses before and during the upgrade")
    ap.add_argument("--tag", default="upgrade_flow")
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--woosh-port", type=int, default=8773)
    ap.add_argument("--stress-s", type=float, default=90.0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args(argv)
    e2e.late_imports()
    os.makedirs(a.out, exist_ok=True)
    for p in e2e.woosh_by_port(a.woosh_port):
        raise SystemExit("a woosh server is already running on port %d (pid %d)" % (a.woosh_port, p))
    log_path = os.path.join(a.out, "sidecar_%s.log" % a.tag)
    ev_path = os.path.join(a.out, "events_%s.jsonl" % a.tag)
    if os.path.exists(ev_path):
        os.remove(ev_path)
    cmd = [sys.executable, "-u", e2e.PIN, os.path.join(e2e.ROOT, "scripts", "run_sidecar.py")] + e2e.ARGS + [
        "--port", str(a.port), "--woosh-port", str(a.woosh_port), "--learn", "--event-log", ev_path,
        "--cache-dir", os.path.join(e2e.ROOT, ".cache", "sfx_bake")] + list(a.extra)
    res = {"tag": a.tag, "cmd": cmd, "bounds": BOUNDS, "gpu_before": e2e.smi(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "fail_hook": os.environ.get("SARGE_UPGRADE_FAIL") or None}
    sampler = e2e.Sampler(every=0.1)
    log = open(log_path, "w", encoding="utf-8")
    t0 = time.time()
    sampler.t0 = t0
    sampler.start()
    srv = subprocess.Popen(cmd, cwd=e2e.ROOT, stdout=log, stderr=subprocess.STDOUT)
    procs, rows, ready = {}, [], None
    try:
        while time.time() - t0 < 900 and srv.poll() is None:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                if "sidecar READY" in fh.read():
                    ready = time.time() - t0
                    break
            time.sleep(0.25)
        if ready is None:
            raise SystemExit("sidecar never became READY, see %s" % log_path)
        res["startup"] = {"ready_s": ready, "peak_used_mib": sampler.peak(0, ready)}
        procs = e2e.tree(srv.pid)
        wpids = [p for p, r in procs.items() if r == "woosh"]
        res["woosh_died_s"] = None
        res["commit_free_gb_ready"] = commit_free_gb()
        res["commit_free_gb_min"] = res["commit_free_gb_ready"]

        def watch():
            while not sampler.stop.is_set():
                cf = commit_free_gb()
                if cf is not None and (res["commit_free_gb_min"] is None or cf < res["commit_free_gb_min"]):
                    res["commit_free_gb_min"] = cf
                if wpids and not any(e2e.alive(p) for p in wpids):
                    res["woosh_died_s"] = time.time() - t0
                    return
                time.sleep(0.2)

        e2e.threading.Thread(target=watch, daemon=True).start()
        time.sleep(2.0)
        res["gpu_ready"] = e2e.smi()
        c = e2e.connect(port=a.port, timeout=120.0)
        t_fill = time.time() - t0
        for et, mat in FILL_REST:
            shot(c, "rest", et, mat, "normal", "default", rows, "fill", t0)
        for et, mat in FILL_GUNS:
            shot(c, "guns", et, mat, "normal", "default", rows, "fill", t0)
        for et, mat in FILL_REST[:5]:
            shot(c, "rest", et, mat, "soft", "default", rows, "fill", t0)
        res["peak_fill_mib"] = sampler.peak(t_fill)
        t_wait = time.time()
        state = None
        while "--woosh-upgrade" in a.extra and time.time() - t_wait < 180:
            st = [x for x in events(ev_path) if x.get("type") == "upgrade_state"]
            state = st[-1]["state"] if st else None
            if state in ("ready", "failed"):
                break
            time.sleep(1.0)
        res["upgrade_state"] = state
        res["upgrade_ready_s"] = time.time() - t0
        res["gpu_upgrade_ready"] = e2e.smi()
        res["peak_until_upgrade_ready_mib"] = sampler.peak(ready, time.time() - t0)
        t_st = time.time() - t0
        i = 0
        while time.time() - t0 - t_st < a.stress_s:
            gap = GAPS[i % len(GAPS)]
            time.sleep(gap)
            k = i % 4
            if k == 0:
                et, mat = FILL_GUNS[i // 4 % len(FILL_GUNS)]
                shot(c, "guns", et, mat, ("soft", "loud")[i // 4 % 2], STRESS_CTX[i // 8 % 4], rows, "stress_miss", t0)
            elif k == 1:
                et, mat = FILL_REST[i // 4 % len(FILL_REST)]
                shot(c, "rest", et, mat, ("loud", "soft")[i // 4 % 2], STRESS_CTX[i // 8 % 4], rows, "stress_miss", t0)
            else:
                et, mat = FILL_REST[i % len(FILL_REST)]
                shot(c, "rest", et, mat, "normal", "default", rows, "stress_hit", t0)
            i += 1
        t_end = time.time() - t0
        res["peak_stress_mib"] = sampler.peak(t_st, t_end)
        time.sleep(15.0)
        for et, mat in FILL_REST:
            for _ in range(3):
                shot(c, "rest", et, mat, "normal", "default", rows, "after", t0)
        res["stats"] = c.stats()
        c.close()
        res["gpu_after"] = e2e.smi()
        res["peak_after_ready_mib"] = sampler.peak(ready, time.time() - t0)
    finally:
        if srv.poll() is None:
            srv.kill()
        for p in set(e2e.woosh_by_port(a.woosh_port)) | set(procs):
            try:
                q = e2e.psutil.Process(p)
                if "python" in q.name().lower():
                    q.kill()
            except e2e.psutil.Error:
                pass
        sampler.stop.set()
        log.close()
    ev = events(ev_path)
    swaps = [x for x in ev if x.get("type") == "upgrade_swap"]
    yields = [x for x in ev if x.get("type") == "upgrade_yield"]
    first = {}
    for r in rows:
        first.setdefault(r["key"], r)
    fill = [r for r in rows if r["pass"] == "fill" and r["served"] == "realtime"]
    miss = [r for r in rows if r["pass"] == "stress_miss" and r["served"] == "realtime"]
    after = [r for r in rows if r["pass"] == "after"]
    changed = sum(1 for r in after if r["rev"] >= 1 and first.get(r["key"], {}).get("sha") != r["sha"])
    lags = [x["first_hit_s"] for x in swaps if x.get("first_swap") and x.get("first_hit_s") is not None]
    res["upgrade"] = {"swaps": len(swaps), "takes": sum(x.get("takes", 0) for x in swaps), "yields": len(yields),
                      "flow_ms_median": med([x["ms"] for x in swaps]), "rev_after": sorted({r["rev"] for r in after}),
                      "after_changed_audio": changed, "after_rows": len(after),
                      "errors": len([x for x in ev if x.get("type") == "upgrade_error"]),
                      "failed_keys": len([x for x in ev if x.get("type") == "upgrade_failed"]),
                      "first_hit_to_swap_s_median": med(lags), "first_hit_to_swap_s_max": max(lags, default=None),
                      "first_hit_to_swap_s_p95": float(e2e.np.percentile(lags, 95)) if lags else None,
                      "first_hit_to_swap_keys": len(lags),
                      "served_upgraded": len([x for x in ev if x.get("type") == "event" and x.get("upgraded")]),
                      "summary": (res.get("stats") or {}).get("upgrade")}
    errs = [x for x in ev if x.get("type") in ("upgrade_error", "upgrade_failed")]
    if errs:
        t_err = min(x["t"] for x in errs)
        bad = sorted({x["key"] for x in errs})
        res["upgrade"]["failure"] = {"keys": bad, "records": errs,
                                     "swaps_after_first_error": len([x for x in swaps if x["t"] > t_err
                                                                     and x["key"] not in bad]),
                                     "after_rows_of_failed_keys": [r for r in after if any(
                                         r["key"] == b.split("|scene=")[0] for b in bad)]}
    res["latency"] = {side: {"fill_realtime_median_ms": med([r["ms"] for r in fill if r["side"] == side]),
                             "stress_realtime_median_ms": med([r["ms"] for r in miss if r["side"] == side]),
                             "stress_realtime_max_ms": max([r["ms"] for r in miss if r["side"] == side], default=None),
                             "n_fill": len([r for r in fill if r["side"] == side]),
                             "n_stress": len([r for r in miss if r["side"] == side])} for side in ("guns", "rest")}
    res["stress_hit_median_ms"] = med([r["ms"] for r in rows if r["pass"] == "stress_hit"])
    res["gate_v"] = {"startup_peak": [res.get("startup", {}).get("peak_used_mib"), BOUNDS["startup_peak_mib"]],
                     "ready": [res.get("gpu_ready", {}).get("used_mib"), BOUNDS["ready_mib"]],
                     "both_rendering": [res.get("peak_after_ready_mib"), BOUNDS["both_rendering_mib"]]}
    res["gate_v_pass"] = all(v[0] is not None and v[0] <= v[1] for v in res["gate_v"].values())
    res["rows"] = rows
    res["vram_timeline"] = [[round(t, 2), u] for t, u in sampler.rows]
    with open(os.path.join(a.out, "gate_%s.json" % a.tag), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    print(json.dumps({k: res.get(k) for k in ("startup", "gate_v", "gate_v_pass", "upgrade_state", "upgrade",
                                               "latency", "stress_hit_median_ms", "woosh_died_s",
                                               "commit_free_gb_ready", "commit_free_gb_min")}, indent=1,
                     default=str))
    return 0 if res["gate_v_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
