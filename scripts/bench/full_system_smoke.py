import _bootstrap

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time

import psutil

BENCH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BENCH))
PIN = os.path.join(os.path.dirname(ROOT), "sarge-research", "modal", "runpinned.py")
FULL = ["--generator", "hybrid", "--realtime", "--stereo", "--amp", "--detailed-prompts", "--learn",
        "--prompts", "template:caption+tt", "--woosh-upgrade", "flow", "--woosh-upgrade-solver", "midpoint:16",
        "--woosh-upgrade-best", "4", "--woosh-upgrade-vram-save", "share,encoder", "--scene", "--world",
        "--writer", "llama", "--writer-style", "canon", "--route", "meaning"]
BALL = {"shape": "sphere", "ext": [0.6, 0.6, 0.6], "m": 1.0, "dyn": 1, "pm": "psfx_bouncy", "bnc": 0.8,
        "name": "AutoCaptureBall_0"}
METAL = {"shape": "box", "ext": [10, 0.2, 10], "dyn": 0, "tag": "metal", "name": "Surface_metal"}
PHYS = {"a": BALL, "b": METAL, "vn": 7.5, "vt": 0.3, "n": [0, 1, 0], "j": 7.5, "dist": 6.0}
SCENE = {"src": "physics", "who": "AutoCaptureBall_0", "obj": "AutoCaptureBall_0", "act": "bounce", "room": "open"}
EVENTS = [("footstep", "gravel", "normal", None, None), ("footstep", "gravel", "normal", None, None),
          ("shoot", "rifle", "normal", None, None), ("shoot", "rifle", "normal", None, None),
          ("impact", "metal", "hard", PHYS, SCENE), ("door creak", "wood", "normal", None, None)]


class MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_commit_gb():
    st = MemoryStatus()
    st.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return None
    return st.ullAvailPageFile / 2 ** 30


def smi():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def stop_tree(proc):
    try:
        root = psutil.Process(proc.pid)
        kids = root.children(recursive=True)
    except psutil.Error:
        return []
    for p in [root] + kids:
        try:
            p.kill()
        except psutil.Error:
            pass
    gone, alive = psutil.wait_procs([root] + kids, timeout=20)
    return [p.pid for p in alive]


def main(argv=None):
    ap = argparse.ArgumentParser(description="start the full-system sidecar line once on spare ports, fire a few "
                                             "events with the trace on, watch the background feed, then stop it")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--woosh-port", type=int, default=8773)
    ap.add_argument("--writer-port", type=int, default=11541)
    ap.add_argument("--cache-dir", default=r"G:\sarge\graph\smoke\cache")
    ap.add_argument("--writer-store", default=r"G:\sarge\graph\smoke\writer")
    ap.add_argument("--watch", type=float, default=150.0, help="seconds after the events to watch the feed")
    ap.add_argument("--min-commit-gb", type=float, default=16.0,
                    help="refuse to start when less than this much commit is free, so the system never runs out")
    ap.add_argument("--out", default=r"G:\sarge\graph\smoke\smoke.json")
    a = ap.parse_args(argv)
    free = free_commit_gb()
    if free is not None and free < a.min_commit_gb:
        raise SystemExit("only %.1f GB of commit free, %.1f needed; close something large first" % (
            free, a.min_commit_gb))
    os.makedirs(a.cache_dir, exist_ok=True)
    os.makedirs(a.writer_store, exist_ok=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    log_path = os.path.splitext(a.out)[0] + "_sidecar.log"
    cmd = [sys.executable, "-u", PIN, os.path.join(ROOT, "scripts", "run_sidecar.py")] + FULL + [
        "--port", str(a.port), "--woosh-port", str(a.woosh_port), "--writer-port", str(a.writer_port),
        "--cache-dir", a.cache_dir, "--writer-store", a.writer_store]
    res = {"cmd": cmd, "free_commit_gb_before": free, "vram_before_mib": smi(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    log = open(log_path, "w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    lowest = free
    try:
        ready = None
        while time.time() - t0 < 900 and proc.poll() is None:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                txt = fh.read()
            if "sidecar READY" in txt:
                ready = time.time() - t0
                break
            f = free_commit_gb()
            lowest = f if lowest is None or (f is not None and f < lowest) else lowest
            time.sleep(0.5)
        res["ready_s"] = ready
        res["exit_before_ready"] = proc.poll()
        print("READY %s s" % ready, flush=True)
        if ready is None:
            raise SystemExit("the sidecar never became READY, see %s" % log_path)
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            res["ready_line"] = next((x.strip() for x in fh if "sidecar READY" in x), None)
        res["vram_ready_mib"] = smi()
        from sarge.sidecar.client import connect
        c = connect(port=a.port, timeout=180.0, trace=True)
        res["welcome"] = c.welcome
        rows = []
        for et, mat, inten, phys, scene in EVENTS:
            t = time.perf_counter()
            body, s = c.event(et, mat, inten, phys=phys, scene=scene)
            rows.append({"key": "%s|%s|%s" % (et, mat, inten), "ms": round((time.perf_counter() - t) * 1e3, 1),
                         "served": body.get("served"), "samples": int(len(s)), "trace": body.get("trace")})
            print(json.dumps(rows[-1]), flush=True)
        res["events"] = rows
        seq, kinds, feed = 0, {}, []
        t_watch = time.time()
        while time.time() - t_watch < a.watch:
            st = c.stats(feed=seq)
            for item in st.get("feed", []):
                kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
                feed.append(item)
            seq = st.get("feed_seq", seq)
            res["upgrade"] = st.get("upgrade")
            res["writer"] = st.get("writer")
            f = free_commit_gb()
            lowest = f if lowest is None or (f is not None and f < lowest) else lowest
            if kinds.get("flow_swap") and kinds.get("writer"):
                break
            time.sleep(2.0)
        res["feed_kinds"] = kinds
        res["feed"] = feed[:80]
        res["after"] = []
        for et, mat, inten, phys, scene in EVENTS[:1] + EVENTS[2:3] + EVENTS[4:]:
            body, _s = c.event(et, mat, inten, phys=phys, scene=scene)
            res["after"].append({"key": "%s|%s|%s" % (et, mat, inten), "trace": body.get("trace"),
                                 "prompt_rev": body.get("prompt_rev")})
        c.close()
        res["vram_end_mib"] = smi()
    finally:
        res["lowest_free_commit_gb"] = lowest
        res["left_running"] = stop_tree(proc) if proc.poll() is None else []
        log.close()
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=str)
        print(json.dumps({k: res.get(k) for k in ("ready_s", "feed_kinds", "lowest_free_commit_gb", "left_running",
                                                 "vram_ready_mib")}, default=str), flush=True)
        print("saved -> %s" % a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
