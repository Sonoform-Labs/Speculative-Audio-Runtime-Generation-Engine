import os
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)

import _bootstrap

import argparse
import hashlib
import json
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(BENCH))
PIN = os.path.join(os.path.dirname(ROOT), "sarge-research", "modal", "runpinned.py")
DEMO = [("shoot", "pistol"), ("shoot", "rifle"), ("reload", "rifle"), ("empty_click", "pistol"), ("equip", "rifle"),
        ("break", "window"), ("bullet impact", "glass"), ("bullet impact", "metal"), ("footstep", "gravel"),
        ("footstep", "stone"), ("footstep", "water"), ("splash", "water"), ("footstep", "wood"), ("footstep", "metal"),
        ("footstep", "grass"), ("punch", "body"), ("land", "stone"), ("jump", "wood"), ("glass bottle hit", "stone"),
        ("take off", "iron boots")]
UNSEEN = [("grappling_hook", "rope"), ("lightsaber", "plasma"), ("door_creak", "wood"), ("dog_bark", "dog"),
          ("explosion", "grenade"), ("campfire", ""), ("camera_snapshot", ""), ("magic_missile", ""), ("ice_crack", "ice"),
          ("peacock_call", ""), ("keyboard_click", ""), ("pistol", ""), ("cannon", ""), ("laser_zap", ""),
          ("rocket_launch", ""), ("gunshot", ""), ("engine_hum", "car"), ("thunder", ""),
          ("Footstep_Concrete", "concrete"), ("cloth_rustle", "cloth")]
EXTRA = [["footstep", "stone", "normal", "indoor_small"], ["shoot", "rifle", "normal", "tunnel"],
         ["footstep", "gravel", "soft", "default"], ["shoot", "pistol", "hard", "default"],
         ["splash", "water", "normal", "default", "", "a heavy rock dropped into a deep pond"],
         ["shoot", "pistol", "normal", "default", "", "a suppressed pistol shot, close"]]
KEYS = [[e, m, "normal", "default"] for e, m in DEMO + UNSEEN] + EXTRA


def busy():
    import psutil
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
        except psutil.Error:
            continue
        if "python" in (p.info["name"] or "").lower() and any(s in cl for s in ("run_sidecar", "woosh_server")):
            return cl
    return ""


def main(argv=None):
    ap = argparse.ArgumentParser(description="render a fixed request list through the real hybrid sidecar with the "
                                             "given flags and save every clip as .npy plus index.json, for "
                                             "generic_page.py; each key is [event, material, intensity, context, "
                                             "class, prompt]")
    ap.add_argument("tag")
    ap.add_argument("out")
    ap.add_argument("--keys", default=None, help="JSON list of keys; default: 20 demo keys, 20 unseen names, 6 extras")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--woosh-port", type=int, default=8772)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="flags passed to run_sidecar.py")
    a = ap.parse_args(argv)
    held = busy()
    if held:
        raise SystemExit("a sidecar or woosh worker is running, wait for it: %s" % held[:200])
    from sarge.sidecar.client import ProtocolError, connect
    keys = json.load(open(a.keys, encoding="utf-8")) if a.keys else KEYS
    os.makedirs(a.out, exist_ok=True)
    cmd = [sys.executable, "-u", PIN, os.path.join(ROOT, "scripts", "run_sidecar.py"), "--generator", "hybrid",
           "--realtime", "--stereo", "--no-cache", "--no-deepen", "--port", str(a.port), "--woosh-port",
           str(a.woosh_port)] + list(a.extra)
    log_path = os.path.join(a.out, "sidecar.log")
    log = open(log_path, "w", encoding="utf-8")
    t0 = time.time()
    srv = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        while time.time() - t0 < 900:
            if srv.poll() is not None:
                raise SystemExit("sidecar exited, see %s" % log_path)
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                if "sidecar READY" in fh.read():
                    break
            time.sleep(0.5)
        c = connect(port=a.port, timeout=120.0)
        for k in keys:
            et, mat, inten, ctx = k[:4]
            cls = k[4] if len(k) > 4 else ""
            prompt = k[5] if len(k) > 5 else ""
            t = time.perf_counter()
            try:
                body, s = c.event(et, mat, inten, ctx, cls=cls, prompt=prompt)
            except ProtocolError as e:
                rows.append({"key": k, "error": str(e)})
                continue
            ms = (time.perf_counter() - t) * 1000.0
            y = np.ascontiguousarray(s, dtype=np.float32)
            name = "%s__%s" % (a.tag, hashlib.md5(json.dumps(k).encode()).hexdigest()[:10])
            np.save(os.path.join(a.out, name + ".npy"), y)
            rows.append({"key": k, "file": name + ".npy", "sr": int(body["sample_rate"]), "ms": round(ms, 1),
                         "served": body.get("served"), "shape": list(y.shape),
                         "sha1": hashlib.sha1(y.tobytes()).hexdigest()})
            print(json.dumps(rows[-1]), flush=True)
        c.close()
    finally:
        srv.kill()
        srv.wait(timeout=30)
        log.close()
        time.sleep(2.0)
        import psutil
        for p in psutil.process_iter(["cmdline", "name"]):
            try:
                cl = " ".join(p.info["cmdline"] or [])
            except psutil.Error:
                continue
            if "woosh_server" in cl and ("--port %d" % a.woosh_port) in cl and "python" in (p.info["name"] or "").lower():
                p.kill()
        with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8") as fh:
            json.dump({"tag": a.tag, "args": a.extra, "rows": rows}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
