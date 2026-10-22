import _bootstrap

import argparse
import datetime
import hashlib
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time

import numpy as np
import psutil

BENCH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BENCH))
PIN = os.path.join(os.path.dirname(ROOT), "sarge-research", "modal", "runpinned.py")
OUT = os.path.join(ROOT, "data", "_gate", "ontime")
CACHE = os.path.join(ROOT, ".cache", "sfx_ontime")
BASE = ["--generator", "hybrid", "--realtime", "--stereo", "--amp", "--detailed-prompts", "--learn"]
MOCK = ["--generator", "mock", "--realtime", "--learn"]
OFF_GEN = ["--no-concurrent-gen", "--no-async-cache-write"]
OFF = OFF_GEN + ["--no-predict-fixes"]
ARMS = {"head": OFF, "ref": ["--vram-kb", "2000000", "--target-variants", "2"] + OFF,
        "cg": ["--concurrent-gen", "--no-async-cache-write", "--no-predict-fixes"],
        "cga": ["--concurrent-gen", "--async-cache-write", "--no-predict-fixes"],
        "fix": ["--predict-fixes", "--pool-fix", "--concurrent-gen", "--no-async-cache-write"],
        "pf": ["--predict-fixes", "--pool-fix"] + OFF_GEN,
        "kw50": ["--keep-warm-ms", "50"] + OFF, "kw200": ["--keep-warm-ms", "200"] + OFF,
        "dflt": ["--no-predict-fixes"], "pfx": ["--predict-fixes"], "pfp": ["--predict-fixes", "--pool-fix"],
        "now": [], "upf": ["--prompts", "template:caption+tt", "--woosh-upgrade", "flow"]}
PHASES = {"head": "bac", "ref": "r", "cg": "ba", "cga": "ba", "fix": "ba", "pf": "b", "kw50": "c", "kw200": "c",
          "dflt": "ba", "pfx": "ba", "pfp": "ba", "now": "bdu", "upf": "bdu"}
DRAIN_S, DRAIN_QUIET_S = 300.0, 3.0
FOREIGN = ("run_sidecar", "woosh_server", "e2e_hybrid", "woosh_opt", "medium_opt", "world_render")
GUN_EVENTS = ("shoot", "reload", "empty_click", "equip")
LATE_MS = 20.0
LOADS = {"light30": (30, 2048, 1), "game60": (60, 2048, 3)}
IDLE_CONDS = ("noload", "light30", "game60")
SESSION_LOAD = "game60"
V_START_MIB, V_READY_MIB, V_BOTH_MIB = 10.5 * 1024, 9.6 * 1024, 11.3 * 1024


def child_env(gpu=True):
    env = dict(os.environ)
    for k in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE", "CUDA_VISIBLE_DEVICES"):
        env.pop(k, None)
    env.update(HF_HOME=r"F:\AI\cache\hf", HF_HUB_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               TEMP=r"D:\tmp", TMP=r"D:\tmp", PYTHONUNBUFFERED="1")
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def foreign_jobs():
    me = os.getpid()
    mine = {me}
    try:
        mine |= {p.pid for p in psutil.Process(me).children(recursive=True)}
        mine |= {psutil.Process(me).ppid()}
    except psutil.Error:
        pass
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if p.pid in mine or "python" not in (p.info["name"] or "").lower():
            continue
        cl = " ".join(p.info["cmdline"] or [])
        if any(s in cl for s in FOREIGN):
            out.append([p.pid, cl[:240]])
    return out


def pct(xs, q):
    xs = [x for x in xs if x is not None]
    return round(float(np.percentile(xs, q)), 2) if xs else None


def family(et):
    from sarge.amp import family_for
    return "guns" if family_for(et) in ("blast", "handling") else "rest"


class Smi(threading.Thread):
    def __init__(self, every_ms=100):
        super().__init__(daemon=True)
        self.rows = []
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,memory.used,clocks.sm,pstate,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits", "-lms", str(every_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def run(self):
        for line in self.proc.stdout:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 6:
                continue
            try:
                t = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                used, sm = int(p[1]), int(p[2])
            except ValueError:
                continue
            try:
                pw = float(p[4])
            except ValueError:
                pw = None
            try:
                ut = int(p[5])
            except ValueError:
                ut = None
            self.rows.append((t, used, sm, p[3], pw, ut))

    def stop(self):
        try:
            self.proc.kill()
        except OSError:
            pass

    def window(self, a, b):
        return [r for r in list(self.rows) if a <= r[0] <= b]

    def peak(self, a, b):
        w = self.window(a, b)
        return max(r[1] for r in w) if w else None

    def last(self, t):
        w = [r for r in list(self.rows) if r[0] <= t]
        return w[-1] if w else None

    def summary(self, a, b):
        w = self.window(a, b)
        pw = [r[4] for r in w if r[4] is not None]
        ut = [r[5] for r in w if r[5] is not None]
        ps = {}
        for r in w:
            ps[r[3]] = ps.get(r[3], 0) + 1
        return {"n": len(w), "power_w_mean": round(float(np.mean(pw)), 2) if pw else None,
                "power_w_max": max(pw) if pw else None, "sm_mhz_median": pct([r[2] for r in w], 50),
                "util_mean": round(float(np.mean(ut)), 2) if ut else None, "pstates": ps,
                "vram_peak_mib": max((r[1] for r in w), default=None)}


class Store:
    def __init__(self, path, keep=True):
        self.path = path
        self.keep = keep
        self.saved = {}
        self.index = []
        self.q = queue.Queue()
        os.makedirs(path, exist_ok=True)
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()

    def _writer(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            it, y, body = job
            h = hashlib.sha1(y.tobytes()).hexdigest()
            k = (it["key"], int(it["variant"]), h, tuple(y.shape))
            name = self.saved.get(k)
            if name is None:
                name = "c%04d.npy" % len(self.saved)
                self.saved[k] = name
                self.index.append({"file": name, "key": it["key"], "variant": int(it["variant"]), "sha1": h,
                                   "shape": list(y.shape), "dtype": str(y.dtype),
                                   "sample_rate": int(body.get("sample_rate", 0)),
                                   "channels": int(body.get("channels", 1)), "served": body.get("served")})
                if self.keep:
                    np.save(os.path.join(self.path, name), y)
            it["sha1"], it["file"] = h, name

    def put(self, it, y, body):
        self.q.put((it, y, body))

    def close(self):
        self.q.put(None)
        self.thread.join(timeout=300)
        with open(os.path.join(self.path, "index.json"), "w", encoding="utf-8") as fh:
            json.dump({"clips": self.index}, fh, indent=1)


class Lane(threading.Thread):
    def __init__(self, game, index):
        super().__init__(daemon=True)
        from sarge.sidecar.client import connect
        self.game = game
        self.index = index
        self.q = queue.Queue()
        self.busy = 0
        self.lock = threading.Lock()
        self.client = connect(port=game.port, timeout=180.0)

    def put(self, item):
        with self.lock:
            self.busy += 1
        self.q.put(item)

    def load(self):
        with self.lock:
            return self.busy

    def run(self):
        while True:
            it = self.q.get()
            if it is None:
                return
            it["t_deq"] = time.perf_counter()
            try:
                if it["kind"] == "event":
                    body, s = self.client.event(it["et"], it["mat"], it["inten"], it["ctx"])
                    it["t_reply"] = time.perf_counter()
                    self.game.on_reply(it, body, s)
                elif it["kind"] == "predict":
                    it["ack"] = self.client.predict(it["preds"], it["now"])
                    it["t_reply"] = time.perf_counter()
                elif it["kind"] == "context":
                    it["ack"] = self.client.context(**it["ctx"])
            except Exception as e:
                it["error"] = repr(e)[:300]
            finally:
                with self.lock:
                    self.busy -= 1
                it["done"].set()


class Game:
    def __init__(self, port, store):
        from sarge.sidecar.client import connect
        self.port = port
        self.store = store
        self.lanes = [Lane(self, 0), Lane(self, 1)]
        self.control = Lane(self, 2)
        self.side = connect(port=port, timeout=180.0)
        for ln in self.lanes + [self.control]:
            ln.start()
        self.next = 0
        self.held = set()
        self.records = []
        self.messages = []
        self.phase = None
        self.last_done = time.time()

    def pick(self):
        best = None
        n = len(self.lanes)
        for k in range(n):
            ln = self.lanes[(self.next + k) % n]
            if best is None or ln.load() < best.load():
                best = ln
        self.next = (best.index + 1) % n
        return best

    def event(self, et, mat, inten="normal", ctx="default", cls="", lane=None, tag=None):
        key = "%s|%s|%s|%s" % (et, mat, inten, ctx)
        it = {"kind": "event", "et": et, "mat": mat, "inten": inten, "ctx": ctx, "key": key, "cls": cls,
              "phase": self.phase, "tag": tag, "held_before": key in self.held, "route": family(et),
              "done": threading.Event()}
        ln = self.lanes[lane] if lane is not None else self.pick()
        it["lane"] = ln.index
        it["t_enq"] = time.perf_counter()
        it["wall"] = time.time()
        self.records.append(it)
        ln.put(it)
        return it

    def predict(self, preds, now):
        it = {"kind": "predict", "preds": preds, "now": now, "phase": self.phase, "done": threading.Event(),
              "t_enq": time.perf_counter(), "wall": time.time()}
        self.messages.append(it)
        self.control.put(it)
        return it

    def context(self, ctx):
        it = {"kind": "context", "ctx": ctx, "phase": self.phase, "done": threading.Event(),
              "t_enq": time.perf_counter(), "wall": time.time()}
        self.messages.append(it)
        self.control.put(it)
        return it

    def on_reply(self, it, body, s):
        self.last_done = time.time()
        y = np.ascontiguousarray(np.asarray(s, dtype=np.float32))
        it["served"] = body.get("served")
        it["variant"] = int(body.get("variant_id", -1))
        it["sample_rate"] = int(body.get("sample_rate", 0))
        it["channels"] = int(body.get("channels", 1))
        it["n"] = int(y.shape[0]) if y.ndim else 0
        if body.get("prompt_rev") is not None:
            it["rev"] = int(body.get("prompt_rev") or 0)
        it["ms"] = (it["t_reply"] - it["t_enq"]) * 1000.0
        it["queue_ms"] = (it["t_deq"] - it["t_enq"]) * 1000.0
        if y.size:
            self.store.put(it, y, body)
        if it["served"] in ("cached", "realtime") and y.size:
            self.held.add(it["key"])

    def wait(self, items, timeout=180.0):
        end = time.time() + timeout
        for it in items:
            it["done"].wait(max(0.0, end - time.time()))

    def drain(self, timeout=180.0):
        self.wait([r for r in self.records + self.messages], timeout)

    def stats(self):
        return self.side.stats()

    def quiesce(self, stable=0.5, timeout=90.0):
        t0 = time.time()
        sig, since = None, time.time()
        while time.time() - t0 < timeout:
            st = self.stats()
            s = st.get("stats", {})
            cur = (s.get("spec_gen"), s.get("demand_gen"), s.get("realtime"), round(float(s.get("gpu_ms", 0.0)), 3),
                   st.get("cached_keys"))
            if cur != sig:
                sig, since = cur, time.time()
            elif time.time() - since >= stable:
                return round(time.time() - t0, 3)
            time.sleep(0.1)
        return None

    def prewarm(self, keys):
        return self.side.prewarm(keys)

    def close(self):
        for ln in self.lanes + [self.control]:
            ln.q.put(None)
        for ln in self.lanes + [self.control]:
            ln.join(timeout=10)
            try:
                ln.client.close()
            except Exception:
                pass
        try:
            self.side.close()
        except Exception:
            pass


class Load:
    def __init__(self, log_path, hz=30, size=2048, iters=1):
        self.log_path = log_path
        self.args = ["--hz", str(hz), "--size", str(size), "--iters", str(iters)]
        self.proc = None
        self.fh = None

    def _count(self):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                return f.read().count("gpuload ready")
        except OSError:
            return 0

    def start(self, timeout=120.0):
        before = self._count()
        self.fh = open(self.log_path, "a", encoding="utf-8")
        cmd = [sys.executable, "-u", PIN, os.path.abspath(__file__), "gpuload", "--parent-pid", str(os.getpid())] +             self.args
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=child_env(), stdout=self.fh, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("gpu load exited early")
            if self._count() > before:
                return round(time.time() - t0, 2)
            time.sleep(0.2)
        raise RuntimeError("gpu load never became ready")

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=20)
            except Exception:
                pass
        self.proc = None
        if self.fh:
            self.fh.close()
            self.fh = None


def gpuload(a):
    import torch
    torch.set_num_threads(1)
    x = torch.randn(a.size, a.size, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("gpuload ready size %d hz %.0f iters %d" % (a.size, a.hz, a.iters), flush=True)
    period = 1.0 / a.hz
    nxt = time.perf_counter()
    frames, busy, t_start = 0, 0.0, time.time()
    while time.time() - t_start < a.seconds:
        t = time.perf_counter()
        for _ in range(a.iters):
            y = x @ x
        torch.cuda.synchronize()
        busy += time.perf_counter() - t
        frames += 1
        if frames % int(a.hz * 5) == 0:
            print(json.dumps({"frames": frames, "gpu_ms_per_frame": round(busy * 1000.0 / frames, 3)}), flush=True)
            if a.parent_pid and not psutil.pid_exists(a.parent_pid):
                break
        nxt += period
        d = nxt - time.perf_counter()
        if d > 0:
            time.sleep(d)
        else:
            nxt = time.perf_counter()
    del y
    return 0


def session(seed=7):
    rng = random.Random(seed)
    acts = []

    def ev(t, cls, et, mat, inten="normal"):
        acts.append((round(t, 4), len(acts), "event", (cls, et, mat, inten)))

    def pr(t, preds):
        acts.append((round(t, 4), len(acts), "predict", preds))

    def p(et, mat, inten, eta, source, conf):
        return {"event_type": et, "material": mat, "intensity": inten, "context": "default", "eta_s": round(eta, 4),
                "confidence": conf, "source": source}

    def weapon(t, w):
        pr(t, [p("shoot", w, "normal", 0.5, "equip", 0.95), p("reload", w, "normal", 3.0, "equip", 0.6),
               p("empty_click", w, "normal", 5.0, "equip", 0.3)])

    def walk(t, plan, name="footstep", cls="footstep"):
        for surf, n in plan:
            for _ in range(n):
                pr(t - 0.25, [p("footstep", surf, "normal", 0.25, "motion", 0.9)])
                ev(t, cls, name, surf)
                t += 0.45
        return t

    def lead_hit(t, cls, et, mat, inten, lead):
        err = rng.gauss(0.0, 0.030)
        pr(t - lead - err, [p(et, mat, inten, lead, "impact", 0.9)])
        ev(t, cls, et, mat, inten)

    def shot(t, w, window=False):
        if window:
            pr(t, [p("break", "window", "normal", 0.0, "aim", 0.6)])
        ev(t, "shoot", "shoot", w)
        if window:
            ev(t, "window", "break", "window")

    acts.append((0.0, -2, "context", {"time_s": 0.0, "equipped": "rifle", "surfaces": ["grass", "stone", "metal", "wood"],
                                      "region": "arena", "gpu_pressure": 0.3}))
    weapon(0.0, "rifle")
    walk(1.0, [("grass", 6), ("stone", 6), ("wood", 6)])
    for t0, tiles in ((2.0, ("metal", "stone")), (3.6, ("grass", "wood"))):
        for k in range(16):
            t = t0 + 3.9 * k
            if t > 58.0:
                break
            lead_hit(t, "ball", "impact", tiles[0] if k < 7 else tiles[1], "hard", rng.uniform(0.10, 0.40))
    for t in (10.0, 10.1, 10.2, 12.0, 12.1, 12.2):
        shot(t, "rifle")
    for t in (14.0, 15.5, 17.0):
        shot(t, "rifle", window=True)
    ev(18.0, "handling", "reload", "rifle")
    ev(20.0, "equip", "equip", "pistol")
    weapon(20.0, "pistol")
    for t in (21.0, 21.6, 22.2, 23.0):
        shot(t, "pistol")
    shot(24.0, "pistol", window=True)
    ev(25.0, "handling", "empty_click", "pistol")
    ev(25.8, "handling", "reload", "pistol")
    for t in (27.0, 27.5):
        shot(t, "pistol")
    for t, mat, inten, lead in ((30.0, "gravel", "normal", 0.37), (30.8, "gravel", "hard", 0.06),
                                (32.0, "glass", "hard", 0.13), (32.6, "glass", "normal", 0.58),
                                (34.0, "gravel", "normal", 0.37), (35.0, "glass", "hard", 0.13),
                                (36.0, "gravel", "hard", 0.06), (37.0, "glass", "normal", 0.58),
                                (38.5, "gravel", "normal", 0.37), (39.5, "glass", "hard", 0.13)):
        lead_hit(t, "bottle", "glass bottle hit", mat, inten, lead)
    walk(41.0, [("stone", 5), ("grass", 5)])
    ev(46.0, "equip", "equip", "rifle")
    weapon(46.0, "rifle")
    for t in (47.0, 47.1, 47.2):
        shot(t, "rifle")
    shot(48.0, "rifle", window=True)
    ev(50.0, "equip", "equip", "iron boots")
    walk(51.0, [("wood", 6), ("metal", 6)], name="iron boot footstep", cls="iron boot footstep")
    acts.sort(key=lambda a: (a[0], a[1]))
    return [(t, kind, payload) for t, _i, kind, payload in acts]


def hol_keys(n=8):
    guns, woosh = [], []
    mats = ("pistol", "rifle")
    for i in range(3 * n):
        guns.append((GUN_EVENTS[i % 4], mats[(i // 4) % 2], "normal", "hol%d" % i))
    pool = (("footstep", "stone"), ("bullet impact", "wood"), ("break", "window"), ("splash", "water"),
            ("punch", "body"), ("glass bottle hit", "gravel"), ("impact", "metal"), ("footstep", "gravel"))
    for i in range(2 * n):
        et, mat = pool[i % len(pool)]
        woosh.append((et, mat, "normal", "holw%d" % i))
    probe = ("footstep", "wood", "normal", "holprobe")
    return probe, guns, woosh


def idle_keys(cond):
    out = []
    i = 0
    for rep in range(2):
        for gap in (1.0, 3.0, 8.0):
            out.append((gap, ("footstep", ("gravel", "stone", "wood")[i % 3], "normal", "idle_%s_%d" % (cond, i))))
            out.append((gap, (("shoot", "reload")[i % 2], ("pistol", "rifle")[(i // 2) % 2], "normal",
                              "idle_%s_%d" % (cond, i))))
            i += 1
    return out


def all_keys():
    keys = []
    for t, kind, payload in session():
        if kind == "event":
            _cls, et, mat, inten = payload
            keys.append((et, mat, inten, "default"))
    probe, guns, woosh = hol_keys()
    keys += [probe] + guns + woosh
    for cond in IDLE_CONDS:
        keys += [k for _g, k in idle_keys(cond)]
        keys += [("footstep", "grass", "normal", "idle_%s_prime" % cond), ("shoot", "pistol", "normal", "idle_%s_prime" % cond)]
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def read_log(path):
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def log_end(rec):
    if rec.get("type") == "event":
        return rec["t_recv"] + (rec.get("await_ms", 0) + rec.get("lock_wait_ms", 0) + rec.get("engine_ms", 0)) / 1e3
    if rec.get("type") in ("deepen", "keep_warm"):
        return rec.get("t")
    return None


def gen_activity(log):
    spans = []
    for r in log:
        if r.get("type") == "event" and r.get("served") == "realtime":
            e = log_end(r)
            spans.append((e - r.get("engine_ms", 0) / 1e3, e, "realtime"))
        elif r.get("type") in ("deepen", "keep_warm"):
            spans.append((r["t"] - r.get("ms", 0) / 1e3, r["t"], r["type"]))
    return sorted(spans)


def match_log(records, log):
    by_key = {}
    for r in log:
        if r.get("type") == "event":
            by_key.setdefault(r["key"], []).append(r)
    for v in by_key.values():
        v.sort(key=lambda r: r["t_recv"])
    used = set()
    for it in sorted(records, key=lambda r: r["wall"]):
        best = None
        for r in by_key.get(it["key"], []):
            if id(r) in used or r["t_recv"] < it["wall"] - 0.05:
                continue
            best = r
            break
        if best is not None:
            used.add(id(best))
            it["log"] = {k: best.get(k) for k in ("served", "variant", "await_ms", "lock_wait_ms", "engine_ms",
                                                   "gen_ms", "encode_ms", "send_ms", "concurrent", "t_recv",
                                                   "upgraded", "rev")}


def clean(it):
    return {k: v for k, v in it.items() if k not in ("done",)}


def run_session(game, rec, smi, name="b"):
    acts = session()
    game.phase = name
    t_wall0 = time.time()
    t0 = time.perf_counter() + 0.5
    slips = []
    for t, kind, payload in acts:
        d = t0 + t - time.perf_counter()
        if d > 0:
            time.sleep(d)
        slips.append((time.perf_counter() - t0 - t) * 1000.0)
        if kind == "event":
            cls, et, mat, inten = payload
            game.event(et, mat, inten, "default", cls=cls)
        elif kind == "predict":
            game.predict(payload, now=t)
        else:
            game.context(payload)
    game.drain()
    t_events_done = time.time()
    q = game.quiesce(stable=1.0)
    t_end = time.time()
    rec[name] = {"t_start": t_wall0, "t_events_done": t_events_done, "t_end": t_end, "quiesce_s": q,
                 "slip_ms_p50": pct(slips, 50), "slip_ms_max": round(max(slips), 2), "stats": game.stats(),
                 "gpu": smi.summary(t_wall0, t_end)}


def run_drain(game, rec, smi, limit=DRAIN_S, quiet=DRAIN_QUIET_S):
    game.phase = "u"
    t0 = time.time()
    poll, since, drained = [], None, None
    while time.time() - t0 < limit:
        up = game.stats().get("upgrade")
        if up is None:
            break
        poll.append([round(time.time() - t0, 2), up.get("queued_takes"), up.get("staged_takes"), up.get("rendered"),
                     up.get("swapped_takes"), up.get("errors"), up.get("skipped_keys")])
        if not up.get("queued_takes") and not up.get("staged_takes"):
            since = since or time.time()
            if time.time() - since >= quiet:
                drained = round(since - t0, 2)
                break
        else:
            since = None
        time.sleep(0.5)
    t1 = time.time()
    keys = []
    for r in game.records:
        if r["kind"] == "event" and r.get("phase") in ("b", "b2") and r["route"] == "rest" and r["key"] not in keys:
            keys.append(r["key"])
    items = []
    for k in keys:
        et, mat, inten, ctx = k.split("|")
        it = game.event(et, mat, inten, ctx, cls="replay", lane=0, tag="replay")
        game.wait([it])
        items.append(it)
    t2 = time.time()
    rec["u"] = {"t_start": t0, "t_drained": t1, "t_end": t2, "drained_s": drained, "limit_s": limit, "poll": poll,
                "replay_keys": len(keys), "stats": game.stats(), "gpu": smi.summary(t0, t1)}


def upgrade_spans(log, t0, t1):
    out = {"swap_takes": 0, "swap_ms": 0.0, "yield_n": 0, "yield_ms": 0.0, "error_n": 0, "error_ms": 0.0}
    for r in log:
        t = r.get("t")
        if t is None or not (t0 <= t <= t1 + 0.5):
            continue
        if r.get("type") == "upgrade_swap":
            ms = [x for x in (r.get("take_ms") or []) if x is not None]
            out["swap_takes"] += len(ms)
            out["swap_ms"] += sum(ms)
        elif r.get("type") == "upgrade_yield":
            out["yield_n"] += 1
            out["yield_ms"] += float(r.get("ms") or 0.0)
        elif r.get("type") in ("upgrade_error", "upgrade_failed"):
            out["error_n"] += 1
            out["error_ms"] += float(r.get("ms") or 0.0)
    out["ms"] = round(out["swap_ms"] + out["yield_ms"] + out["error_ms"], 1)
    return out


def plays(records, phase):
    ev = [r for r in records if r.get("phase") == phase and r["kind"] == "event" and r["route"] == "rest"]
    out = {"woosh_plays": len(ev), "dflow_on_demand": 0, "dflow_take": 0, "flow_take": 0, "unmatched": 0}
    for r in ev:
        lg = r.get("log") or {}
        if not lg:
            out["unmatched"] += 1
        elif lg.get("served") == "realtime":
            out["dflow_on_demand"] += 1
        elif lg.get("upgraded"):
            out["flow_take"] += 1
        else:
            out["dflow_take"] += 1
    return out


def upgrade_lag(log):
    swaps = [r for r in log if r.get("type") == "upgrade_swap"]
    lags = sorted(r["first_hit_s"] for r in swaps if r.get("first_swap") and r.get("first_hit_s") is not None)
    hit = {r["key"] for r in log if r.get("type") == "event" and family(r["key"].split("|")[0]) == "rest"}
    swapped = {r["key"] for r in swaps}
    return {"keys_swapped_after_hit": len(lags), "lag_p50_s": pct(lags, 50), "lag_p95_s": pct(lags, 95),
            "lag_max_s": round(max(lags), 2) if lags else None,
            "swaps_before_hit": sum(1 for r in swaps if r.get("first_swap") and r.get("first_hit_s") is None),
            "keys_played_never_swapped": len(hit - swapped), "swaps": len(swaps),
            "takes_swapped": sum(int(r.get("takes") or 0) for r in swaps),
            "errors": sum(1 for r in log if r.get("type") == "upgrade_error"),
            "failed_keys": sum(1 for r in log if r.get("type") == "upgrade_failed"),
            "yields": sum(1 for r in log if r.get("type") == "upgrade_yield"),
            "state": [r.get("state") for r in log if r.get("type") == "upgrade_state"][-1:]}


def hol_trial(game, smi, kind, gun, probe, rows, i):
    game.quiesce(stable=0.5)
    time.sleep(0.25)
    if kind == "cached":
        touch = game.event(*probe, cls="probe_touch", lane=1, tag="touch")
        game.wait([touch])
        time.sleep(0.15)
    else:
        touch = None
    pre = smi.last(time.time())
    t_a = time.time()
    g = game.event(*gun, cls="hol_gun", lane=0, tag="%s%d" % (kind, i)) if gun else None
    if g is not None:
        time.sleep(0.030)
    w = game.event(*probe, cls="hol_" + kind, lane=1, tag="%s%d" % (kind, i))
    game.wait([x for x in (g, w) if x is not None])
    t_b = time.time() + 0.15
    time.sleep(0.2)
    rows.append({"kind": kind, "i": i, "gun": "|".join(gun) if gun else None, "probe": "|".join(probe),
                 "t_a": t_a, "t_b": t_b, "sm_before": pre[2] if pre else None, "pstate_before": pre[3] if pre else None,
                 "probe_ms": w.get("ms"), "probe_served": w.get("served"), "gun_ms": g.get("ms") if g else None,
                 "gun_served": g.get("served") if g else None,
                 "touch_served": touch.get("served") if touch else None, "touch_ms": touch.get("ms") if touch else None})


def run_hol(game, rec, smi, n=8):
    game.phase = "a"
    probe, guns, woosh = hol_keys(n)
    rows = []
    t_start = time.time()
    first = game.event(*probe, cls="probe_first", lane=1, tag="probe_first")
    game.wait([first])
    for i in range(n):
        hol_trial(game, smi, "cached", guns[3 * i], probe, rows, i)
        hol_trial(game, smi, "fresh", guns[3 * i + 1], woosh[2 * i], rows, i)
        hol_trial(game, smi, "alone", None, woosh[2 * i + 1], rows, i)
    game.drain()
    q = game.quiesce(stable=1.0)
    t_end = time.time()
    for r in rows:
        r["vram_peak_mib"] = smi.peak(r["t_a"], r["t_b"])
    rec["a"] = {"t_start": t_start, "t_end": t_end, "quiesce_s": q, "trials": rows, "stats": game.stats(),
                "gpu": smi.summary(t_start, t_end)}


def run_idle(game, rec, smi, cond, load):
    game.phase = "c_" + cond
    out = {"cond": cond, "load": LOADS.get(cond)}
    t_start = time.time()
    if load is not None:
        out["load_start_s"] = load.start()
        time.sleep(3.0)
    game.quiesce(stable=0.5)
    w0 = time.time()
    time.sleep(12.0)
    w1 = time.time()
    out["idle_window"] = {"t0": w0, "t1": w1, "gpu": smi.summary(w0, w1)}
    prime = [game.event("footstep", "grass", "normal", "idle_%s_prime" % cond, cls="idle_prime", lane=0, tag="prime"),
             game.event("shoot", "pistol", "normal", "idle_%s_prime" % cond, cls="idle_prime", lane=1, tag="prime")]
    game.wait(prime)
    out["prime_ms"] = [p.get("ms") for p in prime]
    trials = []
    lane = 0
    for gap, key in idle_keys(cond):
        game.quiesce(stable=0.5)
        t_q = time.time()
        prev_done = game.last_done
        time.sleep(max(0.0, gap - (t_q - prev_done)))
        pre = smi.last(time.time())
        gw = smi.summary(t_q, time.time())
        it = game.event(*key, cls="idle_" + family(key[0]), lane=lane, tag="gap%.0f" % gap)
        lane = 1 - lane
        game.wait([it])
        trials.append({"gap_s": gap, "key": "|".join(key), "route": family(key[0]), "ms": it.get("ms"),
                       "served": it.get("served"), "t_quiet": t_q, "wall": it["wall"],
                       "since_reply_s": round(it["wall"] - prev_done, 3),
                       "sm_before": pre[2] if pre else None, "pstate_before": pre[3] if pre else None,
                       "power_before": pre[4] if pre else None, "gap_power_w_mean": gw["power_w_mean"],
                       "gap_sm_median": gw["sm_mhz_median"]})
    if load is not None:
        load.stop()
    out["trials"] = trials
    out["t_start"], out["t_end"] = t_start, time.time()
    rec.setdefault("c", []).append(out)


def run_ref(game, rec):
    game.phase = "r"
    keys = all_keys()
    got = {}
    t0 = time.time()
    for k in keys:
        it = game.event(*k, cls="ref", lane=0, tag="ref0")
        game.wait([it])
        vs = {it.get("variant")}
        try:
            game.prewarm([{"event_type": k[0], "material": k[1], "intensity": k[2], "context": k[3]}])
        except Exception as e:
            rec.setdefault("ref_errors", []).append([list(k), repr(e)[:200]])
        for _ in range(4):
            it = game.event(*k, cls="ref", lane=0, tag="ref1")
            game.wait([it])
            vs.add(it.get("variant"))
            if {0, 1} <= vs:
                break
        got["|".join(k)] = sorted(v for v in vs if v is not None)
    rec["r"] = {"keys": len(keys), "variants": got, "seconds": round(time.time() - t0, 1)}


def summarise_b(records, log, t0, t1, phase="b"):
    ev = [r for r in records if r.get("phase") == phase and r["kind"] == "event"]
    classes = sorted({r["cls"] for r in ev})
    out = {}
    for c in classes + ["all"]:
        rs = [r for r in ev if c == "all" or r["cls"] == c]
        ms = [r.get("ms") for r in rs if r.get("ms") is not None]
        fp = [r for r in rs if not r["held_before"]]
        fms = [r.get("ms") for r in fp if r.get("ms") is not None]
        out[c] = {"n": len(rs), "on_demand": sum(1 for r in rs if r.get("served") == "realtime"),
                  "pct_on_demand": round(100.0 * sum(1 for r in rs if r.get("served") == "realtime") / max(1, len(rs)), 1),
                  "p50_ms": pct(ms, 50), "p95_ms": pct(ms, 95), "max_ms": round(max(ms), 2) if ms else None,
                  "late_gt20": sum(1 for x in ms if x > LATE_MS),
                  "first_plays": len(fp), "first_p50_ms": pct(fms, 50), "first_p95_ms": pct(fms, 95),
                  "first_late_gt20": sum(1 for x in fms if x > LATE_MS),
                  "errors": sum(1 for r in rs if r.get("error"))}
    used = {(r["key"], r.get("variant")) for r in ev if r.get("served") in ("cached", "realtime")}
    deepen = [r for r in log if r.get("type") == "deepen" and t0 <= r["t"] <= t1 + 0.5]
    items, wasted = [], []
    for d in deepen:
        per = d.get("ms", 0.0) / max(1, len(d["items"]))
        for (k, v), g in zip(d["items"], d.get("gen_ms") or [None] * len(d["items"])):
            items.append((k, v, per, g, d.get("stat")))
            if (k, v) not in used:
                wasted.append((k, v, per))
    rt = [r for r in log if r.get("type") == "event" and r.get("served") == "realtime" and t0 <= r["t_recv"] <= t1]
    rt_ms = sum(r.get("engine_ms", 0.0) for r in rt)
    dp_ms = sum(x[2] for x in items)
    wall = (t1 - t0) * 1000.0
    out["renders"] = {"realtime_n": len(rt), "realtime_ms": round(rt_ms, 1), "prediction_n": len(items),
                      "prediction_ms": round(dp_ms, 1), "wasted_n": len(wasted),
                      "wasted_ms": round(sum(x[2] for x in wasted), 1),
                      "wasted_keys": sorted({"%s v%s" % (k, v) for k, v, _ in wasted}),
                      "gpu_ms_total": round(rt_ms + dp_ms, 1), "wall_ms": round(wall, 1),
                      "gpu_share": round((rt_ms + dp_ms) / wall, 4) if wall else None,
                      "wasted_share": round(sum(x[2] for x in wasted) / wall, 4) if wall else None,
                      "woosh_gen_ms_p50": pct([r.get("gen_ms") for r in rt if r.get("gen_ms")], 50),
                      "woosh_gen_ms_p95": pct([r.get("gen_ms") for r in rt if r.get("gen_ms")], 95)}
    return out


def summarise_a(trials, records):
    by = {}
    for kind in ("cached", "fresh", "alone"):
        rs = [t for t in trials if t["kind"] == kind]
        by[kind] = {"n": len(rs), "probe_p50_ms": pct([t["probe_ms"] for t in rs], 50),
                    "probe_p95_ms": pct([t["probe_ms"] for t in rs], 95),
                    "probe_max_ms": max((t["probe_ms"] for t in rs if t["probe_ms"] is not None), default=None),
                    "probe_served": sorted({str(t["probe_served"]) for t in rs}),
                    "gun_p50_ms": pct([t["gun_ms"] for t in rs], 50),
                    "probe_gen_ms_p50": pct([t.get("probe_gen_ms") for t in rs], 50),
                    "probe_gen_ms_p95": pct([t.get("probe_gen_ms") for t in rs], 95),
                    "vram_peak_mib": max((t["vram_peak_mib"] for t in rs if t["vram_peak_mib"]), default=None)}
    return by


def run(a):
    arm = a.arm
    tag = "%s_%d" % (arm, a.rep)
    d = os.path.join(OUT, a.set, tag)
    os.makedirs(d, exist_ok=True)
    busy = foreign_jobs()
    if busy:
        print("BUSY %s" % json.dumps(busy), flush=True)
        return 3
    port, wport = a.port, a.woosh_port
    evlog = os.path.join(d, "events.jsonl")
    if os.path.exists(evlog):
        os.remove(evlog)
    side_log = os.path.join(d, "sidecar.log")
    extra = list(ARMS[arm]) + list(a.extra or [])
    cmd = [sys.executable, "-u", PIN, os.path.join(ROOT, "scripts", "run_sidecar.py")] + (MOCK if a.mock else BASE) + [
        "--port", str(port), "--woosh-port", str(wport), "--cache-dir", CACHE, "--event-log", evlog] + extra
    phases = a.phases or PHASES[arm]
    rec = {"arm": arm, "rep": a.rep, "set": a.set, "cmd": cmd, "phases": phases, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "mock": bool(a.mock)}
    smi = Smi()
    smi.start()
    time.sleep(1.0)
    rec["gpu_before"] = smi.summary(time.time() - 1.0, time.time())
    fh = open(side_log, "w", encoding="utf-8")
    t_launch = time.time()
    srv = subprocess.Popen(cmd, cwd=ROOT, env=child_env(gpu=not a.mock), stdout=fh, stderr=subprocess.STDOUT)
    store = Store(os.path.join(d, "clips"), keep=not a.no_clips)
    loads = {} if a.mock or a.no_load else {k: Load(os.path.join(d, "gpuload.log"), *v) for k, v in LOADS.items()}
    load = loads.get(SESSION_LOAD)
    game = None
    try:
        ready = None
        while time.time() - t_launch < 900:
            if srv.poll() is not None:
                break
            with open(side_log, encoding="utf-8", errors="replace") as f:
                if "sidecar READY" in f.read():
                    ready = time.time()
                    break
            time.sleep(0.25)
        if ready is None:
            raise RuntimeError("sidecar never became READY, see %s" % side_log)
        time.sleep(2.0)
        rec["startup"] = {"ready_s": round(ready - t_launch, 1), "peak_mib": smi.peak(t_launch, ready),
                          "ready_mib": (smi.last(time.time()) or [None, None])[1]}
        game = Game(port, store)
        rec["welcome"] = game.side.welcome
        if "r" in phases:
            run_ref(game, rec)
        if "b" in phases:
            rec["b_load"] = SESSION_LOAD if load is not None else None
            if load is not None:
                rec["b_load_start_s"] = load.start()
                time.sleep(2.0)
            run_session(game, rec, smi)
            if "d" in phases:
                run_session(game, rec, smi, "b2")
            if load is not None:
                load.stop()
        if "u" in phases:
            run_drain(game, rec, smi)
        if "a" in phases:
            run_hol(game, rec, smi, n=a.hol_n)
        if "c" in phases:
            for cond in IDLE_CONDS:
                if cond == "noload" or cond in loads:
                    run_idle(game, rec, smi, cond, loads.get(cond))
        rec["stats_end"] = game.stats()
        rec["vram_run_peak_mib"] = smi.peak(ready, time.time())
    except Exception as e:
        rec["error"] = repr(e)[:500]
        print("ERROR %s" % rec["error"], flush=True)
    finally:
        for ld in loads.values():
            ld.stop()
        if game is not None:
            try:
                game.drain(timeout=30)
            except Exception:
                pass
            game.close()
        t_kill = time.time()
        if srv.poll() is None:
            srv.kill()
            try:
                srv.wait(timeout=30)
            except Exception:
                pass
        deadline = time.time() + 15.0
        left = []
        while time.time() < deadline:
            left = [p for p in psutil.process_iter(["pid", "name", "cmdline"])
                    if "python" in (p.info["name"] or "").lower() and "woosh_server" in " ".join(p.info["cmdline"] or [])
                    and ("--port %d" % wport) in " ".join(p.info["cmdline"] or [])]
            if not left:
                break
            time.sleep(0.2)
        for p in left:
            try:
                p.kill()
            except psutil.Error:
                pass
        rec["kill"] = {"woosh_left_after_15s": [p.pid for p in left], "seconds": round(time.time() - t_kill, 2)}
        fh.close()
        smi.stop()
        store.close()
        log = read_log(evlog)
        recs = game.records if game is not None else []
        match_log(recs, log)
        for r in recs:
            r.pop("done", None)
        msgs = game.messages if game is not None else []
        for m in msgs:
            m.pop("done", None)
        if "a" in rec:
            by = {}
            for r in recs:
                if r.get("phase") == "a":
                    by[(r["key"], r.get("tag"))] = r
            for t in rec["a"]["trials"]:
                pr_ = by.get((t["probe"], "%s%d" % (t["kind"], t["i"])))
                if pr_ is not None and pr_.get("log"):
                    t["probe_gen_ms"] = pr_["log"].get("gen_ms")
                    t["probe_engine_ms"] = pr_["log"].get("engine_ms")
                    t["probe_lock_wait_ms"] = pr_["log"].get("lock_wait_ms")
                    t["probe_await_ms"] = pr_["log"].get("await_ms")
                gr = by.get((t["gun"], "%s%d" % (t["kind"], t["i"]))) if t["gun"] else None
                if gr is not None and gr.get("log"):
                    t["gun_engine_ms"] = gr["log"].get("engine_ms")
            rec["a"]["summary"] = summarise_a(rec["a"]["trials"], recs)
        for ph in ("b", "b2"):
            if ph in rec:
                rec[ph]["summary"] = summarise_b(recs, log, rec[ph]["t_start"], rec[ph]["t_end"], ph)
                rec[ph]["upgrade_render"] = upgrade_spans(log, rec[ph]["t_start"], rec[ph]["t_end"])
                rn = rec[ph]["summary"]["renders"]
                wall = rn["wall_ms"] or 0.0
                rn["upgrade_ms"] = rec[ph]["upgrade_render"]["ms"]
                rn["gpu_share_with_upgrade"] = round((rn["gpu_ms_total"] + rn["upgrade_ms"]) / wall, 4) if wall else None
                rec[ph]["plays"] = plays(recs, ph)
        if "u" in rec:
            rec["u"]["upgrade_render"] = upgrade_spans(log, rec["u"]["t_start"], rec["u"]["t_drained"])
            rec["u"]["plays"] = plays(recs, "u")
        rec["upgrade_lag"] = upgrade_lag(log)
        acts = gen_activity(log)
        for c in rec.get("c", []):
            for t in c["trials"]:
                prev = [s for s in acts if s[1] <= t["wall"]]
                t["idle_since_gpu_s"] = round(t["wall"] - max(s[1] for s in prev), 3) if prev else None
                r = next((x for x in recs if x.get("phase") == "c_" + c["cond"] and x["key"] == t["key"]), None)
                if r is not None and r.get("log"):
                    t["gen_ms"] = r["log"].get("gen_ms")
                    t["engine_ms"] = r["log"].get("engine_ms")
            w = c["idle_window"]
            kw = [s for s in acts if s[2] == "keep_warm" and w["t0"] <= s[1] <= w["t1"]]
            w["keep_warm_jobs"] = len(kw)
            w["keep_warm_ms"] = round(sum(s[1] - s[0] for s in kw) * 1000.0, 1)
            w["keep_warm_duty"] = round(sum(s[1] - s[0] for s in kw) / max(1e-6, w["t1"] - w["t0"]), 4)
            span = [s for s in acts if s[2] == "keep_warm" and c["t_start"] <= s[1] <= c["t_end"]]
            c["keep_warm_jobs_total"] = len(span)
        rec["records"] = [clean(r) for r in recs]
        rec["messages"] = [clean(m) for m in msgs]
        rec["smi"] = [[round(r[0], 3), r[1], r[2], r[3], r[4], r[5]] for r in smi.rows]
        rec["event_log_lines"] = len(log)
        with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, default=str)
        short = {k: rec.get(k) for k in ("arm", "rep", "startup", "error", "vram_run_peak_mib", "kill")}
        for ph in ("b", "b2"):
            if ph in rec:
                short[ph] = {k: v for k, v in rec[ph]["summary"].items() if k in ("all", "renders")}
                short[ph]["plays"] = rec[ph]["plays"]
        if "u" in rec:
            short["u"] = {k: rec["u"][k] for k in ("drained_s", "replay_keys", "plays", "upgrade_render")}
        short["upgrade_lag"] = rec["upgrade_lag"]
        if "a" in rec:
            short["a"] = rec["a"].get("summary")
        if "c" in rec:
            short["c"] = [[c["cond"], c["idle_window"]["gpu"], [(t["gap_s"], t["route"], t["ms"]) for t in c["trials"]]]
                          for c in rec["c"]]
        print(json.dumps(short, default=str), flush=True)
        print("saved -> %s" % os.path.join(d, "result.json"), flush=True)
    return 0 if "error" not in rec else 1


def load_runs(set_dir):
    runs = {}
    for name in sorted(os.listdir(set_dir)):
        p = os.path.join(set_dir, name, "result.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                runs[name] = json.load(fh)
    return runs


def q16(y):
    return (np.round(np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2").astype(np.float32) / np.float32(32767.0))


def gate_l(set_dir, runs):
    idx = {}
    for name in runs:
        p = os.path.join(set_dir, name, "clips", "index.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                idx[name] = json.load(fh)["clips"]
    head_like = [n for n in idx if runs[n]["arm"] in ("head", "ref")]
    ref = {}
    for n in head_like:
        for c in idx[n]:
            ref.setdefault((c["key"], c["variant"]), {}).setdefault((c["sha1"], tuple(c["shape"])), []).append(
                (n, c["file"]))
    out = {"head_like_runs": head_like, "per_run": {}, "head_conflicts": []}
    for kv, shas in sorted(ref.items()):
        if len(shas) > 1:
            out["head_conflicts"].append({"key": kv[0], "variant": kv[1],
                                          "versions": [[list(s), [x[0] for x in v]] for s, v in shas.items()]})
    for n, clips in idx.items():
        served = {}
        for r in runs[n].get("records", []):
            if r.get("sha1"):
                served.setdefault((r["key"], r.get("variant"), r["sha1"]), []).append(r.get("served"))
        res = {"clips": len(clips), "identical": 0, "disk16": 0, "different": [], "no_reference": [],
               "replies": sum(len(v) for v in served.values())}
        for c in clips:
            kv = (c["key"], c["variant"])
            if kv not in ref:
                res["no_reference"].append("%s v%d" % kv)
                continue
            if (c["sha1"], tuple(c["shape"])) in ref[kv]:
                res["identical"] += 1
                continue
            y = np.load(os.path.join(set_dir, n, "clips", c["file"])) if os.path.isfile(
                os.path.join(set_dir, n, "clips", c["file"])) else None
            row = {"key": c["key"], "variant": c["variant"], "served": served.get((c["key"], c["variant"], c["sha1"])),
                   "shape": c["shape"]}
            disk = False
            if y is not None:
                for (sha, shape), where in ref[kv].items():
                    rn, rf = where[0]
                    rp = os.path.join(set_dir, rn, "clips", rf)
                    if not os.path.isfile(rp):
                        continue
                    x = np.load(rp)
                    if x.shape == y.shape:
                        row.setdefault("max_abs", []).append(float(np.max(np.abs(x.astype(np.float64) - y))))
                        if np.array_equal(q16(x), y) or np.array_equal(q16(y), x):
                            disk = True
            if disk:
                res["disk16"] += 1
                row["disk16"] = True
            res["different"].append(row)
        out["per_run"][n] = res
    return out


def overlap_stats(set_dir, runs):
    out = {}
    for n, r in runs.items():
        log = read_log(os.path.join(set_dir, n, "events.jsonl"))
        dps = [x for x in log if x.get("type") == "deepen"]
        rt = [x for x in log if x.get("type") == "event" and x.get("served") == "realtime"]
        ov, slow, extra = 0, 0, []
        for e in rt:
            s0 = e["t_recv"]
            s1 = log_end(e)
            if any(any(k == e["key"] for k, _v in d["items"]) and d["t"] - d.get("ms", 0) / 1e3 < s1 and d["t"] > s0
                   for d in dps):
                ov += 1
            if e.get("gen_ms"):
                x = e.get("engine_ms", 0.0) - e["gen_ms"]
                extra.append(x)
                if x > LATE_MS:
                    slow += 1
        out[n] = {"realtime": len(rt), "with_background_render_of_same_key": ov, "woosh_realtime": len(extra),
                  "woosh_engine_minus_gen_gt20": slow, "woosh_engine_minus_gen_p50": pct(extra, 50),
                  "woosh_engine_minus_gen_max": round(max(extra), 1) if extra else None}
    return out


def report(a):
    set_dir = os.path.join(OUT, a.set)
    runs = load_runs(set_dir)
    out = {"set": a.set, "runs": sorted(runs)}
    arms = {}
    for n, r in runs.items():
        arms.setdefault(r["arm"], []).append(n)
    out["arms"] = arms
    vg = {}
    for n, r in runs.items():
        st = r.get("startup") or {}
        both = None
        if "a" in r:
            both = (r["a"].get("summary") or {}).get("fresh", {}).get("vram_peak_mib")
        vg[n] = {"startup_peak_mib": st.get("peak_mib"), "ready_mib": st.get("ready_mib"), "ready_s": st.get("ready_s"),
                 "both_render_peak_mib": both, "run_peak_mib": r.get("vram_run_peak_mib"),
                 "ok": (st.get("peak_mib") or 0) <= V_START_MIB and (st.get("ready_mib") or 0) <= V_READY_MIB and (
                     both is None or both < V_BOTH_MIB)}
    out["gate_v"] = vg
    a_tab = {}
    for arm, names in arms.items():
        trials = [t for n in names if "a" in runs[n] for t in runs[n]["a"]["trials"]]
        if trials:
            a_tab[arm] = summarise_a(trials, [])
            a_tab[arm]["runs"] = [n for n in names if "a" in runs[n]]
    out["a"] = a_tab
    b_tab = {}
    for arm, names in arms.items():
        bs = [runs[n] for n in names if "b" in runs[n]]
        if not bs:
            continue
        recs = [r for x in bs for r in x["records"] if r.get("phase") == "b" and r["kind"] == "event"]
        per = {}
        for c in sorted({r["cls"] for r in recs}) + ["all"]:
            rs = [r for r in recs if c == "all" or r["cls"] == c]
            ms = [r.get("ms") for r in rs if r.get("ms") is not None]
            fp = [r.get("ms") for r in rs if not r["held_before"] and r.get("ms") is not None]
            per[c] = {"n": len(rs), "pct_on_demand": round(100.0 * sum(1 for r in rs if r.get("served") == "realtime")
                                                           / max(1, len(rs)), 1),
                      "p50_ms": pct(ms, 50), "p95_ms": pct(ms, 95), "late_gt20_pct": round(
                          100.0 * sum(1 for x in ms if x > LATE_MS) / max(1, len(ms)), 1),
                      "first_n": len(fp), "first_p50_ms": pct(fp, 50), "first_p95_ms": pct(fp, 95),
                      "first_late_gt20_pct": round(100.0 * sum(1 for x in fp if x > LATE_MS) / max(1, len(fp)), 1),
                      "per_run_pct_on_demand": [x["b"]["summary"].get(c, {}).get("pct_on_demand") for x in bs]}
        rn = [x["b"]["summary"]["renders"] for x in bs]
        per["renders"] = {k: [r[k] for r in rn] for k in ("realtime_n", "realtime_ms", "prediction_n", "prediction_ms",
                                                          "wasted_n", "wasted_ms", "gpu_ms_total", "gpu_share",
                                                          "wasted_share", "woosh_gen_ms_p50", "woosh_gen_ms_p95")}
        per["renders"]["wasted_keys"] = [r["wasted_keys"] for r in rn]
        b_tab[arm] = per
    out["b"] = b_tab
    c_tab = {}
    for arm, names in arms.items():
        for n in names:
            for c in runs[n].get("c", []):
                k = "%s/%s" % (arm, c["cond"])
                e = c_tab.setdefault(k, {"idle_power_w": [], "idle_sm_mhz": [], "keep_warm_duty": [], "trials": []})
                e["idle_power_w"].append(c["idle_window"]["gpu"]["power_w_mean"])
                e["idle_sm_mhz"].append(c["idle_window"]["gpu"]["sm_mhz_median"])
                e["keep_warm_duty"].append(c["idle_window"].get("keep_warm_duty"))
                e["trials"] += c["trials"]
    for k, e in c_tab.items():
        cells = {}
        for t in e["trials"]:
            cells.setdefault("%s_%.0fs" % (t["route"], t["gap_s"]), []).append(t)
        e["cells"] = {cid: {"n": len(ts), "ms_median": pct([t["ms"] for t in ts], 50),
                            "ms_max": max((t["ms"] for t in ts if t["ms"] is not None), default=None),
                            "engine_ms_median": pct([t.get("engine_ms") for t in ts], 50),
                            "gen_ms_median": pct([t.get("gen_ms") for t in ts], 50),
                            "sm_before_median": pct([t["sm_before"] for t in ts], 50),
                            "pstates_before": sorted({str(t["pstate_before"]) for t in ts}),
                            "gap_power_w_mean": pct([t["gap_power_w_mean"] for t in ts], 50),
                            "since_reply_s_median": pct([t.get("since_reply_s") for t in ts], 50),
                            "idle_since_gpu_s_median": pct([t.get("idle_since_gpu_s") for t in ts], 50),
                            "idle_since_gpu_s_max": max((t.get("idle_since_gpu_s") or 0.0 for t in ts), default=None)}
                      for cid, ts in sorted(cells.items())}
        del e["trials"]
    out["c"] = c_tab
    out["gate_l"] = gate_l(set_dir, runs)
    out["overlap"] = overlap_stats(set_dir, runs)
    path = os.path.join(set_dir, "report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("arms", "gate_v")}, default=str, indent=1))
    print("saved -> %s" % path)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="on-time playback bench through the real sidecar: two event lanes and a "
                                             "control lane like the Unity T preset; head-of-line, scripted session "
                                             "with predictions, idle clocks, and clip identity across flags")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=sorted(ARMS), required=True)
    r.add_argument("--rep", type=int, default=1)
    r.add_argument("--set", default="s1")
    r.add_argument("--phases", default=None, help="subset of r (reference), b (session), a (head-of-line), c (idle)")
    r.add_argument("--port", type=int, default=8767)
    r.add_argument("--woosh-port", type=int, default=8772)
    r.add_argument("--hol-n", type=int, default=8)
    r.add_argument("--mock", action="store_true")
    r.add_argument("--no-load", action="store_true")
    r.add_argument("--no-clips", action="store_true")
    r.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    g = sub.add_parser("gpuload")
    g.add_argument("--hz", type=float, default=30.0)
    g.add_argument("--size", type=int, default=2048)
    g.add_argument("--iters", type=int, default=1)
    g.add_argument("--seconds", type=float, default=900.0)
    g.add_argument("--parent-pid", type=int, default=0)
    p = sub.add_parser("report")
    p.add_argument("--set", default="s1")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "gpuload":
        return gpuload(a)
    return report(a)


if __name__ == "__main__":
    raise SystemExit(main())
