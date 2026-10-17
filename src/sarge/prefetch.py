from __future__ import annotations

import json
import os
import re
import struct
import threading
import time

from .startup import stage

BLOCK = 8 << 20
STREAMS = 2
RESERVE = 6 << 30
REPOS = {"medium": "stabilityai/stable-audio-3-medium", "medium-base": "stabilityai/stable-audio-3-medium-base",
         "small-sfx": "stabilityai/stable-audio-3-small-sfx", "small-sfx-base": "stabilityai/stable-audio-3-small-sfx-base",
         "small-music": "stabilityai/stable-audio-3-small-music",
         "small-music-base": "stabilityai/stable-audio-3-small-music-base"}
WOOSH_HOME = r"F:\AI\cache\woosh"
WOOSH_MODELS = {"flow": "Woosh-Flow", "dflow": "Woosh-DFlow"}
_DEP = re.compile(r"^\s*path:\s*(checkpoints/[^\s#]+)", re.M)


def enabled(env=None) -> bool:
    return (env if env is not None else os.environ).get("SARGE_PREFETCH", "1").strip().lower() not in ("0", "off", "no")


def hub_dir(env=None) -> str:
    env = env if env is not None else os.environ
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(k):
            return env[k]
    home = env.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def snapshot(repo_id: str, hub: str | None = None) -> str | None:
    root = os.path.join(hub or hub_dir(), "models--" + repo_id.replace("/", "--"))
    try:
        with open(os.path.join(root, "refs", "main"), encoding="utf-8") as fh:
            rev = fh.read().strip()
    except OSError:
        return None
    path = os.path.join(root, "snapshots", rev)
    return path if os.path.isdir(path) else None


def header(path: str):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return 8 + n, json.loads(fh.read(n))


def span(path: str, prefix: str = ""):
    base, h = header(path)
    offs = [v["data_offsets"] for k, v in h.items() if k != "__metadata__" and k.startswith(prefix)]
    if not offs:
        return None
    return base + min(a for a, _b in offs), base + max(b for _a, b in offs)


def item(path: str, label: str, start: int = 0, end: int | None = None):
    path = os.path.realpath(path)
    if not os.path.isfile(path):
        return None
    size = os.path.getsize(path)
    end = size if end is None else min(int(end), size)
    if end <= start:
        return None
    return {"path": path, "label": label, "start": int(start), "end": int(end)}


def t5_dir(config: dict, snap: str, hub: str):
    for c in config.get("model", {}).get("conditioning", {}).get("configs", []):
        if c.get("type") != "t5gemma":
            continue
        cc = c.get("config", {})
        if cc.get("model_path"):
            return cc["model_path"]
        repo = cc.get("repo_id")
        root = snapshot(repo, hub) if repo else None
        if root is None:
            return None
        return os.path.join(root, cc["subfolder"]) if cc.get("subfolder") else root
    return None


def sa3_items(model_name: str, hub: str | None = None, weights: str | None = None) -> list:
    hub = hub or hub_dir()
    repo = REPOS.get(model_name)
    snap = snapshot(repo, hub) if repo else None
    if snap is None:
        return []
    out = []
    try:
        with open(os.path.join(snap, "model_config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
        t5 = t5_dir(config, snap, hub)
    except (OSError, ValueError):
        t5 = None
    if t5:
        out.append(item(os.path.join(t5, "tokenizer.json"), "t5 tokenizer"))
        st = os.path.join(t5, "model.safetensors")
        try:
            rng = span(st, "model.encoder.") if os.path.isfile(st) else None
        except (OSError, ValueError, struct.error):
            rng = None
        out.append(item(st, "t5 encoder", *rng) if rng else item(st, "t5 weights"))
    out.append(item(weights, "sa3 weight cache") if weights else item(os.path.join(snap, "model.safetensors"),
                                                                          "sa3 weights"))
    return [x for x in out if x]


def woosh_items(model: str = "dflow", home: str | None = None) -> list:
    home = home or os.environ.get("WOOSH_HOME") or WOOSH_HOME
    name = WOOSH_MODELS.get(model)
    if name is None:
        return []
    own = os.path.join(home, "checkpoints", name)
    try:
        with open(os.path.join(own, "config.yaml"), encoding="utf-8") as fh:
            deps = _DEP.findall(fh.read())
    except OSError:
        deps = []
    dirs = [os.path.join(home, *d.split("/")) for d in reversed(deps)] + [own]
    return [x for x in (item(os.path.join(d, "weights.safetensors"), "woosh " + os.path.basename(d)) for d in dirs) if x]


def available() -> int | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        s = Status()
        s.dwLength = ctypes.sizeof(Status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
            return None
        return int(s.ullAvailPhys)
    except Exception:
        return None


def drive(path: str) -> str:
    return os.path.splitdrive(path)[0].upper() or "/"


def _read(path: str, start: int, end: int, block: int) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_SEQUENTIAL", 0)
    fd = os.open(path, flags)
    n = 0
    try:
        with os.fdopen(fd, "rb", buffering=0) as fh:
            fh.seek(start)
            buf = bytearray(block)
            view = memoryview(buf)
            while start + n < end:
                want = min(block, end - start - n)
                got = fh.readinto(view[:want])
                if not got:
                    break
                n += got
    except OSError:
        pass
    return n


class Prefetcher:
    def __init__(self, items, streams: int = STREAMS, block: int = BLOCK, reserve: int = RESERVE, free=available):
        self.items = [x for x in items if x]
        self.streams = max(1, int(streams))
        self.block = int(block)
        self.reserve = int(reserve)
        self.free = free
        self.done = threading.Event()
        self.log = []
        self._threads = []

    def queues(self) -> dict:
        out = {}
        for x in self.items:
            out.setdefault(drive(x["path"]), []).append(x)
        return out

    def fetch(self, x) -> dict:
        size = x["end"] - x["start"]
        free = self.free() if self.free else None
        if free is not None and free < size + self.reserve:
            row = dict(x, mb=round(size / 1e6, 1), skipped="ram %.1f GB free" % (free / 1e9))
            stage("prefetch %s skipped, %.1f GB of RAM available" % (x["label"], free / 1e9))
            self.log.append(row)
            return row
        t = time.time()
        cut = [x["start"] + size * i // self.streams for i in range(self.streams + 1)]
        got = [0] * self.streams

        def one(i):
            got[i] = _read(x["path"], cut[i], cut[i + 1], self.block)

        ts = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(self.streams)]
        for th in ts:
            th.start()
        for th in ts:
            th.join()
        dt = max(1e-6, time.time() - t)
        row = dict(x, mb=round(sum(got) / 1e6, 1), s=round(dt, 2), mb_s=round(sum(got) / 1e6 / dt, 1))
        stage("prefetch %s %.0f MB at %.0f MB/s" % (x["label"], sum(got) / 1e6, sum(got) / 1e6 / dt), t)
        self.log.append(row)
        return row

    def _run(self, q):
        for x in q:
            self.fetch(x)

    def start(self):
        qs = self.queues()
        self._threads = [threading.Thread(target=self._run, args=(q,), name="prefetch-" + d, daemon=True)
                         for d, q in qs.items()]
        for th in self._threads:
            th.start()

        def finish():
            for th in self._threads:
                th.join()
            self.done.set()

        threading.Thread(target=finish, name="prefetch-done", daemon=True).start()
        return self


def start(items, **kw) -> Prefetcher:
    return Prefetcher(items, **kw).start()


def start_hybrid(model_name: str, woosh_model: str | None = "dflow", weights=None, env=None):
    if not enabled(env):
        return None
    box = {}

    def plan():
        t = time.time()
        try:
            w = weights() if callable(weights) else weights
        except Exception:
            w = None
        items = sa3_items(model_name, weights=w) + (woosh_items(woosh_model) if woosh_model else [])
        stage("prefetch plan: %d files, %.1f GB" % (len(items), sum(x["end"] - x["start"] for x in items) / 1e9), t)
        box["p"] = start(items)

    th = threading.Thread(target=plan, name="prefetch-plan", daemon=True)
    th.start()
    return th
