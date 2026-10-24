from __future__ import annotations

import threading

from . import writer as wr

SOURCES = ("canon", "ground", "named")


class SharedBackend:
    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self._lock = threading.Lock()
        self._users = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def start(self):
        with self._lock:
            self._users += 1
            self.inner.start()

    def close(self):
        with self._lock:
            self._users -= 1
            if self._users <= 0:
                self._users = 0
                self.inner.close()


def same_source(w, source):
    g = getattr(w, "ground", None)
    if source == "ground":
        return g is not None and g.mode == "caption"
    return g is None and w.style == source


def decided(w, key):
    st = w.store
    if st is not None and str(key) in st:
        return True
    return w.ready.is_set() and (w.readonly or str(w.state).startswith(("failed", "backend failed")))


class PromptPool:
    def __init__(self, writers, primary=None, route=()):
        self.writers = [w for w in writers if w is not None]
        self.primary = primary
        self.owned = [w for w in self.writers if w is not primary]
        self.route = [w for w in route if w is not None]

    def describe(self):
        return ", ".join(w.describe() + (" (route only)" if self.route_only(w) else "") for w in self.writers)

    def route_only(self, w):
        return any(w is r for r in self.route)

    def offered(self):
        return [w for w in self.writers if not self.route_only(w)]

    def want(self, key, why="seen", eta=None):
        for w in self.owned:
            w.want(key, why, eta)

    def ready(self, key):
        return all(decided(w, key) for w in self.writers)

    def prompts(self, key, route="woosh"):
        out = []
        for w in self.offered():
            rec = w.lookup(key)
            p = rec.get(route) if rec else None
            if p and p not in out:
                out.append(p)
        return out

    def texts(self, key):
        for w in [self.primary] + self.offered():
            rec = w.lookup(key) if w is not None else None
            if rec and isinstance(rec.get("texts"), dict):
                return dict(rec["texts"])
        return None

    def start(self):
        for w in self.owned:
            w.start()

    def stop(self):
        for w in self.owned:
            w.stop()

    def stats(self):
        return [dict(w.stats, style=w.style, state=w.state, stored=len(w.store or ())) for w in self.writers]


def build(sources, primary=None, model="", binary=None, port=wr.DEFAULT_PORT, threads=4, affinity=wr.DEFAULT_AFFINITY,
          store=None, when=wr.WHEN, log_path=None, template=None, ground_index=None, log=None, firearm=False,
          route_sources=()):
    names = [s for s in sources if s]
    bad = [s for s in names if s not in SOURCES]
    if bad or not names:
        raise ValueError("unknown prompt pool source %s (choose from %s)" % (", ".join(bad) or "none",
                                                                             ", ".join(SOURCES)))
    shared = None
    if any(s in ("canon", "named") for s in names):
        base = getattr(primary, "backend", None)
        if isinstance(base, SharedBackend):
            shared = base
        elif base is not None and base.name == "llama":
            shared = SharedBackend(base)
            primary.backend = shared
        else:
            from .canon import SLOTS_WITH_ASK
            shared = SharedBackend(wr.LlamaBackend(model, binary, port, threads, affinity, log_path=log_path,
                                                   slots=SLOTS_WITH_ASK if firearm and "canon" in names else 1))
    writers, route = [], []
    for s in names:
        if primary is not None and same_source(primary, s):
            w = primary
        elif s == "ground":
            w = wr.build("ground", store=store, when=when, ground="caption", ground_index=ground_index,
                         threads=threads, log=log)
        else:
            w = wr.build("llama", store=store, when=when, style=s, template=template, backend=shared, log=log,
                         firearm=firearm and s == "canon")
        writers.append(w)
        if s in route_sources:
            route.append(w)
    return PromptPool(writers, primary, route)
