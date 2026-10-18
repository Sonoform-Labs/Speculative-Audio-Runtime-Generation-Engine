from __future__ import annotations

import threading
import time

from ..amp import family_for
from ..startup import stage
from ..types import AudioClip, CacheKey
from ..words import split_words
from .base import Generator

GUN_FAMILIES = ("blast", "handling")
WARM_KEY = CacheKey("shoot", "pistol", "normal")
ROUTES = ("families", "words", "class", "single")
SIDES = ("guns", "rest")
GUN_CLASSES = ("weapon",)
DISCHARGE_WORDS = frozenset(("shoot", "shoots", "shooting", "shot", "shots", "gunshot", "gunshots", "gunfire",
                             "fire", "fires", "fired", "firing"))
FIREARM_WORDS = frozenset(("gun", "guns", "firearm", "firearms", "pistol", "pistols", "revolver", "revolvers",
                           "handgun", "handguns", "rifle", "rifles", "carbine", "carbines", "shotgun", "shotguns",
                           "smg", "musket", "muskets", "cannon", "cannons"))


def weapon_words(key) -> bool:
    if any(w in DISCHARGE_WORDS or w in FIREARM_WORDS for w in split_words(key.event_type)):
        return True
    return any(w in FIREARM_WORDS for w in split_words(key.material))


def side_for(key, mode="families", gun_families=GUN_FAMILIES, gun_classes=GUN_CLASSES, single="rest") -> str:
    if mode == "words":
        return "guns" if weapon_words(key) else "rest"
    if mode == "class":
        return "guns" if str(getattr(key, "cls", "") or "").strip().lower() in gun_classes else "rest"
    if mode == "single":
        return single
    return "guns" if family_for(key.event_type) in gun_families else "rest"


def release_cuda_cache():
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


class RoutedGenerator(Generator):
    name = "hybrid"
    latency_ms = 260.0

    def __init__(self, woosh_host: str = "127.0.0.1", woosh_port: int = 8771, gun_families=GUN_FAMILIES,
                 guns=None, rest=None, warm_guns: bool = True, start_rest=None, gun_cost_ms: float = 155.0,
                 rest_cost_ms: float = 50.0, load_rest=None, drop_rest=None, route: str = "families",
                 gun_classes=GUN_CLASSES, single_route: str = "rest", **sa3_kwargs):
        if route not in ROUTES:
            raise ValueError("unknown route %r, choose from %s" % (route, ", ".join(ROUTES)))
        if single_route not in SIDES:
            raise ValueError("unknown single route %r, choose from %s" % (single_route, ", ".join(SIDES)))
        if guns is None:
            from .stable_audio3 import StableAudio3Generator
            guns = StableAudio3Generator(**sa3_kwargs)
        if rest is None:
            from .remote import RemoteGenerator
            rest = RemoteGenerator(host=woosh_host, port=int(woosh_port))
        self.guns = guns
        self.rest = rest
        self.gun_families = set(gun_families)
        self.route_mode = route
        self.gun_classes = tuple(str(c).strip().lower() for c in gun_classes if str(c).strip())
        self.single_route = single_route
        self.sample_rate = int(getattr(guns, "sample_rate", self.sample_rate))
        self.prompts = getattr(guns, "prompts", {})
        if isinstance(getattr(rest, "prompts", None), dict):
            rest.prompts = self.prompts
        self._rest_lock = threading.Lock()
        self.counts = {"guns": 0, "rest": 0}
        self.warm_guns = bool(warm_guns)
        self.start_rest = start_rest
        self.load_rest = load_rest
        self.drop_rest = drop_rest
        self._rest_started = False
        self.costs = {"guns": float(gun_cost_ms), "rest": float(rest_cost_ms)}

    def route(self, key: CacheKey) -> str:
        return side_for(key, self.route_mode, self.gun_families, self.gun_classes, self.single_route)

    def cost_ms(self, key: CacheKey) -> float:
        return self.costs[self.route(key)]

    @property
    def post_tag(self) -> str:
        tag = getattr(self.guns, "post_tag", "")
        extra = []
        if self.route_mode != "families":
            extra.append("route:%s" % (self.route_mode + (":" + self.single_route if self.route_mode == "single" else "")))
        if getattr(self.guns, "prompt_style", "") == "generic":
            extra.append("prompts:generic")
        return "|".join([tag] + extra) if extra else tag

    def ensure_ready(self):
        if self.start_rest is not None:
            return self._ready_guns_first()
        for g in (self.guns, self.rest):
            ready = getattr(g, "ensure_ready", None)
            if ready:
                ready()
        if self.warm_guns:
            self.guns.generate(WARM_KEY, variant=0)

    def _drop_rest(self):
        drop, self.drop_rest = self.drop_rest, None
        if drop is not None and not self._rest_started:
            try:
                drop()
            except Exception:
                pass

    def _ready_guns_first(self):
        t = time.time()
        try:
            ready = getattr(self.guns, "ensure_ready", None)
            if ready:
                ready()
            t = stage("guns loaded", t)
            if self.load_rest is not None:
                load, self.load_rest = self.load_rest, None
                load()
            if not self._rest_started:
                if self.warm_guns:
                    self.guns.generate(WARM_KEY, variant=0)
                    t = stage("gun warm 1", t)
                release_cuda_cache()
                t = stage("release_cuda_cache", t)
        except BaseException:
            self._drop_rest()
            raise
        if not self._rest_started:
            self.drop_rest = None
            self.start_rest()
            self._rest_started = True
            t = stage("woosh started", t)
        ready = getattr(self.rest, "ensure_ready", None)
        if ready:
            ready()
        t = stage("woosh connected", t)
        if self.warm_guns:
            self.guns.generate(WARM_KEY, variant=0)
            stage("gun warm 2", t)

    def generate(self, key: CacheKey, *, variant: int = 0) -> AudioClip:
        side = self.route(key)
        self.counts[side] += 1
        if side == "guns":
            return self.guns.generate(key, variant=variant)
        with self._rest_lock:
            return self.rest.generate(key, variant=variant)

    def generate_batch(self, items: list) -> list:
        out = [None] * len(items)
        guns = [i for i, (k, _v) in enumerate(items) if self.route(k) == "guns"]
        if guns:
            clips = self.guns.generate_batch([items[i] for i in guns])
            for i, c in zip(guns, clips):
                out[i] = c
            self.counts["guns"] += len(guns)
        for i, (k, v) in enumerate(items):
            if out[i] is None:
                out[i] = self.generate(k, variant=v)
        return out

    def fast_stats(self) -> dict:
        s = dict(getattr(self.guns, "fast_stats", lambda: {})())
        s.update({"routed_" + k: v for k, v in self.counts.items()})
        return s
