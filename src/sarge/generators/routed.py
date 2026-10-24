from __future__ import annotations

import threading
import time

from ..amp import family_for
from ..startup import stage
from ..types import AudioClip, CacheKey
from ..words import clean_words, split_words
from .base import Generator

GUN_FAMILIES = ("blast", "handling")
WARM_KEY = CacheKey("shoot", "pistol", "normal")
ROUTES = ("families", "words", "class", "single", "meaning")
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


def firearm_words(key) -> bool:
    if any(w in DISCHARGE_WORDS or w in FIREARM_WORDS for w in clean_words(key.event_type)):
        return True
    return any(w in FIREARM_WORDS for w in clean_words(key.material))


def meaning_side(key, meaning=None) -> str:
    ans = None
    if meaning is not None:
        try:
            ans = meaning(key)
        except Exception:
            ans = None
    if ans is None:
        ans = firearm_words(key)
    return "guns" if ans else "rest"


def side_for(key, mode="families", gun_families=GUN_FAMILIES, gun_classes=GUN_CLASSES, single="rest",
             meaning=None) -> str:
    if mode == "meaning":
        return meaning_side(key, meaning)
    if mode == "words":
        return "guns" if weapon_words(key) else "rest"
    if mode == "class":
        return "guns" if str(getattr(key, "cls", "") or "").strip().lower() in gun_classes else "rest"
    if mode == "single":
        return single
    return "guns" if family_for(key.event_type) in gun_families else "rest"


def scene_words(style) -> bool:
    from ..prompts import template_mods
    return "loop" in template_mods(style)


class UpgradeFailed(RuntimeError):
    def __init__(self, message, unreachable=False):
        super().__init__(message)
        self.unreachable = bool(unreachable)


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
                 gun_classes=GUN_CLASSES, single_route: str = "rest", upgrade: str | None = None,
                 first_prompt: str | None = None, sa3_best: int = 1, **sa3_kwargs):
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
        self.upgrade = upgrade or None
        self.first_prompt = first_prompt or None
        self.woosh_host, self.woosh_port = woosh_host, int(woosh_port)
        self._up = self._ctl = None
        self._up_busy = threading.Event()
        self.upgrade_stats = {"renders": 0, "preempted": 0, "failed": 0, "aborts_sent": 0}
        self.send_scene = scene_words(getattr(guns, "prompt_style", ""))
        if self.send_scene and hasattr(rest, "send_scene"):
            rest.send_scene = True
        self.meaning = None
        self.made = {}
        self.flipped = 0
        self.sa3_best = max(1, int(sa3_best or 1))
        self.sa3_stats = {"candidates": 0, "scored": 0, "failed": 0}

    def _remote(self):
        from .remote import RemoteGenerator
        r = RemoteGenerator(host=self.woosh_host, port=self.woosh_port)
        r.prompts = self.prompts
        r.send_scene = self.send_scene
        return r

    def _abort_upgrade(self):
        if not self._up_busy.is_set():
            return
        try:
            if self._ctl is None:
                self._ctl = self._remote()
            self._ctl.control({"type": "upgrade_abort"}, reply=False)
            self.upgrade_stats["aborts_sent"] += 1
        except Exception:
            self._ctl = None

    def upgrade_control(self, kind: str) -> dict:
        if not self.upgrade:
            return {"state": "off"}
        if self._up is None:
            self._up = self._remote()
        try:
            return self._up.control({"type": kind}) or {}
        except Exception as e:
            self._up = None
            return {"state": "unreachable", "error": str(e)}

    def upgrade_take(self, key: CacheKey, variant: int, prompt: str | None = None, pool=None, texts=None):
        if not self.upgrade or self.route(key) == "guns":
            return None
        if self._up is None:
            self._up = self._remote()
        self._up_busy.set()
        extra = {k: v for k, v in (("pool", pool), ("texts", texts)) if v}
        try:
            clip = self._up.generate(key, variant=variant, prompt=prompt, engine=self.upgrade, **extra)
        except (ConnectionError, OSError) as e:
            self._up = None
            self.upgrade_stats["failed"] += 1
            raise UpgradeFailed("%s: %s" % (type(e).__name__, e), unreachable=True) from e
        except Exception as e:
            if isinstance(e, RuntimeError) and "Preempted" in str(e):
                self.upgrade_stats["preempted"] += 1
                return None
            self.upgrade_stats["failed"] += 1
            raise UpgradeFailed("%s: %s" % (type(e).__name__, e)) from e
        finally:
            self._up_busy.clear()
        self.upgrade_stats["renders"] += 1
        return clip

    def sa3_style(self) -> str:
        return str(getattr(self.guns, "words_style", None) or getattr(self.guns, "prompt_style", "") or "")

    def sa3_candidate(self, key: CacheKey, variant: int) -> AudioClip:
        try:
            clip = self.guns.generate(key, variant=int(variant))
        except Exception as e:
            self.sa3_stats["failed"] += 1
            raise UpgradeFailed("%s: %s" % (type(e).__name__, e)) from e
        self.sa3_stats["candidates"] += 1
        return clip

    def sa3_verify(self, key: CacheKey, clips: list) -> list:
        if self._up is None:
            self._up = self._remote()
        try:
            out = self._up.verify(key, clips, self.sa3_style())
        except (ConnectionError, OSError) as e:
            self._up = None
            self.sa3_stats["failed"] += 1
            raise UpgradeFailed("%s: %s" % (type(e).__name__, e), unreachable=True) from e
        except Exception as e:
            self.sa3_stats["failed"] += 1
            raise UpgradeFailed("%s: %s" % (type(e).__name__, e)) from e
        self.sa3_stats["scored"] += len(clips)
        return out

    def route(self, key: CacheKey) -> str:
        return side_for(key, self.route_mode, self.gun_families, self.gun_classes, self.single_route, self.meaning)

    def route_tag(self, key: CacheKey) -> str:
        return "side:" + self.route(key) if self.route_mode == "meaning" else ""

    def _made(self, key, side):
        if self.route_mode == "meaning":
            self.made[key] = side

    def flips(self) -> list:
        if self.route_mode != "meaning":
            return []
        return [k for k, side in list(self.made.items()) if self.route(k) != side]

    def settle(self, key) -> str:
        side = self.route(key)
        if self.made.get(key, side) != side:
            self.flipped += 1
        self.made[key] = side
        return side

    def cost_ms(self, key: CacheKey) -> float:
        return self.costs[self.route(key)]

    @property
    def post_tag(self) -> str:
        tag = getattr(self.guns, "post_tag", "")
        extra = []
        if self.route_mode != "families":
            extra.append("route:%s" % (self.route_mode + (":" + self.single_route if self.route_mode == "single" else "")))
        style = str(getattr(self.guns, "prompt_style", "") or "")
        if style == "generic" or style.startswith("template:"):
            extra.append("prompts:" + style)
        if self.first_prompt:
            extra.append("first:" + self.first_prompt)
        if getattr(self.guns, "raw_words", False):
            extra.append("sa3:raw")
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

    def generate(self, key: CacheKey, *, variant: int = 0, prompt: str | None = None) -> AudioClip:
        side = self.route(key)
        self.counts[side] += 1
        self._made(key, side)
        kw = {"prompt": prompt} if prompt else {}
        if side == "guns":
            if self.upgrade:
                self._abort_upgrade()
            return self.guns.generate(key, variant=variant, **kw)
        with self._rest_lock:
            return self.rest.generate(key, variant=variant, **kw)

    def generate_batch(self, items: list) -> list:
        out = [None] * len(items)
        guns = [i for i, (k, _v) in enumerate(items) if self.route(k) == "guns"]
        if guns:
            if self.upgrade:
                self._abort_upgrade()
            clips = self.guns.generate_batch([items[i] for i in guns])
            for i, c in zip(guns, clips):
                out[i] = c
                self._made(items[i][0], "guns")
            self.counts["guns"] += len(guns)
        for i, (k, v) in enumerate(items):
            if out[i] is None:
                out[i] = self.generate(k, variant=v)
        return out

    def fast_stats(self) -> dict:
        s = dict(getattr(self.guns, "fast_stats", lambda: {})())
        s.update({"routed_" + k: v for k, v in self.counts.items()})
        if self.route_mode == "meaning":
            s["routed_flips"] = self.flipped
        if self.upgrade:
            s.update({"upgrade_" + k: v for k, v in self.upgrade_stats.items()})
        if self.sa3_best > 1:
            s.update({"sa3_upgrade_" + k: v for k, v in self.sa3_stats.items()})
        return s
