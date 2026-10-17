from __future__ import annotations

import json
import re
import socket
import socketserver
import threading
import time

from . import wire
from .wire import ClientGone, FrameError, IdleTimeout, MsgType
from .. import __version__
from ..affordance import AffordanceEngine, GameContext
from ..calibration import PredictorCalibration
from ..fusion import PredictionFusion
from ..generators import get_generator
from ..predictor import EventSequencePredictor
from ..startup import stage
from ..types import CacheKey

engine_lock = threading.Lock()
INFLIGHT_WAIT_S = 0.5

MAX_PREBAKE_CONTEXTS = 256
INTENT_IGNORE = {"footstep", "jump", "land"}


class SfxServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr, engine, default_pcm: str = "f32le", recv_timeout: float = 30.0):
        super().__init__(addr, _Handler)
        self.engine = engine
        self.default_pcm = default_pcm
        self.recv_timeout = recv_timeout
        self.generator_name = getattr(engine.gen, "name", "unknown")
        self.sample_rate = int(getattr(engine.gen, "sample_rate", 48000))


class _Handler(socketserver.BaseRequestHandler):
    def setup(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.request.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.request.settimeout(self.server.recv_timeout)
        self.pcm = self.server.default_pcm
        self.ready = False
        self.send_lock = threading.Lock()
        if getattr(self.server, "keep_warm_s", 0.0) > 0.0:
            with engine_lock:
                self.server.clients += 1

    def finish(self):
        if getattr(self.server, "keep_warm_s", 0.0) > 0.0:
            with engine_lock:
                self.server.clients -= 1

    def handle(self):
        while True:
            try:
                msg_type, body, _ = wire.read_frame(self.request, idle_ok=True)
            except IdleTimeout:
                continue
            except (ClientGone, OSError):
                return
            except FrameError as e:
                self._send_error(e.code, e.message, e.fatal)
                if e.fatal:
                    return
                continue
            try:
                self._dispatch(msg_type, body)
            except FrameError as e:
                self._send_error(e.code, e.message, e.fatal)
                if e.fatal:
                    return
            except (ClientGone, OSError):
                return
            except Exception as e:
                self._send_error("internal", str(e), False)

    def _send(self, msg_type, body, pcm: bytes = b""):
        frame = wire.pack_frame(msg_type, body, pcm)
        with self.send_lock:
            self.request.sendall(frame)

    def _send_error(self, code, message, fatal):
        try:
            self._send(MsgType.ERROR, {"type": "error", "code": code, "message": message, "fatal": bool(fatal)})
        except OSError:
            pass

    def _dispatch(self, msg_type, body):
        if msg_type == MsgType.HELLO:
            return self._on_hello(body)
        if msg_type == MsgType.PING:
            return self._on_ping(body)
        if not self.ready:
            raise FrameError("bad_fields", "hello required first", fatal=False)
        if msg_type == MsgType.CONTEXT:
            return self._on_context(body)
        if msg_type == MsgType.EVENT:
            return self._on_event(body)
        if msg_type == MsgType.PREBAKE:
            return self._on_prebake(body)
        if msg_type == MsgType.PREDICT:
            return self._on_predict(body)
        if msg_type == MsgType.STATS:
            return self._on_stats(body)
        raise FrameError("unknown_type", f"unknown msg_type {msg_type}", fatal=False)

    def _float_field(self, body, name, default):
        try:
            return float(body.get(name, default))
        except (TypeError, ValueError):
            raise FrameError("bad_fields", f"{name} must be a number", fatal=False)

    def _on_hello(self, body):
        if not isinstance(body, dict):
            raise FrameError("bad_fields", "hello must be an object", fatal=False)
        proto = body.get("proto", body.get("protocol"))
        if proto != wire.PROTOCOL_VERSION:
            raise FrameError("bad_version", f"client proto {proto}", fatal=True)
        pcm = body.get("pcm", self.server.default_pcm)
        if pcm not in ("f32le", "i16le"):
            raise FrameError("bad_fields", f"unknown pcm {pcm}", fatal=True)
        self.pcm = pcm
        self.ready = True
        self._send(MsgType.WELCOME, {
            "type": "welcome",
            "proto": wire.PROTOCOL_VERSION,
            "server_version": __version__,
            "generator": self.server.generator_name,
            "sample_rate": self.server.sample_rate,
            "channels": 1,
            "pcm": self.pcm,
            "max_pcm_bytes": wire.MAX_PCM,
        })

    def _context_from(self, body):
        if not isinstance(body, dict):
            raise FrameError("bad_fields", "context must be an object", fatal=False)
        surfaces = body.get("surfaces", [])
        if not isinstance(surfaces, list) or not all(isinstance(s, str) for s in surfaces):
            raise FrameError("bad_fields", "surfaces must be list[str]", fatal=False)
        equipped = body.get("equipped")
        if equipped is not None and not isinstance(equipped, str):
            raise FrameError("bad_fields", "equipped must be str or null", fatal=False)
        region = body.get("region", "default")
        if not isinstance(region, str):
            raise FrameError("bad_fields", "region must be a str", fatal=False)
        return GameContext(self._float_field(body, "time_s", 0.0), equipped, list(surfaces), region)

    def _on_context(self, body):
        ctx = self._context_from(body)
        gpu = self._float_field(body, "gpu_pressure", 1.0)
        with engine_lock:
            self.server.engine.on_context_change(ctx, gpu, now=ctx.time_s)
            stats = dict(self.server.engine.stats)
            cached = len(self.server.engine.cache)
        self._send(MsgType.CONTEXT_ACK, {
            "type": "context_ack",
            "spec_gen": int(stats["spec_gen"]),
            "evicted": int(stats["evicted"]),
            "cached_keys": cached,
        })

    def _await_inflight(self, key):
        eng = self.server.engine
        deadline = time.monotonic() + INFLIGHT_WAIT_S
        while True:
            with engine_lock:
                busy = key not in eng.cache and any(k == key for k, _v in getattr(eng, "inflight", ()))
            if not busy or time.monotonic() >= deadline:
                return
            time.sleep(0.002)

    def _key_from(self, body):
        key = CacheKey(str(body["event_type"]), str(body["material"]),
                       str(body.get("intensity", "normal")), str(body.get("context", "default")))
        phys = body.get("phys")
        if not getattr(self.server, "world", False) or not isinstance(phys, dict):
            return key
        from ..world import bucket, intensity_of
        try:
            b = bucket(phys, key.event_type)
            inten = key.intensity if "intensity" in body else intensity_of(phys)
        except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError):
            return key
        return CacheKey(key.event_type, key.material, inten, key.context, b)

    def _note(self, key):
        srv = self.server
        if key.phys and getattr(srv, "world", False):
            srv.last_phys[(key.event_type, key.material, key.context)] = (key.phys, key.intensity)
        intent = self._update_intent(key)
        if getattr(srv, "predict_fixes", False):
            srv.fired_seq += 1
            srv.fired_at[key.event_type] = srv.fired_seq
            srv.last_intensity[key.event_type] = key.intensity
        return intent

    def _served(self, intent, clip):
        lrn = getattr(self.server.engine, "learner", None)
        learned = len(lrn.event_count) if lrn is not None else 0
        return clip, clip.samples, int(clip.variant_id), int(clip.sample_rate), intent, learned

    def _generate_unlocked(self, key, intent):
        eng = self.server.engine
        try:
            made = eng.gen.generate(key, variant=0)
        except BaseException as e:
            with engine_lock:
                eng.release_realtime(key)
            _wake_deepen(self.server)
            if isinstance(e, OSError):
                raise FrameError("internal", "generation failed: %s" % e, fatal=False)
            raise
        with engine_lock:
            served = eng.complete_realtime(key, made)
            out = (served,) + self._served(intent, made)
        _wake_deepen(self.server)
        return out

    def _on_event(self, body):
        t_recv = time.time()
        t0 = time.perf_counter()
        try:
            key = self._key_from(body)
        except (KeyError, TypeError):
            raise FrameError("bad_fields", "event needs event_type and material", fatal=False)
        self._await_inflight(key)
        t1 = time.perf_counter()
        eng = self.server.engine
        with engine_lock:
            t2 = time.perf_counter()
            claimed = getattr(self.server, "concurrent_gen", False) and eng.claim_realtime(key)
            if claimed:
                intent = self._note(key)
            else:
                try:
                    served = eng.on_event(key)
                except OSError as e:
                    raise FrameError("internal", "generation failed: %s" % e, fatal=False)
                clip = eng.last_served
                clip, samples, variant_id, sr, intent, learned = self._served(self._note(key), clip)
        if claimed:
            served, clip, samples, variant_id, sr, intent, learned = self._generate_unlocked(key, intent)
        t3 = time.perf_counter()
        if served == "realtime":
            self.server.last_gen_t = time.monotonic()
        have = body.get("have")
        meta_only = (isinstance(have, list) and served in ("cached", "realtime") and variant_id in have
                     and str(body.get("have_phys", "")) == key.phys)
        pcm = b"" if meta_only else wire.encode_pcm(samples, self.pcm)
        n = int(samples.shape[0])
        ch = 1 if samples.ndim == 1 else int(samples.shape[1])
        meta = {
            "type": "audio",
            "served": served,
            "variant_id": variant_id,
            "format": self.pcm,
            "sample_rate": sr,
            "channels": ch,
            "n_samples": n,
            "byte_len": len(pcm),
            "duration_s": (n / sr) if sr else 0.0,
            "intent": intent,
            "learned": learned,
        }
        if key.phys:
            meta["phys_key"] = key.phys
        if meta_only:
            meta["meta_only"] = True
        t4 = time.perf_counter()
        self._send(MsgType.AUDIO, meta, pcm)
        if getattr(self.server, "event_log", None) is not None:
            _log(self.server, {"type": "event", "t_recv": t_recv, "key": str(key), "served": served,
                               "variant": variant_id, "concurrent": bool(claimed),
                               "await_ms": (t1 - t0) * 1e3, "lock_wait_ms": (t2 - t1) * 1e3,
                               "engine_ms": (t3 - t2) * 1e3,
                               "gen_ms": clip.gen_ms if served == "realtime" else 0.0,
                               "encode_ms": (t4 - t3) * 1e3, "send_ms": (time.perf_counter() - t4) * 1e3,
                               "bytes": len(pcm), "meta_only": meta_only, "client": id(self)})

    def _update_intent(self, key):
        srv = self.server
        intent = getattr(srv, "intent", None)
        if intent is None or not hasattr(srv.engine, "register_predictions"):
            return getattr(srv, "last_intent", "")
        if key.event_type in INTENT_IGNORE:
            return srv.last_intent
        for pk, conf in srv.intent_pending:
            srv.calib.observe("intent", conf, pk == key)
        srv.intent_pending = []
        ctx = key.context
        intent.observe(key.event_type, ctx)
        srv.intent_last_material[key.event_type] = key.material
        pending = []
        raw = []
        for et, prob in intent.predict(ctx):
            pk = _with_phys(srv, CacheKey(et, srv.intent_last_material.get(et, key.material), "normal", ctx), True)
            raw.append(("intent", pk, srv.intent_lead_s, prob))
            pending.append((pk, prob))
        proposals = srv.fusion.fuse(raw)
        if proposals:
            srv.engine.register_predictions(proposals)
            _wake_deepen(srv)
            top = proposals[0]
            srv.last_intent = f"{top.key.event_type} {top.p_use:.2f}"
        srv.intent_pending = pending
        return srv.last_intent

    def _on_prebake(self, body):
        if not isinstance(body, dict):
            raise FrameError("bad_fields", "prebake must be an object", fatal=False)
        keys_spec = body.get("keys")
        if keys_spec is not None:
            if not isinstance(keys_spec, list):
                raise FrameError("bad_fields", "keys must be a list", fatal=False)
            keys = []
            prompts = {}
            for k in keys_spec:
                if not isinstance(k, dict) or "event_type" not in k or "material" not in k:
                    raise FrameError("bad_fields", "each key needs event_type and material", fatal=False)
                ck = CacheKey(str(k["event_type"]), str(k["material"]),
                              str(k.get("intensity", "normal")), str(k.get("context", "default")))
                keys.append(ck)
                if isinstance(k.get("prompt"), str) and k["prompt"]:
                    prompts[str(ck)] = k["prompt"]
            with engine_lock:
                try:
                    self.server.engine.prewarm(keys, prompts or None)
                except OSError as e:
                    raise FrameError("internal", "generation failed: %s" % e, fatal=False)
                stats = dict(self.server.engine.stats)
                cached = len(self.server.engine.cache)
            self._send(MsgType.PREBAKE_ACK, {"type": "prebake_ack", "spec_gen": int(stats["spec_gen"]),
                                             "peak_bytes": int(stats["peak_bytes"]), "cached_keys": cached})
            return
        contexts = body.get("contexts", [])
        if not isinstance(contexts, list):
            raise FrameError("bad_fields", "contexts must be a list", fatal=False)
        if len(contexts) > MAX_PREBAKE_CONTEXTS:
            raise FrameError("bad_fields", f"too many contexts (> {MAX_PREBAKE_CONTEXTS})", fatal=False)
        parsed = [(self._context_from(c), self._float_field(c, "gpu_pressure", 1.0)) for c in contexts]
        for ctx, gpu in parsed:
            with engine_lock:
                self.server.engine.on_context_change(ctx, gpu, now=ctx.time_s)
        with engine_lock:
            stats = dict(self.server.engine.stats)
            cached = len(self.server.engine.cache)
        self._send(MsgType.PREBAKE_ACK, {
            "type": "prebake_ack",
            "spec_gen": int(stats["spec_gen"]),
            "peak_bytes": int(stats["peak_bytes"]),
            "cached_keys": cached,
        })

    def _on_predict(self, body):
        if not isinstance(body, dict):
            raise FrameError("bad_fields", "predict must be an object", fatal=False)
        preds = body.get("predictions")
        if not isinstance(preds, list):
            raise FrameError("bad_fields", "predictions must be a list", fatal=False)
        now = self._float_field(body, "now", 0.0)
        t_recv = time.time()
        raw = []
        phys_keys = []
        for p in preds:
            if not isinstance(p, dict) or "event_type" not in p or "material" not in p:
                raise FrameError("bad_fields", "each prediction needs event_type and material", fatal=False)
            key = self._key_from(p)
            phys_keys.append(key.phys)
            try:
                eta = float(p.get("eta_s", 0.0))
                conf = float(p.get("confidence", 1.0))
            except (TypeError, ValueError):
                raise FrameError("bad_fields", "eta_s and confidence must be numbers", fatal=False)
            if getattr(self.server, "predict_fixes", False):
                with engine_lock:
                    fixed = [(_with_phys(self.server, k2), scale) for k2, scale in _fixed(self.server, key, p)]
                for k2, scale in fixed:
                    raw.append((str(p.get("source", "motion")), k2, eta, conf * scale))
            else:
                raw.append((str(p.get("source", "motion")), _with_phys(self.server, key), eta, conf))
        with engine_lock:
            proposals = self.server.fusion.fuse(raw)
            result = self.server.engine.register_predictions(proposals, now=now)
            cached = len(self.server.engine.cache)
        _wake_deepen(self.server)
        ack = {
            "type": "predict_ack",
            "queued": int(result["queued"]),
            "lead_gated": int(result["lead_gated"]),
            "cached_keys": cached,
        }
        if any(phys_keys):
            ack["phys_keys"] = phys_keys
        self._send(MsgType.PREDICT_ACK, ack)
        if getattr(self.server, "event_log", None) is not None:
            _log(self.server, {"type": "predict", "t_recv": t_recv, "keys": [str(r[1]) for r in raw],
                               "eta_s": [r[2] for r in raw], "queued": ack["queued"],
                               "lead_gated": ack["lead_gated"], "client": id(self)})

    def _on_stats(self, body):
        with engine_lock:
            stats = dict(self.server.engine.stats)
            wasted = int(self.server.engine.wasted())
            cached = len(self.server.engine.cache)
        self._send(MsgType.STATS_REPLY, {"type": "stats_reply", "stats": stats, "wasted": wasted, "cached_keys": cached})

    def _on_ping(self, body):
        t = body.get("t") if isinstance(body, dict) else None
        self._send(MsgType.PONG, {"type": "pong", "t": t, "server_t": time.time()})


def _log(server, rec):
    line = json.dumps(rec) + "\n"
    with server.event_log_lock:
        f = server.event_log
        if f is None:
            return
        try:
            f.write(line)
            f.flush()
        except (OSError, ValueError):
            server.event_log = None


def _with_phys(server, key, intensity=False):
    if key.phys or not getattr(server, "world", False):
        return key
    hit = server.last_phys.get((key.event_type, key.material, key.context))
    if hit is None:
        return key
    return CacheKey(key.event_type, key.material, hit[1] if intensity else key.intensity, key.context, hit[0])


def _alias(server, event_type):
    if not event_type:
        return event_type
    pat = re.compile(r"(^|\s)" + re.escape(event_type) + r"(s|es|ed|ing)?$")
    best, seq = event_type, server.fired_at.get(event_type, 0)
    for f, n in server.fired_at.items():
        if n > seq and pat.search(f):
            best, seq = f, n
    return best


def _successors(server, after, context, top=3):
    counts = {}
    for (ctx, hist), d in getattr(server.intent, "ngram", {}).items():
        if ctx == context and hist and hist[-1] == after:
            for e, n in d.items():
                counts[e] = counts.get(e, 0) + n
    mats = server.intent_last_material
    keep = [(e, n) for e, n in counts.items() if e != after and mats.get(e) != mats.get(after)]
    total = float(sum(n for _e, n in keep))
    keep.sort(key=lambda t: t[1], reverse=True)
    return [(e, n / total) for e, n in keep[:top]] if total else []


def _fixed(server, key, p):
    after = p.get("after")
    keys = []
    if isinstance(after, str) and after:
        keys = [(CacheKey(e, key.material, key.intensity, key.context, key.phys), prob)
                for e, prob in _successors(server, after, key.context)]
    if not keys:
        keys = [(key, 1.0)]
    given = "intensity" in p
    out = []
    for k, scale in keys:
        et = _alias(server, k.event_type)
        inten = k.intensity if k.phys or given else server.last_intensity.get(et, k.intensity)
        out.append((CacheKey(et, k.material, inten, k.context, k.phys), scale))
    return out


def _wake_deepen(server):
    ev = getattr(server, "deepen_wake", None)
    if ev is not None:
        ev.set()


def _idle(server, seconds):
    ev = getattr(server, "deepen_wake", None)
    if ev is None:
        time.sleep(seconds)
        return
    ev.wait(seconds)


def _deepen_loop(server):
    import time
    eng = server.engine
    planner = getattr(eng, "plan_batch", None)
    while getattr(server, "deepening", False):
        ev = getattr(server, "deepen_wake", None)
        if ev is not None:
            ev.clear()
        batch = []
        stat = "spec_gen"
        try:
            if planner is None:
                with engine_lock:
                    did = eng.deepen_step()
                _idle(server, 0.05 if did else 0.5)
                continue
            with engine_lock:
                held = bool(getattr(eng, "claims", None))
                if not held:
                    batch, stat = planner(max_items=getattr(server, "deepen_batch", 1))
            if held:
                _idle(server, 0.5)
                continue
            if batch:
                tg = time.perf_counter()
                clips = eng.gen.generate_batch(batch)
                with engine_lock:
                    eng.commit_clips(batch, clips, stat)
                server.last_gen_t = time.monotonic()
                if getattr(server, "event_log", None) is not None:
                    _log(server, {"type": "deepen", "t": time.time(), "stat": stat,
                                  "items": [[str(k), v] for k, v in batch],
                                  "gen_ms": [getattr(c, "gen_ms", None) for c in clips],
                                  "ms": (time.perf_counter() - tg) * 1e3})
        except Exception:
            if batch:
                try:
                    with engine_lock:
                        eng.inflight.difference_update(batch)
                except Exception:
                    pass
            batch = []
        if batch and getattr(server, "predict_fixes", False) and eng.demand:
            continue
        _idle(server, 0.05 if batch else 0.5)


def _keep_warm_loop(server):
    inner = getattr(server.engine.gen, "inner", server.engine.gen)
    key = CacheKey("warmup", "grass", "normal")
    while getattr(server, "keep_warm_s", 0.0) > 0.0:
        time.sleep(server.keep_warm_s / 2.0)
        with engine_lock:
            idle = (server.clients > 0 and not getattr(server.engine, "claims", None)
                    and time.monotonic() - server.last_gen_t >= server.keep_warm_s)
        if not idle:
            continue
        t = time.perf_counter()
        try:
            inner.generate(key, variant=0)
        except Exception:
            pass
        server.last_gen_t = time.monotonic()
        server.warm_jobs += 1
        if getattr(server, "event_log", None) is not None:
            _log(server, {"type": "keep_warm", "t": time.time(), "ms": (time.perf_counter() - t) * 1e3})


def serve(engine=None, generator: str = "mock", host: str = "127.0.0.1", port: int = wire.DEFAULT_PORT,
          pcm: str = "f32le", recv_timeout: float = 30.0, warmup: bool = True,
          cache_dir: str | None = None, deepen: bool = False, gen_kwargs: dict | None = None,
          learn: bool = False, realtime: bool = False, deepen_batch: int = 1, world: bool = False,
          event_log: str | None = None, predict_fixes: bool = False, pool_fix: bool = False,
          concurrent_gen: bool = False, keep_warm_ms: float = 0.0, sibling_on_miss: bool = False,
          cache_async: bool = False, **engine_kwargs) -> SfxServer:
    if engine is None:
        gen = get_generator(generator, **(gen_kwargs or {}))
        if cache_dir:
            from ..generators.cached import CachedGenerator
            gen = CachedGenerator(gen, cache_dir, async_write=cache_async)
        if learn:
            from ..affordance_learn import RuntimeAffordanceLearner
            from ..router import EventRouter
            engine_kwargs.setdefault("learner", RuntimeAffordanceLearner())
            engine_kwargs.setdefault("router", EventRouter())
        if realtime:
            engine_kwargs["realtime"] = True
            engine_kwargs.setdefault("authored", set())
        engine = AffordanceEngine(gen, **engine_kwargs)
    if warmup:
        try:
            t = time.time()
            ready = getattr(engine.gen, "ensure_ready", None)
            if ready:
                ready()
            t = stage("generator ready", t)
            inner = getattr(engine.gen, "inner", engine.gen)
            inner.generate(CacheKey("warmup", "grass", "normal"), variant=0)
            stage("warmup request", t)
            if deepen and int(deepen_batch) > 1:
                inner.generate_batch([(CacheKey("warmup", "grass", "normal"), v) for v in range(int(deepen_batch))])
        except Exception:
            pass
    server = SfxServer((host, port), engine, default_pcm=pcm, recv_timeout=recv_timeout)
    server.world = bool(world)
    server.last_phys = {}
    server.predict_fixes = bool(predict_fixes)
    server.concurrent_gen = bool(concurrent_gen)
    server.fired_at = {}
    server.fired_seq = 0
    server.last_intensity = {}
    server.event_log_lock = threading.Lock()
    server.event_log = open(event_log, "a", encoding="utf-8") if event_log else None
    if pool_fix:
        engine.pool_fix = True
    if sibling_on_miss:
        engine.sibling_on_miss = True
    server.clients = 0
    server.last_gen_t = time.monotonic()
    server.warm_jobs = 0
    server.keep_warm_s = max(0.0, float(keep_warm_ms or 0.0)) / 1000.0
    if server.keep_warm_s > 0.0:
        threading.Thread(target=_keep_warm_loop, args=(server,), name="sfx-keep-warm", daemon=True).start()
    if predict_fixes and getattr(engine, "realtime", False):
        engine.lead_gate = False
    server.calib = PredictorCalibration()
    server.fusion = PredictionFusion(
        calibration=server.calib,
        gen_cost_ms=getattr(engine, "gen_cost_ms", 393.0),
        latency_saved_ms=getattr(engine, "latency_saved_ms", 400.0),
        owner={"footstep": "motion"})
    server.intent = EventSequencePredictor()
    server.intent_last_material = {}
    server.intent_pending = []
    server.intent_lead_s = 0.6
    server.last_intent = ""
    thread = threading.Thread(target=server.serve_forever, name="sfx-sidecar", daemon=True)
    thread.start()
    server.runner = thread
    server.deepen_batch = max(1, int(deepen_batch))
    if deepen:
        server.deepening = True
        server.deepen_wake = threading.Event()
        dthread = threading.Thread(target=_deepen_loop, args=(server,), name="sfx-deepener", daemon=True)
        dthread.start()
        server.deepener = dthread
    return server
