from __future__ import annotations

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

    def _on_event(self, body):
        try:
            event_type = str(body["event_type"])
            material = str(body["material"])
        except (KeyError, TypeError):
            raise FrameError("bad_fields", "event needs event_type and material", fatal=False)
        key = CacheKey(event_type, material, str(body.get("intensity", "normal")), str(body.get("context", "default")))
        self._await_inflight(key)
        with engine_lock:
            served = self.server.engine.on_event(key)
            clip = self.server.engine.last_served
            samples = clip.samples
            variant_id = int(clip.variant_id)
            sr = int(clip.sample_rate)
            intent = self._update_intent(key)
            lrn = getattr(self.server.engine, "learner", None)
            learned = len(lrn.event_count) if lrn is not None else 0
        pcm = wire.encode_pcm(samples, self.pcm)
        n = int(samples.shape[0])
        self._send(MsgType.AUDIO, {
            "type": "audio",
            "served": served,
            "variant_id": variant_id,
            "format": self.pcm,
            "sample_rate": sr,
            "channels": 1,
            "n_samples": n,
            "byte_len": len(pcm),
            "duration_s": (n / sr) if sr else 0.0,
            "intent": intent,
            "learned": learned,
        }, pcm)

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
            pk = CacheKey(et, srv.intent_last_material.get(et, key.material), "normal", ctx)
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
                self.server.engine.prewarm(keys, prompts or None)
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
        raw = []
        for p in preds:
            if not isinstance(p, dict) or "event_type" not in p or "material" not in p:
                raise FrameError("bad_fields", "each prediction needs event_type and material", fatal=False)
            key = CacheKey(str(p["event_type"]), str(p["material"]),
                           str(p.get("intensity", "normal")), str(p.get("context", "default")))
            try:
                eta = float(p.get("eta_s", 0.0))
                conf = float(p.get("confidence", 1.0))
            except (TypeError, ValueError):
                raise FrameError("bad_fields", "eta_s and confidence must be numbers", fatal=False)
            raw.append((str(p.get("source", "motion")), key, eta, conf))
        with engine_lock:
            proposals = self.server.fusion.fuse(raw)
            result = self.server.engine.register_predictions(proposals, now=now)
            cached = len(self.server.engine.cache)
        _wake_deepen(self.server)
        self._send(MsgType.PREDICT_ACK, {
            "type": "predict_ack",
            "queued": int(result["queued"]),
            "lead_gated": int(result["lead_gated"]),
            "cached_keys": cached,
        })

    def _on_stats(self, body):
        with engine_lock:
            stats = dict(self.server.engine.stats)
            wasted = int(self.server.engine.wasted())
            cached = len(self.server.engine.cache)
        self._send(MsgType.STATS_REPLY, {"type": "stats_reply", "stats": stats, "wasted": wasted, "cached_keys": cached})

    def _on_ping(self, body):
        t = body.get("t") if isinstance(body, dict) else None
        self._send(MsgType.PONG, {"type": "pong", "t": t, "server_t": time.time()})


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
                batch, stat = planner(max_items=getattr(server, "deepen_batch", 1))
            if batch:
                clips = eng.gen.generate_batch(batch)
                with engine_lock:
                    eng.commit_clips(batch, clips, stat)
        except Exception:
            if batch:
                try:
                    with engine_lock:
                        eng.inflight.difference_update(batch)
                except Exception:
                    pass
            batch = []
        _idle(server, 0.05 if batch else 0.5)


def serve(engine=None, generator: str = "mock", host: str = "127.0.0.1", port: int = wire.DEFAULT_PORT,
          pcm: str = "f32le", recv_timeout: float = 30.0, warmup: bool = True,
          cache_dir: str | None = None, deepen: bool = False, gen_kwargs: dict | None = None,
          learn: bool = False, realtime: bool = False, deepen_batch: int = 1,
          **engine_kwargs) -> SfxServer:
    if engine is None:
        gen = get_generator(generator, **(gen_kwargs or {}))
        if cache_dir:
            from ..generators.cached import CachedGenerator
            gen = CachedGenerator(gen, cache_dir)
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
            ready = getattr(engine.gen, "ensure_ready", None)
            if ready:
                ready()
            inner = getattr(engine.gen, "inner", engine.gen)
            inner.generate(CacheKey("warmup", "grass", "normal"), variant=0)
            if deepen and int(deepen_batch) > 1:
                inner.generate_batch([(CacheKey("warmup", "grass", "normal"), v) for v in range(int(deepen_batch))])
        except Exception:
            pass
    server = SfxServer((host, port), engine, default_pcm=pcm, recv_timeout=recv_timeout)
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
