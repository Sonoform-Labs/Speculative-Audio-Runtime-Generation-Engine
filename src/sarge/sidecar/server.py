from __future__ import annotations

import collections
import contextlib
import json
import socket
import socketserver
import statistics
import threading
import time
from dataclasses import replace

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
from ..words import names, split_words

engine_lock = threading.Lock()
INFLIGHT_WAIT_S = 0.5

MAX_PREBAKE_CONTEXTS = 256
MAX_MANIFEST_KEYS = 4096
RAPID_HZ = 1.5
MOTION_SOURCE = "motion"
KIND_WINDOW_S = 10.0
KIND_DECAY = 0.8
UPGRADE_DELAY_S = 10.0
UPGRADE_IDLE_S = 1.0
UPGRADE_POLL_S = 2.0
UPGRADE_RETRIES = 2
UPGRADE_BACKOFF_S = 5.0
UPGRADE_UNREACH_MAX_S = 60.0
UPGRADE_LINE_S = 15.0
FEED_MAX = 256
FEED_REPLY_MAX = 64
TRACE_PATHS = {"cached": "cache", "realtime": "render"}
GEN_LABELS = {"stableaudio3": "sa3", "stableaudio": "sa1", "remote": "woosh"}


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
        self.trace = False
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
        if msg_type == MsgType.MANIFEST:
            return self._on_manifest(body)
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
        self.trace = body.get("trace") is True and getattr(self.server, "trace", False)
        welcome = {
            "type": "welcome",
            "proto": wire.PROTOCOL_VERSION,
            "server_version": __version__,
            "generator": self.server.generator_name,
            "sample_rate": self.server.sample_rate,
            "channels": 1,
            "pcm": self.pcm,
            "max_pcm_bytes": wire.MAX_PCM,
        }
        if self.trace:
            welcome["trace"] = True
        self._send(MsgType.WELCOME, welcome)

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
        if getattr(self.server, "read_class", False):
            key = replace(key, cls=str(body.get("class") or "").strip().lower())
        phys = body.get("phys")
        if getattr(self.server, "world", False) and isinstance(phys, dict):
            from ..world import bucket, intensity_of
            try:
                b = bucket(phys, key.event_type)
                inten = key.intensity if "intensity" in body else intensity_of(phys)
            except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError):
                b = None
            if b is not None:
                key = replace(key, intensity=inten, phys=b)
        if getattr(self.server, "scene", False) and isinstance(body.get("scene"), dict):
            from ..scene import quantise
            s = quantise(body["scene"], phys if isinstance(phys, dict) else None, key.event_type)
            if s:
                key = replace(key, scene=s)
        _register_prompt(self.server, key, body.get("prompt"))
        return key

    def _note(self, key):
        srv = self.server
        if key.phys and getattr(srv, "world", False):
            srv.last_phys[(key.event_type, key.material, key.context)] = (key.phys, key.intensity)
        if key.scene and getattr(srv, "scene", False):
            srv.last_scene[(key.event_type, key.material, key.context)] = key.scene
        rapid = _rapid(srv, key.event_type)
        if getattr(srv, "read_class", False):
            srv.last_class[key.event_type] = key.cls
        _confirm(srv, key, rapid)
        intent = self._update_intent(key, rapid)
        srv.fired_seq += 1
        srv.fired_at[key.event_type] = srv.fired_seq
        if getattr(srv, "predict_fixes", False):
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
        if _writes(self.server):
            _writer_seen(self.server, key)
        self._await_inflight(key)
        t1 = time.perf_counter()
        eng = self.server.engine
        with engine_lock:
            t2 = time.perf_counter()
            if getattr(self.server, "upgrade", None):
                _upgrade_sync(self.server, key)
                _sa3_sync(self.server, key)
            rev = _rev(self.server, key)
            predicted = self.trace and key in (getattr(eng, "eta", None) or ())
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
        meta_only =(isinstance(have, list) and served in ("cached", "realtime") and variant_id in have
                     and str(body.get("have_phys", "")) == key.phys
                     and str(body.get("have_scene", "")) == key.scene and _have_rev(body) == rev)
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
        if key.scene:
            meta["scene_key"] = key.scene
        if rev:
            meta["prompt_rev"] = rev
        if meta_only:
            meta["meta_only"] = True
        if self.trace:
            meta["trace"] = _trace(self.server, key, served, clip, variant_id, meta_only, predicted)
        t4 = time.perf_counter()
        self._send(MsgType.AUDIO, meta, pcm)
        up = getattr(self.server, "upgrade", None)
        if up:
            self.server.upgrade_first.setdefault(key, time.monotonic())
        if getattr(self.server, "event_log", None) is not None:
            rec = {"type": "event", "t_recv": t_recv, "key": str(key), "served": served,
                   "variant": variant_id, "concurrent": bool(claimed),
                   "await_ms": (t1 - t0) * 1e3, "lock_wait_ms": (t2 - t1) * 1e3,
                   "engine_ms": (t3 - t2) * 1e3,
                   "gen_ms": clip.gen_ms if served == "realtime" else 0.0,
                   "encode_ms": (t4 - t3) * 1e3, "send_ms": (time.perf_counter() - t4) * 1e3,
                   "bytes": len(pcm), "meta_only": meta_only, "client": id(self)}
            if up:
                rec["upgraded"] = (self.server.upgrade_done.get(key, {}).get(variant_id) is clip
                                   or _sa3_served(self.server, key, variant_id, clip))
                rec["rev"] = rev
            _log(self.server, rec)

    def _update_intent(self, key, rapid=False):
        srv = self.server
        intent = getattr(srv, "intent", None)
        if intent is None or not hasattr(srv.engine, "register_predictions"):
            return getattr(srv, "last_intent", "")
        if rapid:
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
            pk = _with_phys(srv, CacheKey(et, srv.intent_last_material.get(et, key.material), "normal", ctx,
                                          cls=_cls(srv, et)), True)
            raw.append(("intent", pk, srv.intent_lead_s, prob))
            pending.append((pk, prob))
            if _writes(srv):
                _writer_want(srv, pk, "predict", srv.intent_lead_s)
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
                if getattr(self.server, "read_class", False):
                    ck = replace(ck, cls=str(k.get("class") or "").strip().lower())
                keys.append(ck)
                if isinstance(k.get("prompt"), str) and k["prompt"]:
                    prompts[str(ck)] = k["prompt"]
            w = getattr(self.server, "writer", None)
            if w is not None:
                with engine_lock:
                    for k in prompts:
                        self.server.writer_owned.discard(k)
            if _writes(self.server):
                for ck in keys:
                    if str(ck) not in prompts:
                        _writer_want(self.server, ck, "manifest")
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
        seen = {}
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
            src = str(p.get("source", MOTION_SOURCE))
            pend = seen.setdefault(src, (set(), set()))
            pend[0].add((key.material, key.context))
            if isinstance(p.get("after"), str) and p["after"]:
                pend[1].add(p["after"])
            if getattr(self.server, "predict_fixes", False):
                with engine_lock:
                    fixed = [(_with_phys(self.server, k2), scale) for k2, scale in _fixed(self.server, key, p, src)]
                for k2, scale in fixed:
                    raw.append((src, k2, eta, conf * scale))
            else:
                with engine_lock:
                    et = key.event_type or _kind(self.server, src)
                if et:
                    raw.append((src, _with_phys(self.server, replace(key, event_type=et)), eta, conf))
        if _writes(self.server):
            for _src, k2, eta, _conf in raw:
                _writer_want(self.server, k2, "predict", eta)
        with engine_lock:
            _pend(self.server, seen)
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

    def _on_manifest(self, body):
        if not isinstance(body, dict) or not isinstance(body.get("keys"), list):
            raise FrameError("bad_fields", "manifest needs a keys list", fatal=False)
        keys = body["keys"]
        if len(keys) > MAX_MANIFEST_KEYS:
            raise FrameError("bad_fields", f"too many manifest keys (> {MAX_MANIFEST_KEYS})", fatal=False)
        parsed = []
        for k in keys:
            if not isinstance(k, dict) or "event_type" not in k or "material" not in k:
                raise FrameError("bad_fields", "each manifest key needs event_type and material", fatal=False)
            parsed.append(self._key_from(k))
        w = getattr(self.server, "writer", None)
        counts = {"known": 0, "queued": 0}
        if _writes(self.server):
            for key in parsed:
                got = _writer_want(self.server, key, "manifest")
                if got in counts:
                    counts[got] += 1
        self._send(MsgType.MANIFEST_ACK, {"type": "manifest_ack", "keys": len(parsed), "known": counts["known"],
                                          "queued": counts["queued"], "writer": w is not None})

    def _on_stats(self, body):
        with engine_lock:
            stats = dict(self.server.engine.stats)
            wasted = int(self.server.engine.wasted())
            cached = len(self.server.engine.cache)
            pending = len(self.server.writer_pending)
        reply = {"type": "stats_reply", "stats": stats, "wasted": wasted, "cached_keys": cached}
        if getattr(self.server, "upgrade", None):
            with engine_lock:
                reply["upgrade"] = _upgrade_summary(self.server)
        w = getattr(self.server, "writer", None)
        if w is not None:
            reply["writer"] = dict(w.stats, state=w.state, id=w.id[:12], stored=len(w.store or ()),
                                   swaps_pending=pending)
        if isinstance(body, dict) and "feed" in body and getattr(self.server, "feed", None) is not None:
            try:
                since = int(body.get("feed") or 0)
            except (TypeError, ValueError):
                raise FrameError("bad_fields", "feed must be an int", fatal=False)
            reply["feed"], reply["feed_seq"] = _feed_since(self.server, since)
        self._send(MsgType.STATS_REPLY, reply)

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


def _inner_name(gen):
    g = gen
    for _ in range(8):
        d = getattr(g, "__dict__", None) or {}
        nxt = d.get("_g") if d.get("_g") is not None else d.get("inner")
        if nxt is None:
            break
        g = nxt
    name = str(getattr(g, "name", "") or "")
    return GEN_LABELS.get(name, name)


def _gen_label(server, g, route, take=""):
    if take == "flow":
        return "woosh_flow"
    if g is None:
        return _inner_name(server.engine.gen)
    if route == "guns":
        return "sa3"
    model = str(getattr(getattr(g, "rest", None), "model", "") or "")
    return model.replace("-", "_") if model.startswith("woosh") else "woosh"


def _prompt_source(server, key):
    pm = _prompts(server)
    if pm is None:
        return ""
    k = str(key)
    if k not in pm:
        return "template"
    owned = getattr(server, "writer_owned", None)
    return "writer" if owned is not None and k in owned else "game"


def _take_of(server, key, variant_id, clip):
    if getattr(server, "upgrade", None) and server.upgrade_done.get(key, {}).get(variant_id) is clip:
        return "flow"
    if _sa3_served(server, key, variant_id, clip):
        return "best"
    return ""


def _trace_key(server, key, take=""):
    g = _routed(server)
    route = ""
    if g is not None:
        made = getattr(g, "made", None)
        route = (made.get(key) if isinstance(made, dict) else None) or g.route(key)
    return {"route": route, "gen": _gen_label(server, g, route, take), "prompt": _prompt_source(server, key),
            "phys": bool(key.phys), "scene": bool(getattr(key, "scene", ""))}


def _trace(server, key, served, clip, variant_id, meta_only, predicted):
    path = "meta" if meta_only else TRACE_PATHS.get(served, served)
    out = {"path": path, "route": "", "gen": "", "prompt": "", "take": "", "phys": bool(key.phys),
           "scene": bool(getattr(key, "scene", "")), "predicted": bool(predicted)}
    try:
        take = _take_of(server, key, variant_id, clip)
        out.update(_trace_key(server, key, take), take=take)
        if served in ("fallback", "silence"):
            out.update(route="", gen=served)
    except Exception:
        pass
    return out


def _feed(server, kind, key=None, **extra):
    q = getattr(server, "feed", None)
    if q is None:
        return
    rec = {"kind": kind}
    if key is not None:
        rec["key"] = "%s|%s" % (key.event_type, key.material)
        try:
            rec.update(_trace_key(server, key, extra.get("take", "")))
        except Exception:
            pass
    rec.update(extra)
    with server.feed_lock:
        server.feed_seq += 1
        rec["seq"] = server.feed_seq
        q.append(rec)


def _feed_since(server, since):
    with server.feed_lock:
        seq = server.feed_seq
        if since > seq or since < 0:
            since = 0
        items = [dict(r) for r in server.feed if r["seq"] > since]
    return items[-FEED_REPLY_MAX:], seq


def _register_prompt(server, key, prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        return
    eng = server.engine
    g = getattr(eng.gen, "inner", eng.gen)
    pm = getattr(g, "prompts", None)
    if not isinstance(pm, dict):
        return
    k = str(key)
    owned = getattr(server, "writer_owned", None)
    with engine_lock:
        if owned is not None:
            owned.discard(k)
            server.writer_pending.pop(key, None)
        if pm.get(k) == prompt:
            return
        had = k in pm
        pm[k] = prompt
        if had or key in eng.cache:
            eng.cache.pop(key, None)
            eng.recent.pop(key, None)
            _bump_rev(server, key)
            if getattr(server, "upgrade", None):
                _upgrade_drop(server, key)
                _sa3_drop(server, key)


def _rapid(server, event_type):
    hz = float(getattr(server, "rapid_hz", 0.0) or 0.0)
    now = server.clock()
    prev = server.last_fire_t.get(event_type)
    server.last_fire_t[event_type] = now
    return hz > 0.0 and prev is not None and now - prev < 1.0 / hz


def _pend(server, seen):
    now = server.clock()
    for src, (pairs, triggers) in seen.items():
        server.kind_pending[src] = (pairs, triggers, now)


def _confirm(server, key, rapid):
    now = server.clock()
    pair = (key.material, key.context)
    for src in sorted(server.kind_pending, key=lambda s: s != MOTION_SOURCE):
        pairs, triggers, t = server.kind_pending[src]
        if now - t > KIND_WINDOW_S:
            del server.kind_pending[src]
            continue
        if pair not in pairs or key.event_type in triggers:
            continue
        if src != MOTION_SOURCE and (rapid or key.event_type == _kind(server, MOTION_SOURCE)):
            continue
        score = server.kind_score.setdefault(src, {})
        for e in score:
            score[e] *= KIND_DECAY
        score[key.event_type] = score.get(key.event_type, 0.0) + 1.0
        server.kind_targets.setdefault(src, {}).setdefault(key.event_type, set()).add(pair)


def _kind(server, source):
    score = getattr(server, "kind_score", {}).get(source)
    if not score:
        return ""
    targets = server.kind_targets.get(source, {})
    return max(score, key=lambda e: (len(targets.get(e, ())), score[e], server.fired_at.get(e, 0)))


def _cls(server, event_type, given=""):
    if not getattr(server, "read_class", False):
        return ""
    return given or server.last_class.get(event_type, "")


def _with_phys(server, key, intensity=False):
    if getattr(server, "scene", False) and not key.scene:
        s = server.last_scene.get((key.event_type, key.material, key.context))
        if s:
            key = replace(key, scene=s)
    if key.phys or not getattr(server, "world", False):
        return key
    hit = server.last_phys.get((key.event_type, key.material, key.context))
    if hit is None:
        return key
    return replace(key, intensity=hit[1] if intensity else key.intensity, phys=hit[0])


def _alias(server, event_type):
    if not event_type:
        return event_type
    stem = split_words(event_type)
    best, seq = event_type, server.fired_at.get(event_type, 0)
    for f, n in server.fired_at.items():
        if n > seq and names(stem, split_words(f)):
            best, seq = f, n
    return best


def _successors(server, after, context, source, top=3):
    counts = {}
    for (ctx, hist), d in getattr(server.intent, "ngram", {}).items():
        if ctx == context and hist and hist[-1] == after:
            for e, n in d.items():
                counts[e] = counts.get(e, 0) + n
    landed = getattr(server, "kind_targets", {}).get(source, {})
    keep = [(e, n) for e, n in counts.items() if e != after and len(landed.get(e, ())) >= 2]
    total = float(sum(n for _e, n in keep))
    keep.sort(key=lambda t: t[1], reverse=True)
    return [(e, n / total) for e, n in keep[:top]] if total else []


def _fixed(server, key, p, source=MOTION_SOURCE):
    after = p.get("after")
    keys = []
    if isinstance(after, str) and after:
        keys = [(replace(key, event_type=e, cls=_cls(server, e)), prob)
                for e, prob in _successors(server, after, key.context, source)]
    if not keys:
        et = key.event_type or _kind(server, source)
        if not et:
            return []
        keys = [(replace(key, event_type=et, cls=_cls(server, et, key.cls)), 1.0)]
    given = "intensity" in p
    out = []
    for k, scale in keys:
        et = _alias(server, k.event_type)
        inten = k.intensity if k.phys or given else server.last_intensity.get(et, k.intensity)
        out.append((replace(k, event_type=et, intensity=inten, cls=_cls(server, et, k.cls)), scale))
    return out


def _wake_deepen(server):
    ev = getattr(server, "deepen_wake", None)
    if ev is not None:
        ev.set()


def _prompts(server):
    eng = server.engine
    g = getattr(eng.gen, "inner", eng.gen)
    pm = getattr(g, "prompts", None)
    return pm if isinstance(pm, dict) else None


def _writer_route(server, key):
    eng = server.engine
    g = getattr(eng.gen, "inner", eng.gen)
    route = getattr(g, "route", None)
    if callable(route):
        try:
            return "sa3" if route(key) == "guns" else "woosh"
        except Exception:
            return "woosh"
    return "sa3" if str(getattr(g, "name", "")).startswith("stableaudio") else "woosh"


def _rev(server, key):
    revs = getattr(server, "prompt_rev", None)
    return int(revs.get(key, 0)) if revs else 0


def _have_rev(body):
    try:
        return int(body.get("have_rev", 0) or 0)
    except (TypeError, ValueError):
        return -1


def _bump_rev(server, key):
    revs = getattr(server, "prompt_rev", None)
    if revs is not None:
        revs[key] = revs.get(key, 0) + 1


def _busy(eng, key):
    return bool(eng.cache.get(key)) or key in eng.claims or any(k == key for k, _v in eng.inflight)


def _writer_set(server, key, prompt, evict):
    eng = server.engine
    pm = _prompts(server)
    k = str(key)
    pm[k] = prompt
    server.writer_owned.add(k)
    server.writer_pending.pop(key, None)
    if evict and eng.cache.get(key):
        eng.cache.pop(key, None)
        eng.recent.pop(key, None)
        _bump_rev(server, key)


def _writer_deliver(server, key, rec):
    prompt = rec.get(_writer_route(server, key)) if isinstance(rec, dict) else None
    pm = _prompts(server)
    if not prompt or pm is None:
        return False
    k = str(key)
    if pm.get(k) == prompt:
        return False
    eng = server.engine
    with engine_lock:
        cur = pm.get(k)
        if cur == prompt or (cur is not None and k not in server.writer_owned):
            return False
        if not _busy(eng, key):
            _writer_set(server, key, prompt, False)
            return True
        server.writer_pending[key] = prompt
    _wake_deepen(server)
    return True


def _writes(server):
    return getattr(server, "writer", None) is not None or getattr(server, "prompt_pool", None) is not None


def _writer_want(server, key, why, eta=None):
    pool = getattr(server, "prompt_pool", None)
    if pool is not None:
        pool.want(key, why, eta)
    w = getattr(server, "writer", None)
    if w is None:
        return None
    pm = _prompts(server)
    k = str(key)
    if pm is not None and k in pm and k not in server.writer_owned:
        return "game"
    return w.want(key, why, eta)


def _writer_seen(server, key):
    _writer_want(server, key, "seen")
    if key not in server.writer_pending:
        return
    if getattr(server, "deepening", False) and server.writer_swap == "revalidate":
        return
    eng = server.engine
    with engine_lock:
        prompt = server.writer_pending.get(key)
        if prompt is None or key in eng.claims or any(k == key for k, _v in eng.inflight):
            return
        cur = _prompts(server).get(str(key))
        if cur is not None and str(key) not in server.writer_owned:
            server.writer_pending.pop(key, None)
            return
        _writer_set(server, key, prompt, True)


def _accepts_prompt(gen):
    import inspect
    try:
        return "prompt" in inspect.signature(gen.generate).parameters
    except (TypeError, ValueError):
        return False


def _writer_step(server):
    eng = server.engine
    pm = _prompts(server)
    with engine_lock:
        if eng.claims or not server.writer_pending or pm is None:
            return False
        pick = None
        for key, prompt in list(server.writer_pending.items()):
            if not any(k == key for k, _v in eng.inflight):
                pick = (key, prompt)
                break
        if pick is None:
            return False
        key, prompt = pick
        k = str(key)
        cur = pm.get(k)
        if cur is not None and k not in server.writer_owned:
            server.writer_pending.pop(key, None)
            return False
        if (server.writer_swap != "revalidate" or not eng.cache.get(key)
                or not _accepts_prompt(eng.gen)):
            _writer_set(server, key, prompt, server.writer_swap != "revalidate" or not _accepts_prompt(eng.gen))
            return False
        n_old = len(eng.cache.get(key, ()))
        server.writer_rendering = key
    t = time.perf_counter()
    try:
        clip = eng.gen.generate(key, variant=0, prompt=prompt)
    except Exception:
        with engine_lock:
            if server.writer_pending.get(key) == prompt:
                _writer_set(server, key, prompt, True)
        return True
    finally:
        server.writer_rendering = None
    server.last_gen_t = time.monotonic()
    ms = (time.perf_counter() - t) * 1e3
    with engine_lock:
        if server.writer_pending.get(key) != prompt:
            return True
        if key in eng.claims or any(kk == key for kk, _v in eng.inflight):
            return True
        cur = pm.get(k)
        if cur is not None and k not in server.writer_owned:
            server.writer_pending.pop(key, None)
            return True
        _writer_set(server, key, prompt, False)
        if eng.cache.get(key):
            eng.cache[key] = [clip]
            eng.recent.pop(key, None)
            _bump_rev(server, key)
        else:
            eng.cache[key] = [clip]
        eng.generated.add(key)
        eng.stats["writer_swaps"] = eng.stats.get("writer_swaps", 0) + 1
        eng.stats["gpu_ms"] += eng.cost_ms(key)
        if n_old > 1:
            server.writer_fill[key] = n_old
    _feed(server, "writer_swap", key, ms=round(ms, 1))
    if getattr(server, "event_log", None) is not None:
        _log(server, {"type": "writer_swap", "t": time.time(), "key": k, "ms": ms, "old_clips": n_old})
    return True


def _writer_fill_step(server):
    eng = server.engine
    with engine_lock:
        if eng.claims or not server.writer_fill:
            return False
        item = None
        for key, want in list(server.writer_fill.items()):
            have = len(eng.cache.get(key, ()))
            if not eng.cache.get(key) or have >= want or key in server.writer_pending:
                server.writer_fill.pop(key, None)
                continue
            held = {c.variant_id for c in eng.cache[key]}
            v = next((i for i in range(want) if i not in held and (key, i) not in eng.inflight), None)
            if v is None:
                server.writer_fill.pop(key, None)
                continue
            item = (key, v)
            eng.inflight.add(item)
            break
        if item is None:
            return False
    try:
        clip = eng.gen.generate(item[0], variant=item[1])
    except Exception:
        with engine_lock:
            eng.inflight.discard(item)
        raise
    server.last_gen_t = time.monotonic()
    with engine_lock:
        eng.commit_clips([item], [clip], "spec_gen")
    _feed(server, "render", item[0], variant=item[1], why="writer_fill")
    return True


def _routed(server):
    g = server.engine.gen
    for _ in range(4):
        if callable(getattr(g, "upgrade_take", None)):
            return g
        g = getattr(g, "inner", None)
        if g is None:
            return None
    return None


def _upgrade_ready(server, g, now):
    st = server.upgrade_state
    if st == "ready":
        return True
    if st in ("failed", "off"):
        return False
    if now - server.upgrade_t0 < UPGRADE_DELAY_S or now - server.last_gen_t < UPGRADE_IDLE_S:
        return False
    if now - server.upgrade_polled < UPGRADE_POLL_S:
        return False
    server.upgrade_polled = now
    info = g.upgrade_control("upgrade_load" if st == "idle" else "upgrade_state")
    got = info.get("state")
    if got in ("loading", "ready", "failed", "off"):
        server.upgrade_state = "failed" if got == "off" else got
    if getattr(server, "event_log", None) is not None and server.upgrade_state != st:
        _log(server, {"type": "upgrade_state", "t": time.time(), "state": server.upgrade_state,
                      "error": info.get("error")})
    return server.upgrade_state == "ready"


def _upgrade_pending(server, key, pool):
    done = server.upgrade_done.get(key, {})
    st = server.upgrade_stage.get(key) or {"clips": {}, "orig": {}}
    return [c for c in pool if done.get(c.variant_id) is not c and st["orig"].get(c.variant_id) is not c]


def _upgrade_init(server):
    server.upgrade_fail = {}
    server.upgrade_rev = {}
    server.upgrade_first = {}
    server.upgrade_timed = set()
    server.upgrade_lag = []
    server.upgrade_unreach = 0
    server.upgrade_line_t = 0.0
    server.upgrade_line_dirty = False
    server.upgrade_count = {"rendered": 0, "preempted": 0, "errors": 0, "unreachable": 0, "failed_keys": 0,
                            "before_hit": 0}


def _upgrade_drop(server, key):
    server.upgrade_done.pop(key, None)
    server.upgrade_stage.pop(key, None)
    return server.upgrade_rev.pop(key, None)


def _upgrade_forget(server, key):
    at = _upgrade_drop(server, key)
    if at is not None and _rev(server, key) == at:
        _bump_rev(server, key)


def _upgrade_sync(server, key):
    done = server.upgrade_done.get(key)
    if done and not any(done.get(c.variant_id) is c for c in server.engine.cache.get(key) or ()):
        _upgrade_forget(server, key)


def _upgrade_dead(server, key, prompt):
    rec = server.upgrade_fail.get(key)
    return rec is not None and rec["dead"] and rec["prompt"] == prompt


def _upgrade_blocked(server, key, prompt, now):
    rec = server.upgrade_fail.get(key)
    if rec is None:
        return False
    if rec["prompt"] != prompt:
        server.upgrade_fail.pop(key, None)
        return False
    return rec["dead"] or now < rec["until"]


def _pool_backends(pool, writer):
    out = []
    mine = getattr(writer, "backend", None)
    for w in getattr(pool, "owned", ()):
        b = getattr(w, "backend", None)
        if (callable(getattr(b, "pause", None)) and getattr(b, "name", "") != "ground" and b is not mine
                and all(b is not o for o in out)):
            out.append(b)
    return out


def _game_prompt(server, key, pm):
    k = str(key)
    owned = getattr(server, "writer_owned", None)
    return k in pm and (owned is None or k not in owned)


def _upgrade_extra(server, key, pm=None):
    pool = getattr(server, "prompt_pool", None)
    w = getattr(server, "writer", None)
    pm = _prompts(server) or {} if pm is None else pm
    if _game_prompt(server, key, pm):
        return {}
    out = {}
    rec = w.lookup(key) if w is not None else None
    texts = rec.get("texts") if rec else None
    if pool is not None:
        prompts = pool.prompts(key, "woosh")
        if prompts:
            out["pool"] = prompts
        texts = texts or pool.texts(key)
    if isinstance(texts, dict) and texts:
        out["texts"] = dict(texts)
    return out


def _pool_waits(server, key, pm):
    pool = getattr(server, "prompt_pool", None)
    return pool is not None and not _game_prompt(server, key, pm) and not pool.ready(key)


def _upgrade_pick(server, g, now=None):
    eng = server.engine
    pm = _prompts(server) or {}
    now = time.monotonic() if now is None else now
    for key in list(server.upgrade_done):
        _upgrade_sync(server, key)
    for key in list(server.upgrade_stage):
        if key not in eng.cache:
            server.upgrade_stage.pop(key, None)
    keys = sorted((k for k in eng.cache if g.route(k) != "guns"), key=lambda k: k not in eng.fired)
    for key in keys:
        prompt = pm.get(str(key))
        st = server.upgrade_stage.get(key)
        if st is not None and st["prompt"] != prompt:
            server.upgrade_stage.pop(key, None)
        if _upgrade_blocked(server, key, prompt, now) or _pool_waits(server, key, pm):
            continue
        for c in _upgrade_pending(server, key, eng.cache.get(key) or []):
            if (key, c.variant_id) not in eng.inflight:
                return key, c, prompt
    return None


def _upgrade_summary(server):
    eng = server.engine
    g = _routed(server)
    pm = _prompts(server) or {}
    queued = keys = staged = skipped = 0
    for key, pool in list(eng.cache.items()):
        if not pool or g is None or g.route(key) == "guns":
            continue
        if _upgrade_dead(server, key, pm.get(str(key))):
            skipped += 1
            continue
        n = len(_upgrade_pending(server, key, pool))
        queued += n
        keys += 1 if n else 0
        st = server.upgrade_stage.get(key)
        if st:
            staged += sum(1 for c in pool if st["orig"].get(c.variant_id) is c)
    c = server.upgrade_count
    lag = sorted(server.upgrade_lag)
    return {"engine": server.upgrade, "state": server.upgrade_state, "queued_takes": queued, "queued_keys": keys,
            "staged_takes": staged, "rendered": c["rendered"], "swapped_takes": int(eng.stats.get("upgrades", 0)),
            "swaps": int(eng.stats.get("upgrade_swaps", 0)), "preempted": c["preempted"], "errors": c["errors"],
            "unreachable": c["unreachable"], "failed_keys": c["failed_keys"], "skipped_keys": skipped,
            "first_hit_to_swap_s": {"keys": len(lag), "median": statistics.median(lag) if lag else None,
                                    "max": lag[-1] if lag else None},
            "swapped_before_hit": c["before_hit"], **({"sa3": _sa3_summary(server)} if _sa3_on(server) else {})}


def _upgrade_line(s):
    lag = s["first_hit_to_swap_s"]
    when = ("first hit to swap median %.1f s, max %.1f s over %d keys" % (lag["median"], lag["max"], lag["keys"])
            if lag["keys"] else "no swap after a first hit yet")
    return ("upgrade %s: queued %d takes (%d keys), staged %d, rendered %d, swapped %d takes in %d swaps, "
            "%d before a first hit, preempted %d, failed %d renders, %d keys skipped, %s"
            % (s["engine"], s["queued_takes"], s["queued_keys"], s["staged_takes"], s["rendered"],
               s["swapped_takes"], s["swaps"], s["swapped_before_hit"], s["preempted"], s["errors"],
               s["skipped_keys"], when)) + (_sa3_line(s["sa3"]) if "sa3" in s else "")


def _upgrade_say(server, summary, force=False):
    now = time.monotonic()
    if summary is None or not (force or now - server.upgrade_line_t >= UPGRADE_LINE_S):
        return
    server.upgrade_line_t = now
    server.upgrade_line_dirty = False
    print(_upgrade_line(summary), flush=True)


def _upgrade_error(server, key, variant, prompt, err, ms):
    now = time.monotonic()
    msg = str(err)[:300]
    c = server.upgrade_count
    gone = None
    with engine_lock:
        if getattr(err, "unreachable", False):
            server.upgrade_unreach += 1
            wait = min(UPGRADE_UNREACH_MAX_S, UPGRADE_BACKOFF_S * 2 ** (server.upgrade_unreach - 1))
            server.upgrade_block = now + wait
            c["unreachable"] += 1
            rec = {"type": "upgrade_error", "t": time.time(), "key": str(key), "variant": variant, "ms": ms,
                   "error": msg, "unreachable": True, "retry_in_s": wait}
        else:
            fail = server.upgrade_fail.get(key)
            if fail is None or fail["prompt"] != prompt:
                fail = server.upgrade_fail[key] = {"n": 0, "prompt": prompt, "until": 0.0, "dead": False}
            fail["n"] += 1
            fail["error"] = msg
            c["errors"] += 1
            server.upgrade_line_dirty = True
            if fail["n"] > UPGRADE_RETRIES:
                fail["dead"] = True
                server.upgrade_stage.pop(key, None)
                c["failed_keys"] += 1
                rec = {"type": "upgrade_failed", "t": time.time(), "key": str(key), "variant": variant, "ms": ms,
                       "attempts": fail["n"], "error": msg}
                gone = _upgrade_summary(server)
            else:
                wait = UPGRADE_BACKOFF_S * 2 ** (fail["n"] - 1)
                fail["until"] = now + wait
                rec = {"type": "upgrade_error", "t": time.time(), "key": str(key), "variant": variant, "ms": ms,
                       "error": msg, "attempt": fail["n"], "retry_in_s": wait}
    if gone is not None:
        print("upgrade %s: %s failed %d times (%s), skipped for the session"
              % (server.upgrade, key, rec["attempts"], msg), flush=True)
        _upgrade_say(server, gone, True)
    elif rec.get("unreachable") and server.upgrade_unreach == 1:
        print("upgrade %s: the worker did not answer (%s), retrying with backoff" % (server.upgrade, msg), flush=True)
    if getattr(server, "event_log", None) is not None:
        _log(server, rec)


def _upgrade_step(server):
    g = _routed(server)
    if g is None:
        return False
    now = time.monotonic()
    if now < server.upgrade_block or not _upgrade_ready(server, g, now):
        return False
    eng = server.engine
    drained = None
    with engine_lock:
        if eng.claims or eng.inflight or now - server.last_gen_t < UPGRADE_IDLE_S:
            return False
        pick = _upgrade_pick(server, g, now)
        if pick is None and server.upgrade_line_dirty:
            drained = _upgrade_summary(server)
    if pick is None:
        _upgrade_say(server, drained, True)
        return False
    key, orig, prompt = pick
    extra = _upgrade_extra(server, key)
    t = time.perf_counter()
    try:
        clip = g.upgrade_take(key, orig.variant_id, prompt, **extra)
    except Exception as e:
        _upgrade_error(server, key, orig.variant_id, prompt, e, (time.perf_counter() - t) * 1e3)
        return not getattr(e, "unreachable", False)
    ms = (time.perf_counter() - t) * 1e3
    if clip is None:
        server.upgrade_count["preempted"] += 1
        server.upgrade_block = time.monotonic() + UPGRADE_IDLE_S
        if getattr(server, "event_log", None) is not None:
            _log(server, {"type": "upgrade_yield", "t": time.time(), "key": str(key), "variant": orig.variant_id,
                          "ms": ms})
        return False
    _feed(server, "flow_take", key, variant=orig.variant_id, take="flow", ms=round(ms, 1))
    swapped = 0
    with engine_lock:
        server.upgrade_count["rendered"] += 1
        server.upgrade_unreach = 0
        fail = server.upgrade_fail.get(key)
        if fail is not None and not fail["dead"]:
            server.upgrade_fail.pop(key, None)
        pool = eng.cache.get(key)
        if not pool or (_prompts(server) or {}).get(str(key)) != prompt or not any(c is orig for c in pool):
            return True
        st = server.upgrade_stage.setdefault(key, {"prompt": prompt, "clips": {}, "orig": {}, "ms": {}})
        st["clips"][orig.variant_id] = clip
        st["orig"][orig.variant_id] = orig
        st.setdefault("ms", {})[orig.variant_id] = ms
        if _upgrade_pending(server, key, pool):
            return True
        new = [st["clips"][c.variant_id] if st["orig"].get(c.variant_id) is c else c for c in pool]
        eng.cache[key] = new
        done = server.upgrade_done.setdefault(key, {})
        for c in new:
            if st["clips"].get(c.variant_id) is c:
                done[c.variant_id] = c
                swapped += 1
        server.upgrade_stage.pop(key, None)
        _bump_rev(server, key)
        rev = server.upgrade_rev[key] = _rev(server, key)
        eng.stats["upgrades"] = eng.stats.get("upgrades", 0) + swapped
        eng.stats["upgrade_swaps"] = eng.stats.get("upgrade_swaps", 0) + 1
        t_first = server.upgrade_first.get(key)
        first_hit_s = time.monotonic() - t_first if t_first is not None else None
        first_swap = key not in server.upgrade_timed
        if first_swap:
            server.upgrade_timed.add(key)
            if first_hit_s is None:
                server.upgrade_count["before_hit"] += 1
            else:
                server.upgrade_lag.append(first_hit_s)
        server.upgrade_line_dirty = True
        summary = _upgrade_summary(server)
    _upgrade_say(server, summary)
    _feed(server, "flow_swap", key, takes=swapped, take="flow")
    if getattr(server, "event_log", None) is not None:
        _log(server, {"type": "upgrade_swap", "t": time.time(), "key": str(key), "takes": swapped, "ms": ms,
                      "rev": rev, "take_ms": [st["ms"].get(v) for v in sorted(st["ms"])],
                      "first_hit_s": first_hit_s, "first_swap": first_swap,
                      "queued": summary["queued_takes"], "rendered": summary["rendered"],
                      "swapped": summary["swapped_takes"], "failed": summary["errors"],
                      "skipped_keys": summary["skipped_keys"]})
    return True


def _sa3_gen(server):
    g = _routed(server)
    return g if g is not None and int(getattr(g, "sa3_best", 1) or 1) > 1 else None


def _sa3_on(server):
    return getattr(server, "sa3_done", None) is not None


def _yield_gates(server):
    out, g = [], server.engine.gen
    for _ in range(8):
        d = getattr(g, "__dict__", None)
        if not d:
            break
        if d.get("_gate") is not None:
            out.append(d["_gate"])
        g = d.get("_g") if d.get("_g") is not None else d.get("inner")
    return out


def _sa3_init(server):
    server.sa3_done = {}
    server.sa3_stage = {}
    server.sa3_rev = {}
    server.sa3_fail = {}
    server.sa3_timed = set()
    server.sa3_lag = []
    server.sa3_ms = {"render": [], "verify": []}
    server.sa3_state = "on"
    server.sa3_error = None
    server.sa3_turn = False
    server.sa3_count = {"candidates": 0, "scored": 0, "swaps": 0, "swapped": 0, "kept": 0, "errors": 0,
                        "failed_keys": 0, "before_hit": 0}


def _sa3_drop(server, key):
    if not _sa3_on(server):
        return None
    server.sa3_done.pop(key, None)
    server.sa3_stage.pop(key, None)
    return server.sa3_rev.pop(key, None)


def _sa3_sync(server, key):
    if not _sa3_on(server):
        return
    done = server.sa3_done.get(key)
    if done and not any(done.get(c.variant_id) is c for c in server.engine.cache.get(key) or ()):
        at = _sa3_drop(server, key)
        if at is not None and _rev(server, key) == at:
            _bump_rev(server, key)


def _sa3_served(server, key, variant_id, clip):
    return _sa3_on(server) and server.sa3_done.get(key, {}).get(variant_id) is clip


def _sa3_pending(server, key, pool):
    done = server.sa3_done.get(key, {})
    st = server.sa3_stage.get(key) or {"orig": {}}
    return [c for c in pool if done.get(c.variant_id) is not c and st["orig"].get(c.variant_id) is not c]


def _sa3_blocked(server, key, prompt, now):
    rec = server.sa3_fail.get(key)
    if rec is None:
        return False
    if rec["prompt"] != prompt:
        server.sa3_fail.pop(key, None)
        return False
    return rec["dead"] or now < rec["until"]


def _sa3_pick(server, g, now):
    eng = server.engine
    pm = _prompts(server) or {}
    for key in list(server.sa3_done):
        _sa3_sync(server, key)
    for key in list(server.sa3_stage):
        if key not in eng.cache:
            server.sa3_stage.pop(key, None)
    keys = sorted((k for k in eng.cache if g.route(k) == "guns"), key=lambda k: k not in eng.fired)
    for key in keys:
        prompt = pm.get(str(key))
        st = server.sa3_stage.get(key)
        if st is not None and st["prompt"] != prompt:
            server.sa3_stage.pop(key, None)
            st = None
        if _sa3_blocked(server, key, prompt, now):
            continue
        for c in _sa3_pending(server, key, eng.cache.get(key) or []):
            if (key, c.variant_id) in eng.inflight:
                continue
            if st is None:
                st = server.sa3_stage[key] = {"prompt": prompt, "work": {}, "cands": {}, "ms": {}, "orig": {},
                                              "chosen": {}, "pick": {}, "scores": {}}
            if st["work"].get(c.variant_id) is not c:
                st["work"][c.variant_id] = c
                st["cands"][c.variant_id] = []
                st["ms"][c.variant_id] = []
            return key, c, prompt, st
    return None


def _sa3_summary(server):
    eng = server.engine
    g = _sa3_gen(server)
    queued = keys = 0
    for key, pool in list(eng.cache.items()):
        if not pool or g is None or g.route(key) != "guns":
            continue
        rec = server.sa3_fail.get(key)
        if rec is not None and rec["dead"]:
            continue
        n = len(_sa3_pending(server, key, pool))
        queued += n
        keys += 1 if n else 0
    c = server.sa3_count
    lag = sorted(server.sa3_lag)
    rm, vm = server.sa3_ms["render"], server.sa3_ms["verify"]
    return {"best": int(getattr(g, "sa3_best", 0) or 0), "state": server.sa3_state, "error": server.sa3_error,
            "queued_takes": queued, "queued_keys": keys, "candidates": c["candidates"], "scored": c["scored"],
            "swaps": c["swaps"], "swapped_takes": c["swapped"], "kept_takes": c["kept"], "errors": c["errors"],
            "failed_keys": c["failed_keys"], "render_ms_median": statistics.median(rm) if rm else None,
            "verify_ms_median": statistics.median(vm) if vm else None,
            "first_hit_to_swap_s": {"keys": len(lag), "median": statistics.median(lag) if lag else None,
                                    "max": lag[-1] if lag else None},
            "swapped_before_hit": c["before_hit"]}


def _sa3_line(x):
    return ("; sa3 best of %d (%s): queued %d takes (%d keys), %d candidates rendered (median %s ms), %d takes "
            "scored (median %s ms a group), swapped %d takes in %d swaps, kept %d served takes, failed %d"
            % (x["best"], x["state"], x["queued_takes"], x["queued_keys"], x["candidates"],
               "%.0f" % x["render_ms_median"] if x["render_ms_median"] is not None else "-", x["scored"],
               "%.0f" % x["verify_ms_median"] if x["verify_ms_median"] is not None else "-", x["swapped_takes"],
               x["swaps"], x["kept_takes"], x["errors"]))


def _sa3_error(server, key, variant, prompt, err, ms):
    now = time.monotonic()
    msg = str(err)[:300]
    first_fail = False
    with engine_lock:
        server.sa3_count["errors"] += 1
        if getattr(err, "unreachable", False):
            server.upgrade_block = max(server.upgrade_block, now + UPGRADE_BACKOFF_S)
            rec = {"type": "sa3_error", "t": time.time(), "key": str(key), "variant": variant, "ms": ms,
                   "error": msg, "unreachable": True}
        elif "the verifier is" in msg or "no upgrade" in msg:
            first_fail = server.sa3_state == "on"
            server.sa3_state = "failed"
            server.sa3_error = msg
            server.sa3_stage.clear()
            rec = {"type": "sa3_failed", "t": time.time(), "key": str(key), "variant": variant, "error": msg,
                   "lane": True}
        else:
            fail = server.sa3_fail.get(key)
            if fail is None or fail["prompt"] != prompt:
                fail = server.sa3_fail[key] = {"n": 0, "prompt": prompt, "until": 0.0, "dead": False}
            fail["n"] += 1
            fail["error"] = msg
            if fail["n"] > UPGRADE_RETRIES:
                fail["dead"] = True
                server.sa3_stage.pop(key, None)
                server.sa3_count["failed_keys"] += 1
                rec = {"type": "sa3_failed", "t": time.time(), "key": str(key), "variant": variant, "ms": ms,
                       "attempts": fail["n"], "error": msg}
            else:
                fail["until"] = now + UPGRADE_BACKOFF_S * 2 ** (fail["n"] - 1)
                rec = {"type": "sa3_error", "t": time.time(), "key": str(key), "variant": variant, "ms": ms,
                       "error": msg, "attempt": fail["n"]}
    if first_fail:
        print("upgrade sa3: the worker does not score SA3 takes (%s), single SA3 takes stay for the session" % msg,
              flush=True)
    if getattr(server, "event_log", None) is not None:
        _log(server, rec)


def _sa3_step(server):
    from .. import verify
    from ..types import AudioClip
    g = _sa3_gen(server)
    if g is None or not _sa3_on(server) or server.sa3_state != "on":
        return False
    now = time.monotonic()
    if now < server.upgrade_block or not _upgrade_ready(server, g, now):
        return False
    eng = server.engine
    with engine_lock:
        if eng.claims or eng.inflight or now - server.last_gen_t < UPGRADE_IDLE_S:
            return False
        pick = _sa3_pick(server, g, now)
    if pick is None:
        return False
    key, orig, prompt, st = pick
    v = orig.variant_id
    have = st["cands"][v]
    t = time.perf_counter()
    if len(have) < g.sa3_best - 1:
        try:
            with contextlib.ExitStack() as gates:
                for gate in _yield_gates(server):
                    gates.enter_context(gate)
                clip = g.sa3_candidate(key, verify.candidate_variant(v, len(have) + 1))
        except Exception as e:
            _sa3_error(server, key, v, prompt, e, (time.perf_counter() - t) * 1e3)
            return True
        ms = (time.perf_counter() - t) * 1e3
        with engine_lock:
            server.sa3_count["candidates"] += 1
            server.sa3_ms["render"].append(ms)
            if server.sa3_stage.get(key) is st and st["work"].get(v) is orig and st["cands"].get(v) is have:
                have.append(clip)
                st["ms"][v].append(ms)
        _feed(server, "sa3_candidate", key, variant=v, takes=len(have) + 1, ms=round(ms, 1))
        return True
    group = [orig] + list(have)
    try:
        scores = g.sa3_verify(key, group)
    except Exception as e:
        _sa3_error(server, key, v, prompt, e, (time.perf_counter() - t) * 1e3)
        return not getattr(e, "unreachable", False)
    vms = (time.perf_counter() - t) * 1e3
    _s, j, best = verify.pick([(float(x), i, c) for i, (x, c) in enumerate(zip(scores, group))])
    chosen = orig if j == 0 else AudioClip(best.samples, best.sample_rate, variant_id=v, gen_ms=best.gen_ms)
    rec = None
    with engine_lock:
        server.sa3_count["scored"] += len(group)
        server.sa3_ms["verify"].append(vms)
        fail = server.sa3_fail.get(key)
        if fail is not None and not fail["dead"]:
            server.sa3_fail.pop(key, None)
        if server.sa3_stage.get(key) is not st or st["work"].get(v) is not orig:
            return True
        st["work"].pop(v, None)
        st["cands"].pop(v, None)
        st["orig"][v] = orig
        st["chosen"][v] = chosen
        st["pick"][v] = j
        st["scores"][v] = [float(x) for x in scores]
        pool = eng.cache.get(key)
        if not pool or (_prompts(server) or {}).get(str(key)) != prompt or not any(c is orig for c in pool):
            return True
        if _sa3_pending(server, key, pool):
            return True
        new = [st["chosen"][c.variant_id] if st["orig"].get(c.variant_id) is c else c for c in pool]
        changed = sum(1 for a, b in zip(new, pool) if a is not b)
        decided = sum(1 for c in pool if st["orig"].get(c.variant_id) is c)
        done = server.sa3_done.setdefault(key, {})
        for a, b in zip(new, pool):
            if st["orig"].get(b.variant_id) is b:
                done[b.variant_id] = a
        server.sa3_stage.pop(key, None)
        c = server.sa3_count
        c["kept"] += decided - changed
        rev = _rev(server, key)
        if changed:
            eng.cache[key] = new
            _bump_rev(server, key)
            rev = server.sa3_rev[key] = _rev(server, key)
            c["swaps"] += 1
            c["swapped"] += changed
        t_first = server.upgrade_first.get(key) if hasattr(server, "upgrade_first") else None
        first_hit_s = time.monotonic() - t_first if t_first is not None else None
        first_swap = key not in server.sa3_timed
        if first_swap:
            server.sa3_timed.add(key)
            if first_hit_s is None:
                c["before_hit"] += 1
            else:
                server.sa3_lag.append(first_hit_s)
        rec = {"type": "sa3_swap", "t": time.time(), "key": str(key), "takes": changed, "decided": decided,
               "picks": {str(k): x for k, x in sorted(st["pick"].items())},
               "scores": {str(k): x for k, x in sorted(st["scores"].items())},
               "render_ms": {str(k): x for k, x in sorted(st["ms"].items())}, "verify_ms": vms, "rev": rev,
               "first_hit_s": first_hit_s, "first_swap": first_swap}
        if changed:
            server.upgrade_line_dirty = True
        summary = _upgrade_summary(server)
    _upgrade_say(server, summary)
    _feed(server, "sa3_pick", key, takes=changed, decided=decided)
    if getattr(server, "event_log", None) is not None and rec is not None:
        _log(server, rec)
    return True


def _upgrade_any(server):
    if not _sa3_on(server):
        return _upgrade_step(server)
    for step in ((_sa3_step, _upgrade_step) if server.sa3_turn else (_upgrade_step, _sa3_step)):
        if step(server):
            server.sa3_turn = step is _upgrade_step
            return True
    return False


def _meaning_sources(server):
    from ..canon import asks_firearm
    out = []
    for w in [getattr(server, "writer", None)] + list(getattr(getattr(server, "prompt_pool", None), "writers", ())):
        if w is not None and asks_firearm(getattr(w, "spec", None)) and all(w is not o for o in out):
            out.append(w)
    return out


def _meaning_lookup(sources):
    from ..canon import firearm_of

    def lookup(key):
        for w in sources:
            ans = firearm_of(w.lookup(key))
            if ans is not None:
                return ans
        return None

    return lookup


def _route_step(server):
    g = _routed(server)
    flips = getattr(g, "flips", None) if g is not None else None
    if not callable(flips):
        return False
    eng = server.engine
    done = None
    with engine_lock:
        for key in flips():
            if key in eng.claims or any(k == key for k, _v in eng.inflight):
                continue
            side = g.settle(key)
            held = bool(eng.cache.get(key))
            keep = held and side == "rest" and bool(server.upgrade)
            if held and not keep:
                eng.cache.pop(key, None)
                eng.recent.pop(key, None)
                _upgrade_drop(server, key)
                _sa3_drop(server, key)
                _bump_rev(server, key)
            eng.stats["route_flips"] = eng.stats.get("route_flips", 0) + 1
            done = (key, side, held, keep)
            break
    if done is None:
        return False
    key, side, held, keep = done
    _feed(server, "route_flip", key, side=side)
    w = getattr(server, "writer", None)
    rec = w.lookup(key) if w is not None else None
    if rec:
        _writer_deliver(server, key, rec)
    if getattr(server, "event_log", None) is not None:
        _log(server, {"type": "route_flip", "t": time.time(), "key": str(key), "side": side, "held": held,
                      "kept_for_upgrade": keep})
    return True


def _writer_written(server, key, rec):
    _writer_deliver(server, key, rec)
    ms = rec.get("ms")
    _feed(server, "writer", key, ms=round(float(ms), 1) if isinstance(ms, (int, float)) else 0.0)
    if getattr(server, "event_log", None) is not None:
        _log(server, {"type": "writer", "t": time.time(), "key": str(key), "ms": rec.get("ms"),
                      "repaired": bool(rec.get("repaired")), "route": _writer_route(server, key)})


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
            if not batch and getattr(server, "route_meaning", False) and _route_step(server):
                continue
            if not batch and getattr(server, "writer", None) is not None and (
                    _writer_step(server) or _writer_fill_step(server)):
                continue
            if not batch and getattr(server, "upgrade", None) and _upgrade_any(server):
                continue
            if batch:
                tg = time.perf_counter()
                clips = eng.gen.generate_batch(batch)
                with engine_lock:
                    eng.commit_clips(batch, clips, stat)
                server.last_gen_t = time.monotonic()
                for k, v in batch:
                    _feed(server, "render", k, variant=v, why=stat)
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
          cache_async: bool = False, rapid_hz: float = RAPID_HZ, read_class: bool = False, clock=None,
          scene: bool = False, writer=None, writer_swap: str = "revalidate", writer_yield: bool = True,
          upgrade: str | None = None, prompt_pool=None, trace: bool = True, **engine_kwargs) -> SfxServer:
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
    if writer is not None and writer_yield and callable(getattr(getattr(writer, "backend", None), "pause", None)):
        from ..writer import Yielding
        engine.gen = Yielding(engine.gen, writer.backend)
    for b in _pool_backends(prompt_pool, writer) if writer_yield else ():
        from ..writer import Yielding
        engine.gen = Yielding(engine.gen, b)
    server = SfxServer((host, port), engine, default_pcm=pcm, recv_timeout=recv_timeout)
    server.trace = bool(trace)
    server.feed = collections.deque(maxlen=FEED_MAX) if server.trace else None
    server.feed_seq = 0
    server.feed_lock = threading.Lock()
    server.world = bool(world)
    server.last_phys = {}
    server.predict_fixes = bool(predict_fixes)
    server.concurrent_gen = bool(concurrent_gen)
    server.fired_at = {}
    server.fired_seq = 0
    server.last_intensity = {}
    server.rapid_hz = float(rapid_hz or 0.0)
    server.read_class = bool(read_class)
    server.clock = clock or time.monotonic
    server.last_fire_t = {}
    server.last_class = {}
    server.kind_pending = {}
    server.kind_score = {}
    server.kind_targets = {}
    server.scene = bool(scene)
    server.last_scene = {}
    server.writer = writer
    server.writer_swap = writer_swap if writer_swap in ("revalidate", "next-miss") else "revalidate"
    server.writer_owned = set() if writer is not None else None
    server.writer_pending = {}
    server.writer_fill = {}
    server.writer_rendering = None
    server.prompt_pool = prompt_pool
    routed = _routed(server)
    server.route_meaning = getattr(routed, "route_mode", None) == "meaning"
    if server.route_meaning:
        sources = _meaning_sources(server)
        routed.meaning = _meaning_lookup(sources) if sources else None
    server.prompt_rev = {} if writer is not None or upgrade or server.route_meaning else None
    server.upgrade = upgrade or None
    server.upgrade_state = "idle"
    server.upgrade_t0 = time.monotonic()
    server.upgrade_polled = 0.0
    server.upgrade_block = 0.0
    server.upgrade_done = {}
    server.upgrade_stage = {}
    _upgrade_init(server)
    if server.upgrade and _sa3_gen(server) is not None:
        _sa3_init(server)
    if writer is not None:
        writer.on_written = lambda key, rec: _writer_written(server, key, rec)
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
        owner=lambda source, key: source == MOTION_SOURCE)
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
