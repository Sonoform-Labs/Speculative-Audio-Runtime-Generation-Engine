from __future__ import annotations

import socket
import socketserver
import threading
import time

from . import wire
from .wire import ClientGone, FrameError, IdleTimeout, MsgType
from .. import __version__
from ..affordance import AffordanceEngine, GameContext
from ..generators import get_generator
from ..types import CacheKey

engine_lock = threading.Lock()

MAX_PREBAKE_CONTEXTS = 256


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
                msg_type, body, _ = wire.read_frame(self.request)
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
            self.server.engine.on_context_change(ctx, gpu)
            stats = dict(self.server.engine.stats)
            cached = len(self.server.engine.cache)
        self._send(MsgType.CONTEXT_ACK, {
            "type": "context_ack",
            "spec_gen": int(stats["spec_gen"]),
            "evicted": int(stats["evicted"]),
            "cached_keys": cached,
        })

    def _on_event(self, body):
        try:
            event_type = str(body["event_type"])
            material = str(body["material"])
        except (KeyError, TypeError):
            raise FrameError("bad_fields", "event needs event_type and material", fatal=False)
        key = CacheKey(event_type, material, str(body.get("intensity", "normal")), str(body.get("context", "default")))
        with engine_lock:
            served = self.server.engine.on_event(key)
            clip = self.server.engine.last_served
            samples = clip.samples
            variant_id = int(clip.variant_id)
            sr = int(clip.sample_rate)
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
        }, pcm)

    def _on_prebake(self, body):
        if not isinstance(body, dict):
            raise FrameError("bad_fields", "prebake must be an object", fatal=False)
        contexts = body.get("contexts", [])
        if not isinstance(contexts, list):
            raise FrameError("bad_fields", "contexts must be a list", fatal=False)
        if len(contexts) > MAX_PREBAKE_CONTEXTS:
            raise FrameError("bad_fields", f"too many contexts (> {MAX_PREBAKE_CONTEXTS})", fatal=False)
        parsed = [(self._context_from(c), self._float_field(c, "gpu_pressure", 1.0)) for c in contexts]
        for ctx, gpu in parsed:
            with engine_lock:
                self.server.engine.on_context_change(ctx, gpu)
        with engine_lock:
            stats = dict(self.server.engine.stats)
            cached = len(self.server.engine.cache)
        self._send(MsgType.PREBAKE_ACK, {
            "type": "prebake_ack",
            "spec_gen": int(stats["spec_gen"]),
            "peak_bytes": int(stats["peak_bytes"]),
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


def serve(engine=None, generator: str = "mock", host: str = "127.0.0.1", port: int = wire.DEFAULT_PORT,
          pcm: str = "f32le", recv_timeout: float = 30.0, **engine_kwargs) -> SfxServer:
    if engine is None:
        engine = AffordanceEngine(get_generator(generator), **engine_kwargs)
    server = SfxServer((host, port), engine, default_pcm=pcm, recv_timeout=recv_timeout)
    thread = threading.Thread(target=server.serve_forever, name="sfx-sidecar", daemon=True)
    thread.start()
    server.runner = thread
    return server
