from __future__ import annotations

import socket

import numpy as np

from . import wire
from .wire import MsgType


class ProtocolError(Exception):
    def __init__(self, code: str, message: str, fatal: bool = False):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.fatal = fatal


class SfxClient:
    def __init__(self, host: str = "127.0.0.1", port: int = wire.DEFAULT_PORT,
                 pcm: str = "f32le", timeout: float = 30.0, trace: bool = False):
        self.host = host
        self.port = port
        self.pcm = pcm
        self.timeout = timeout
        self.trace = bool(trace)
        self.sock = None
        self.welcome = None

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return self.hello()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def _round_trip(self, msg_type, body, pcm: bytes = b""):
        self.sock.sendall(wire.pack_frame(msg_type, body, pcm))
        rtype, rbody, rpcm = wire.read_frame(self.sock)
        if rtype == MsgType.ERROR:
            raise ProtocolError(rbody.get("code", "error"), rbody.get("message", ""), bool(rbody.get("fatal", False)))
        return rtype, rbody, rpcm

    def hello(self) -> dict:
        req = {"type": "hello", "proto": wire.PROTOCOL_VERSION, "client": "python-test", "pcm": self.pcm}
        if self.trace:
            req["trace"] = True
        _, body, _ = self._round_trip(MsgType.HELLO, req)
        self.welcome = body
        self.pcm = body.get("pcm", self.pcm)
        return body

    def context(self, time_s: float, equipped, surfaces, region: str = "default", gpu_pressure: float = 1.0) -> dict:
        _, body, _ = self._round_trip(MsgType.CONTEXT, {
            "type": "context", "time_s": time_s, "equipped": equipped,
            "surfaces": list(surfaces), "region": region, "gpu_pressure": gpu_pressure})
        return body

    def event(self, event_type: str, material: str, intensity: str = "normal", context: str = "default",
              phys=None, have=None, have_phys: str = "", cls: str = "", prompt: str = "", scene=None,
              have_scene: str = "", have_rev: int = 0):
        req = {"type": "event", "event_type": event_type, "material": material,
               "intensity": intensity, "context": context}
        if cls:
            req["class"] = cls
        if prompt:
            req["prompt"] = prompt
        if phys is not None:
            req["phys"] = phys
        if scene is not None:
            req["scene"] = scene
        if have is not None:
            req["have"] = list(have)
            req["have_phys"] = str(have_phys or "")
            if have_scene:
                req["have_scene"] = str(have_scene)
            if have_rev:
                req["have_rev"] = int(have_rev)
        _, body, pcm = self._round_trip(MsgType.EVENT, req)
        samples = wire.decode_pcm(pcm, body.get("format", self.pcm))
        ch = int(body.get("channels", 1) or 1)
        if ch > 1:
            samples = samples.reshape(-1, ch)
        return body, samples

    def prebake(self, contexts) -> dict:
        _, body, _ = self._round_trip(MsgType.PREBAKE, {"type": "prebake", "contexts": list(contexts)})
        return body

    def prewarm(self, keys) -> dict:
        _, body, _ = self._round_trip(MsgType.PREBAKE, {"type": "prewarm", "keys": list(keys)})
        return body

    def predict(self, predictions, now: float = 0.0) -> dict:
        _, body, _ = self._round_trip(MsgType.PREDICT, {
            "type": "predict", "now": now, "predictions": list(predictions)})
        return body

    def manifest(self, keys) -> dict:
        _, body, _ = self._round_trip(MsgType.MANIFEST, {"type": "manifest", "keys": list(keys)})
        return body

    def stats(self, feed=None) -> dict:
        req = {"type": "stats"}
        if feed is not None:
            req["feed"] = int(feed)
        _, body, _ = self._round_trip(MsgType.STATS, req)
        return body

    def ping(self, t=None) -> dict:
        _, body, _ = self._round_trip(MsgType.PING, {"type": "ping", "t": t})
        return body


def connect(host: str = "127.0.0.1", port: int = wire.DEFAULT_PORT,
            pcm: str = "f32le", timeout: float = 30.0, trace: bool = False) -> SfxClient:
    c = SfxClient(host, port, pcm, timeout, trace)
    c.connect()
    return c
