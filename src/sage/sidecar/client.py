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
                 pcm: str = "f32le", timeout: float = 30.0):
        self.host = host
        self.port = port
        self.pcm = pcm
        self.timeout = timeout
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
        _, body, _ = self._round_trip(MsgType.HELLO, {
            "type": "hello", "proto": wire.PROTOCOL_VERSION, "client": "python-test", "pcm": self.pcm})
        self.welcome = body
        self.pcm = body.get("pcm", self.pcm)
        return body

    def context(self, time_s: float, equipped, surfaces, region: str = "default", gpu_pressure: float = 1.0) -> dict:
        _, body, _ = self._round_trip(MsgType.CONTEXT, {
            "type": "context", "time_s": time_s, "equipped": equipped,
            "surfaces": list(surfaces), "region": region, "gpu_pressure": gpu_pressure})
        return body

    def event(self, event_type: str, material: str, intensity: str = "normal", context: str = "default"):
        _, body, pcm = self._round_trip(MsgType.EVENT, {
            "type": "event", "event_type": event_type, "material": material,
            "intensity": intensity, "context": context})
        samples = wire.decode_pcm(pcm, body.get("format", self.pcm))
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

    def stats(self) -> dict:
        _, body, _ = self._round_trip(MsgType.STATS, {"type": "stats"})
        return body

    def ping(self, t=None) -> dict:
        _, body, _ = self._round_trip(MsgType.PING, {"type": "ping", "t": t})
        return body


def connect(host: str = "127.0.0.1", port: int = wire.DEFAULT_PORT,
            pcm: str = "f32le", timeout: float = 30.0) -> SfxClient:
    c = SfxClient(host, port, pcm, timeout)
    c.connect()
    return c
