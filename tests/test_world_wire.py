import json
import socket
import struct
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from sarge import world
from sarge.generators.remote import RemoteGenerator
from sarge.sidecar import SfxClient, serve, wire
from sarge.sidecar.wire import MsgType
from sarge.types import CacheKey

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

BALL = {"shape": "sphere", "ext": [0.6, 0.6, 0.6], "m": 1.0, "dyn": 1, "pm": "sarge_bouncy", "bnc": 1.0,
        "name": "AutoCaptureBall_0"}
METAL = {"shape": "box", "ext": [10, 0.2, 10], "dyn": 0, "tag": "metal", "name": "Surface_metal"}
PHYS = {"a": BALL, "b": METAL, "vn": 12.0, "vt": 0.3, "n": [0, 1, 0], "j": 12.0, "dist": 6.5}


def _server(world_on=False):
    return serve(generator="mock", host="127.0.0.1", port=0, realtime=True, vram_budget_kb=200_000,
                 world=world_on)


def _raw(sock, msg_type, body):
    sock.sendall(wire.pack_frame(msg_type, body))
    head = wire.recv_exactly(sock, wire.HEADER_SIZE)
    jl, pl = struct.unpack("<II", bytes(head[8:16]))
    return bytes(head) + bytes(wire.recv_exactly(sock, jl)) + bytes(wire.recv_exactly(sock, pl))


def _session(srv, events):
    s = socket.create_connection(srv.server_address)
    try:
        _raw(s, MsgType.HELLO, {"type": "hello", "proto": 1, "pcm": "f32le"})
        return [_raw(s, t, b) for t, b in events]
    finally:
        s.close()


def _stop(srv):
    srv.shutdown()
    srv.server_close()


EV = {"type": "event", "event_type": "impact", "material": "metal", "intensity": "hard", "context": "default"}
PR = {"type": "predict", "now": 0.0, "predictions": [
    {"source": "impact", "event_type": "impact", "material": "metal", "intensity": "hard", "eta_s": 1.0,
     "confidence": 0.9}]}


def test_without_world_phys_is_ignored_and_replies_are_byte_identical():
    plain, phys = _server(), _server()
    try:
        seq = [(MsgType.EVENT, EV), (MsgType.EVENT, EV), (MsgType.PREDICT, PR)]
        seq_phys = [(MsgType.EVENT, dict(EV, phys=PHYS)), (MsgType.EVENT, dict(EV, phys=PHYS)),
                    (MsgType.PREDICT, dict(PR, predictions=[dict(PR["predictions"][0], phys=PHYS)]))]
        a = _session(plain, seq)
        b = _session(phys, seq_phys)
    finally:
        _stop(plain)
        _stop(phys)
    assert a == b
    assert b"phys" not in b"".join(a)


def test_world_on_without_phys_matches_world_off():
    off, on = _server(False), _server(True)
    try:
        seq = [(MsgType.EVENT, EV), (MsgType.EVENT, dict(EV, event_type="footstep", material="grass")),
               (MsgType.PREDICT, PR)]
        assert _session(off, seq) == _session(on, seq)
    finally:
        _stop(off)
        _stop(on)


def test_world_on_buckets_the_event_and_returns_the_phys_key():
    srv = _server(True)
    try:
        with SfxClient(*srv.server_address) as c:
            m1, s1 = c.event("impact", "metal", "hard", phys=PHYS)
            m2, s2 = c.event("impact", "metal", "hard", phys=dict(PHYS, vn=20.0))
            m3, _ = c.event("impact", "metal", "hard")
        expect = world.bucket(PHYS, "impact")
        assert m1["phys_key"] == m2["phys_key"] == expect
        assert m1["served"] == "realtime" and m2["served"] == "cached"
        assert "phys_key" not in m3 and m3["served"] == "realtime"
        keys = set(srv.engine.cache)
        assert CacheKey("impact", "metal", "hard", "default", expect) in keys
        assert CacheKey("impact", "metal", "hard", "default") in keys
        assert not np.array_equal(s1, srv.engine.cache[CacheKey("impact", "metal", "hard")][0].samples)
    finally:
        _stop(srv)


def test_world_on_takes_intensity_from_speed_only_when_the_client_sends_none():
    srv = _server(True)
    try:
        with SfxClient(*srv.server_address) as c:
            c._round_trip(MsgType.EVENT, {"type": "event", "event_type": "impact", "material": "metal",
                                          "phys": dict(PHYS, vn=1.0)})
            c.event("impact", "metal", "hard", phys=dict(PHYS, vn=1.0))
        ints = {k.intensity for k in srv.engine.cache if k.phys}
        assert ints == {"soft", "hard"}
    finally:
        _stop(srv)


def test_world_on_bad_phys_falls_back_to_the_plain_key():
    srv = _server(True)
    try:
        with SfxClient(*srv.server_address) as c:
            m, _ = c.event("impact", "metal", "hard", phys=[1, 2, 3])
            m2, _ = c.event("impact", "metal", "hard", phys={"a": "nonsense"})
        assert "phys_key" not in m
        assert m2["phys_key"].startswith("hit;a=object,")
    finally:
        _stop(srv)


def test_predict_ack_lists_phys_keys_aligned_with_the_predictions():
    srv = _server(True)
    try:
        with SfxClient(*srv.server_address) as c:
            ack = c.predict([dict(PR["predictions"][0], phys=PHYS),
                             {"event_type": "footstep", "material": "grass", "eta_s": 1.0}])
            ack2 = c.predict([{"event_type": "footstep", "material": "grass", "eta_s": 1.0}])
        assert ack["phys_keys"] == [world.bucket(PHYS, "impact"), ""]
        assert "phys_keys" not in ack2
        assert any(k.phys for k in srv.engine.demand_set) or any(k.phys for k in srv.engine.cache)
    finally:
        _stop(srv)


class _Recorder:
    def __init__(self):
        self.reqs = []
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._run, daemon=True).start()

    def _recv(self, c, n):
        b = b""
        while len(b) < n:
            k = c.recv(n - len(b))
            if not k:
                raise ConnectionError
            b += k
        return b

    def _run(self):
        c, _ = self.srv.accept()
        try:
            while True:
                jl, pl = struct.unpack("<II", self._recv(c, 8))
                req = json.loads(self._recv(c, jl))
                self.reqs.append(req)
                pcm = np.zeros(8, dtype="<f4").tobytes()
                body = json.dumps({"type": "audio", "sample_rate": 48000, "channels": 1, "frames": 8,
                                   "gen_ms": 33.5}).encode()
                c.sendall(struct.pack("<II", len(body), len(pcm)) + body + pcm)
        except (ConnectionError, OSError):
            pass


def test_remote_sends_phys_only_when_set_and_keeps_gen_ms():
    rec = _Recorder()
    g = RemoteGenerator(port=rec.port)
    clip = g.generate(CacheKey("impact", "metal", "hard"))
    assert clip.gen_ms == 33.5
    g.generate(CacheKey("impact", "metal", "hard", "default", "bounce;a=x"))
    g.close()
    assert rec.reqs[0]["type"] == "hello"
    assert "phys" not in rec.reqs[1]
    assert rec.reqs[1] == {"type": "gen", "event_type": "impact", "material": "metal", "intensity": "hard",
                           "context": "default", "variant": 0}
    assert rec.reqs[2]["phys"] == "bounce;a=x"


def test_woosh_physics_shape_is_empty_without_phys_and_follows_finish_signature(monkeypatch):
    ws = pytest.importorskip("woosh_server")
    assert ws.physics_shape("") == {}
    k = world.bucket(PHYS, "impact")

    def with_shape(x, event, intensity, amp, stereo, level="peak", shape=None):
        return x

    def without(x, event, intensity, amp, stereo):
        return x

    monkeypatch.setattr(ws, "finish", without)
    assert ws.physics_shape(k) == {}
    monkeypatch.setattr(ws, "finish", with_shape)
    assert ws.physics_shape(k) == {"shape": world.shape(k)}


def test_remote_raises_on_an_error_reply_without_retrying():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(2)
    seen = []

    def run():
        c, _ = srv.accept()
        seen.append(1)
        rd = _Recorder._recv
        with c:
            while True:
                try:
                    jl, pl = struct.unpack("<II", rd(None, c, 8))
                    req = json.loads(rd(None, c, jl))
                except (ConnectionError, OSError):
                    return
                body = {"type": "hello", "sample_rate": 48000} if req["type"] == "hello" else \
                    {"type": "error", "message": "ValueError: not a physics bucket"}
                b = json.dumps(body).encode()
                c.sendall(struct.pack("<II", len(b), 0) + b)

    threading.Thread(target=run, daemon=True).start()
    g = RemoteGenerator(port=srv.getsockname()[1])
    with pytest.raises(RuntimeError, match="not a physics bucket"):
        g.generate(CacheKey("impact", "metal", "hard", "default", "nonsense"))
    g.close()
    srv.close()
    assert seen == [1]
