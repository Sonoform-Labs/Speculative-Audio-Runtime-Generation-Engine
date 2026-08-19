import socket
import threading
import time

import numpy as np
import pytest

import sage.sidecar as sidecar
from sage.sidecar import serve, SfxClient, ProtocolError, wire
from sage.sidecar.wire import MsgType
from sage.types import AudioClip, CacheKey


@pytest.fixture
def server():
    srv = serve(generator="mock", host="127.0.0.1", port=0,
                vram_budget_kb=50_000, gpu_ms_budget=10_000_000, variants_per_key=3)
    yield srv
    srv.shutdown()
    srv.server_close()


def test_handshake_welcome_protocol_1(server):
    with SfxClient(*server.server_address) as c:
        w = c.welcome
        assert w["proto"] == 1
        assert w["generator"] == "mock"
        assert w["sample_rate"] == 48000
        assert w["pcm"] == "f32le"


def test_equip_then_fire_is_cached_with_valid_pcm(server):
    with SfxClient(*server.server_address) as c:
        ack = c.context(0.0, "gun", ["grass"], "field")
        assert ack["cached_keys"] >= 1
        meta, samples = c.event("shoot", "metal", "normal")
        assert meta["served"] == "cached"
        assert samples.dtype == np.float32
        assert len(samples) == meta["n_samples"]
        assert meta["byte_len"] == len(samples) * 4
        assert samples.min() >= -1.0 and samples.max() <= 1.0


def test_unanticipated_event_falls_back(server):
    with SfxClient(*server.server_address) as c:
        c.context(0.0, None, ["grass"], "field")
        meta, samples = c.event("explosion", "metal", "hard")
        assert meta["served"] == "fallback"
        assert len(samples) > 0


def test_pcm_roundtrips_engine_last_served(server):
    with SfxClient(*server.server_address) as c:
        c.context(0.0, "gun", ["grass"], "field")
        meta, samples = c.event("shoot", "metal", "normal")
    assert np.array_equal(samples, server.engine.last_served.samples)


def test_stats_match_engine(server):
    with SfxClient(*server.server_address) as c:
        c.context(0.0, "gun", ["grass"], "field")
        c.event("shoot", "metal", "normal")
        c.event("explosion", "metal", "hard")
        reply = c.stats()
    assert reply["stats"] == server.engine.stats
    assert reply["cached_keys"] == len(server.engine.cache)
    assert reply["wasted"] == server.engine.wasted()


def test_ping_pong_rtt(server):
    with SfxClient(*server.server_address) as c:
        pong = c.ping(t=123.5)
        assert pong["t"] == 123.5
        assert "server_t" in pong


def test_i16le_negotiated_via_hello(server):
    with SfxClient(*server.server_address, pcm="i16le") as c:
        assert c.welcome["pcm"] == "i16le"
        c.context(0.0, "gun", ["grass"], "field")
        meta, samples = c.event("shoot", "metal", "normal")
        assert meta["format"] == "i16le"
        assert meta["byte_len"] == meta["n_samples"] * 2
        assert samples.min() >= -1.0 and samples.max() <= 1.0


def test_prebake_warms_cache(server):
    with SfxClient(*server.server_address) as c:
        ack = c.prebake([{"time_s": 0.0, "equipped": "gun", "surfaces": ["grass"], "region": "field"}])
        assert ack["cached_keys"] >= 1
        meta, _ = c.event("shoot", "metal", "normal")
        assert meta["served"] == "cached"


def test_two_clients_no_last_served_corruption(server):
    addr = server.server_address
    with SfxClient(*addr) as setup:
        setup.context(0.0, "gun", ["grass"], "field")

    a_key = CacheKey("shoot", "metal", "normal")
    b_key = CacheKey("footstep", "grass", "normal")
    a_pool = [c.samples for c in server.engine.cache[a_key]]
    b_pool = [c.samples for c in server.engine.cache[b_key]]
    errors = []

    def hammer(event_type, material, pool):
        try:
            with SfxClient(*addr) as c:
                for _ in range(40):
                    _, samples = c.event(event_type, material, "normal")
                    if not any(np.array_equal(samples, p) for p in pool):
                        errors.append((event_type, material))
                        return
        except Exception as e:
            errors.append(e)

    ta = threading.Thread(target=hammer, args=("shoot", "metal", a_pool))
    tb = threading.Thread(target=hammer, args=("footstep", "grass", b_pool))
    ta.start(); tb.start(); ta.join(); tb.join()
    assert errors == []


def test_oversized_frame_errors_and_closes(server):
    raw = socket.create_connection(server.server_address, timeout=5.0)
    try:
        raw.sendall(wire.pack_frame(MsgType.HELLO, {"type": "hello", "proto": 1, "pcm": "f32le"}))
        wire.read_frame(raw)
        bad = wire.HEADER.pack(wire.MAGIC, 1, int(MsgType.EVENT), wire.FLAG_HAS_PCM, 0, 2, wire.MAX_PCM + 1)
        raw.sendall(bad + b"{}")
        mtype, body, _ = wire.read_frame(raw)
        assert mtype == MsgType.ERROR
        assert body["code"] == "oversized" and body["fatal"] is True
    finally:
        raw.close()


def test_malformed_json_keeps_connection_open(server):
    raw = socket.create_connection(server.server_address, timeout=5.0)
    try:
        raw.sendall(wire.pack_frame(MsgType.HELLO, {"type": "hello", "proto": 1, "pcm": "f32le"}))
        wire.read_frame(raw)
        bad = b"{not json"
        raw.sendall(wire.HEADER.pack(wire.MAGIC, 1, int(MsgType.EVENT), 0, 0, len(bad), 0) + bad)
        mtype, body, _ = wire.read_frame(raw)
        assert mtype == MsgType.ERROR
        assert body["code"] == "bad_json" and body["fatal"] is False
        raw.sendall(wire.pack_frame(MsgType.PING, {"type": "ping", "t": 1.0}))
        mtype, body, _ = wire.read_frame(raw)
        assert mtype == MsgType.PONG and body["t"] == 1.0
    finally:
        raw.close()


def test_reconnect_rehandshakes_warm_cache_persists(server):
    with SfxClient(*server.server_address) as c:
        c.context(0.0, "gun", ["grass"], "field")
    with SfxClient(*server.server_address) as c2:
        meta, _ = c2.event("shoot", "metal", "normal")
        assert meta["served"] == "cached"


def test_disconnect_midsession_server_survives(server):
    raw = socket.create_connection(server.server_address, timeout=5.0)
    raw.sendall(b"PSFX\x01")
    raw.close()
    with SfxClient(*server.server_address) as c:
        assert c.ping(t=1.0)["t"] == 1.0


def test_event_before_hello_is_non_fatal(server):
    raw = socket.create_connection(server.server_address, timeout=5.0)
    try:
        raw.sendall(wire.pack_frame(MsgType.EVENT, {"type": "event", "event_type": "shoot", "material": "metal"}))
        mtype, body, _ = wire.read_frame(raw)
        assert mtype == MsgType.ERROR and body["fatal"] is False
        raw.sendall(wire.pack_frame(MsgType.HELLO, {"type": "hello", "proto": 1, "pcm": "f32le"}))
        mtype, body, _ = wire.read_frame(raw)
        assert mtype == MsgType.WELCOME
    finally:
        raw.close()


def test_idle_connection_survives_recv_timeout():
    srv = serve(generator="mock", host="127.0.0.1", port=0, recv_timeout=0.3,
                vram_budget_kb=50_000, gpu_ms_budget=10_000_000, variants_per_key=3)
    try:
        with SfxClient(*srv.server_address) as c:
            c.context(0.0, "gun", ["grass"], "field")
            time.sleep(0.6)
            meta, _ = c.event("shoot", "metal", "normal")
            assert meta["served"] == "cached"
    finally:
        srv.shutdown()
        srv.server_close()


def test_wrong_typed_fields_are_bad_fields(server):
    with SfxClient(*server.server_address) as c:
        with pytest.raises(ProtocolError) as ei:
            c.context("notanumber", "gun", ["grass"])
        assert ei.value.code == "bad_fields"
        with pytest.raises(ProtocolError) as ej:
            c.context(0.0, "gun", ["grass"], gpu_pressure="x")
        assert ej.value.code == "bad_fields"
        assert c.ping(t=1.0)["t"] == 1.0


def test_non_dict_body_is_bad_fields(server):
    raw = socket.create_connection(server.server_address, timeout=5.0)
    try:
        raw.sendall(wire.pack_frame(MsgType.HELLO, 5))
        mtype, body, _ = wire.read_frame(raw)
        assert mtype == MsgType.ERROR and body["code"] == "bad_fields" and body["fatal"] is False
    finally:
        raw.close()


def test_prebake_cap_rejected(server):
    with SfxClient(*server.server_address) as c:
        ctxs = [{"time_s": 0.0, "equipped": None, "surfaces": ["grass"]} for _ in range(300)]
        with pytest.raises(ProtocolError) as ei:
            c.prebake(ctxs)
        assert ei.value.code == "bad_fields"


def test_i16le_roundtrips_engine_last_served(server):
    with SfxClient(*server.server_address, pcm="i16le") as c:
        c.context(0.0, "gun", ["grass"], "field")
        meta, samples = c.event("shoot", "metal", "normal")
    assert meta["format"] == "i16le"
    assert len(samples) == meta["n_samples"]
    assert meta["byte_len"] == meta["n_samples"] * 2
    ref = server.engine.last_served.samples
    assert np.max(np.abs(samples - ref)) <= 1.0 / 32767 + 1e-6


def test_ping_works_while_engine_lock_held(server):
    with SfxClient(*server.server_address, timeout=3.0) as c:
        with sidecar.engine_lock:
            pong = c.ping(t=5.0)
        assert pong["t"] == 5.0


def test_unknown_opcode_is_non_fatal_and_keeps_sync(server):
    raw = socket.create_connection(server.server_address, timeout=5.0)
    try:
        raw.sendall(wire.pack_frame(MsgType.HELLO, {"type": "hello", "proto": 1, "pcm": "f32le"}))
        wire.read_frame(raw)
        raw.sendall(wire.HEADER.pack(wire.MAGIC, 1, 100, 0, 0, 2, 0) + b"{}")
        mtype, body, _ = wire.read_frame(raw)
        assert mtype == MsgType.ERROR and body["code"] == "unknown_type" and body["fatal"] is False
        raw.sendall(wire.pack_frame(MsgType.PING, {"type": "ping", "t": 2.0}))
        mtype, body, _ = wire.read_frame(raw)
        assert mtype == MsgType.PONG and body["t"] == 2.0
    finally:
        raw.close()


class _BigEngine:
    def __init__(self, n):
        self.gen = type("g", (), {"name": "stub", "sample_rate": 48000})()
        self.cache = {}
        self.stats = {"spec_gen": 0, "demand_gen": 0, "hits": 0, "fallback": 0,
                      "evicted": 0, "gpu_ms": 0.0, "peak_bytes": 0}
        self.last_served = None
        self._n = n

    def on_context_change(self, ctx, gpu_pressure=1.0):
        pass

    def on_event(self, key):
        rng = np.random.default_rng(7)
        self.last_served = AudioClip((rng.standard_normal(self._n).astype(np.float32) * 0.5), 48000, 0)
        return "fallback"

    def wasted(self):
        return 0


def test_large_pcm_multipacket_roundtrip():
    n = 40000
    srv = serve(engine=_BigEngine(n), host="127.0.0.1", port=0)
    try:
        with SfxClient(*srv.server_address) as c:
            meta, samples = c.event("boom", "metal", "hard")
        assert meta["n_samples"] == n
        assert meta["byte_len"] == n * 4
        assert meta["byte_len"] > wire.RECV_CHUNK
        assert np.array_equal(samples, srv.engine.last_served.samples)
    finally:
        srv.shutdown()
        srv.server_close()
