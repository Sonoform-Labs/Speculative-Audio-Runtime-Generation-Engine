import json
import threading
import time

import numpy as np

from sarge.affordance import AffordanceEngine
from sarge.fusion import FusedProposal
from sarge.generators.mock import MockGenerator
from sarge.sidecar import SfxClient, serve
from sarge.sidecar.server import engine_lock
from sarge.types import CacheKey


class SlowGen(MockGenerator):
    def __init__(self, slow_s=0.15, **kw):
        super().__init__(**kw)
        self.slow_s = slow_s
        self.calls = []

    def generate(self, key, *, variant=0):
        self.calls.append((str(key), variant))
        if key.event_type == "slow":
            time.sleep(self.slow_s)
        return super().generate(key, variant=variant)


def _srv(gen=None, **kw):
    eng = AffordanceEngine(gen or SlowGen(), realtime=True, vram_budget_kb=200_000, authored=set())
    return serve(engine=eng, host="127.0.0.1", port=0, warmup=False, **kw)


def _stop(srv):
    srv.deepening = False
    srv.shutdown()
    srv.server_close()


def _timed_cached_during_slow(srv):
    out = {}
    with SfxClient(*srv.server_address) as c:
        c.event("fast", "wood")

    def slow():
        with SfxClient(*srv.server_address) as c2:
            out["slow"] = c2.event("slow", "metal")

    t = threading.Thread(target=slow)
    with SfxClient(*srv.server_address) as c:
        t.start()
        time.sleep(0.03)
        t0 = time.perf_counter()
        meta, _ = c.event("fast", "wood")
        out["fast_ms"] = (time.perf_counter() - t0) * 1000.0
        out["fast_meta"] = meta
    t.join()
    return out


def test_concurrent_gen_serves_cached_keys_while_a_miss_generates():
    srv = _srv(concurrent_gen=True)
    try:
        out = _timed_cached_during_slow(srv)
    finally:
        _stop(srv)
    assert out["fast_meta"]["served"] == "cached"
    assert out["fast_ms"] < 60.0
    assert out["slow"][0]["served"] == "realtime"


def test_without_concurrent_gen_a_cached_key_waits_for_the_miss():
    srv = _srv()
    try:
        out = _timed_cached_during_slow(srv)
    finally:
        _stop(srv)
    assert out["fast_ms"] > 80.0


def test_concurrent_gen_clips_and_stats_match_a_serial_run():
    seq = [("slow", "metal"), ("fast", "wood"), ("slow", "metal"), ("other", "stone"), ("fast", "wood")]
    results = []
    for flag in (False, True):
        srv = _srv(SlowGen(slow_s=0.01), concurrent_gen=flag)
        try:
            with SfxClient(*srv.server_address) as c:
                got = [c.event(e, m) for e, m in seq]
            st = dict(srv.engine.stats)
            inflight = set(srv.engine.inflight)
        finally:
            _stop(srv)
        results.append((got, st, inflight))
    (a, sa, ia), (b, sb, ib) = results
    for (ma, xa), (mb, xb) in zip(a, b):
        assert ma == mb and np.array_equal(xa, xb)
    assert sa == sb and ia == ib == set()


def test_concurrent_duplicate_requests_generate_once():
    gen = SlowGen(slow_s=0.1)
    srv = _srv(gen, concurrent_gen=True)
    metas = []
    try:
        def go():
            with SfxClient(*srv.server_address) as c:
                metas.append(c.event("slow", "metal")[0]["served"])
        ts = [threading.Thread(target=go) for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        _stop(srv)
    assert gen.calls.count(("slow|metal|normal|default", 0)) == 1
    assert sorted(metas) == ["cached", "cached", "realtime"]


def test_concurrent_gen_failure_releases_the_claim():
    class Boom(SlowGen):
        def generate(self, key, *, variant=0):
            if key.event_type == "boom":
                raise OSError("gpu fell over")
            return super().generate(key, variant=variant)

    srv = _srv(Boom(), concurrent_gen=True)
    try:
        with SfxClient(*srv.server_address) as c:
            try:
                c.event("boom", "metal")
            except Exception as e:
                assert "generation failed" in str(e)
            meta, _ = c.event("fast", "wood")
        assert meta["served"] == "realtime"
        assert not srv.engine.inflight
    finally:
        _stop(srv)


def test_event_log_writes_one_line_per_request(tmp_path):
    log = tmp_path / "events.jsonl"
    srv = _srv(event_log=str(log))
    try:
        with SfxClient(*srv.server_address) as c:
            c.event("fast", "wood", "hard")
            c.event("fast", "wood", "hard")
            c.predict([{"event_type": "next", "material": "wood", "eta_s": 1.0, "confidence": 0.9}])
    finally:
        _stop(srv)
        srv.event_log.close()
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert [r["type"] for r in rows] == ["event", "event", "predict"]
    first, second = rows[0], rows[1]
    for f in ("t_recv", "key", "served", "lock_wait_ms", "gen_ms", "encode_ms", "bytes", "engine_ms", "send_ms"):
        assert f in first
    assert first["served"] == "realtime" and second["served"] == "cached"
    assert first["key"] == "fast|wood|hard|default" and first["bytes"] > 0
    assert rows[2]["keys"] == ["next|wood|normal|default"]


def test_no_event_log_by_default():
    srv = _srv()
    try:
        assert srv.event_log is None and srv.predict_fixes is False and srv.concurrent_gen is False
        assert srv.engine.lead_gate is True and srv.engine.pool_fix is False
    finally:
        _stop(srv)


def test_have_gets_a_meta_only_reply_and_old_clients_get_pcm():
    srv = _srv()
    try:
        with SfxClient(*srv.server_address) as c:
            m1, x1 = c.event("fast", "wood")
            m2, x2 = c.event("fast", "wood", have=[m1["variant_id"]])
            m3, x3 = c.event("fast", "wood", have=[99])
            m4, x4 = c.event("fast", "wood")
    finally:
        _stop(srv)
    assert m2["meta_only"] is True and m2["byte_len"] == 0 and len(x2) == 0
    assert m2["n_samples"] == m1["n_samples"] and m2["variant_id"] == m1["variant_id"]
    assert "meta_only" not in m3 and np.array_equal(x3, x1)
    assert "meta_only" not in m4 and m4["byte_len"] == m1["byte_len"]


def test_have_with_a_different_phys_key_still_sends_pcm():
    srv = _srv(world=True)
    phys = {"a": {"shape": "sphere", "ext": [0.6] * 3, "m": 1, "dyn": 1, "bnc": 1},
            "b": {"shape": "box", "ext": [10, 0.2, 10], "tag": "metal"}, "vn": 9}
    try:
        with SfxClient(*srv.server_address) as c:
            m1, _ = c.event("impact", "metal", "hard", phys=phys)
            m2, x2 = c._round_trip(5, {"type": "event", "event_type": "impact", "material": "metal",
                                       "intensity": "hard", "phys": phys, "have": [m1["variant_id"]],
                                       "have_phys": "hit;a=other"})[1:]
            m3, _ = c.event("impact", "metal", "hard", phys=phys, have=[m1["variant_id"]], have_phys=m1["phys_key"])
            m4, x4 = c.event("impact", "metal", "hard", phys=phys, have=[m1["variant_id"]])
    finally:
        _stop(srv)
    assert "meta_only" not in m2 and len(x2) > 0
    assert m3["meta_only"] is True and m3["phys_key"] == m1["phys_key"]
    assert "meta_only" not in m4 and len(x4) > 0


def test_predict_fixes_alias_a_predicted_stem_to_the_fired_event():
    srv = _srv(predict_fixes=True)
    try:
        with SfxClient(*srv.server_address) as c:
            c.event("footstep", "wood")
            c.event("iron boot footstep", "wood", "hard")
            c.predict([{"source": "motion", "event_type": "footstep", "material": "stone", "eta_s": 0.6}])
            c.event("footstep", "grass")
            c.predict([{"source": "motion", "event_type": "footstep", "material": "metal", "eta_s": 0.6}])
            c.event("iron boot landing", "dirt", "soft")
            c.predict([{"source": "motion", "event_type": "land", "material": "sand", "eta_s": 0.6}])
        queued = set(srv.engine.demand) | set(srv.engine.cache) | {k for k, _v in srv.engine.inflight}
    finally:
        _stop(srv)
    assert CacheKey("iron boot footstep", "stone", "hard") in queued
    assert CacheKey("footstep", "metal", "normal") in queued
    assert CacheKey("iron boot landing", "sand", "soft") in queued


def test_without_predict_fixes_the_prediction_is_taken_literally():
    srv = _srv()
    try:
        with SfxClient(*srv.server_address) as c:
            c.event("iron boot footstep", "wood", "hard")
            ack = c.predict([{"source": "motion", "event_type": "footstep", "material": "stone", "eta_s": 0.6}])
        queued = set(srv.engine.demand) | set(srv.engine.cache)
    finally:
        _stop(srv)
    assert ack["queued"] == 1
    assert CacheKey("footstep", "stone", "normal") in queued


def test_predict_fixes_turn_off_the_lead_gate_in_realtime_mode():
    eng = AffordanceEngine(MockGenerator(), realtime=True, gen_cost_ms=150.0)
    fp = FusedProposal(CacheKey("shoot", "rifle", "normal"), 0.05, 0.9, ("equip",))
    assert eng.register_predictions([fp])["lead_gated"] == 1
    eng.lead_gate = False
    r = eng.register_predictions([fp])
    assert r["lead_gated"] == 0 and r["queued"] == 1
    srv = _srv(predict_fixes=True)
    try:
        assert srv.engine.lead_gate is False
    finally:
        _stop(srv)


def test_predict_fixes_expand_after_to_learned_target_successors():
    srv = _srv(predict_fixes=True)
    try:
        with SfxClient(*srv.server_address) as c:
            for _ in range(3):
                c.event("shoot", "rifle")
                c.event("bullet impact", "metal")
            c.event("shoot", "rifle")
            c.event("reload", "rifle")
            ack = c.predict([{"source": "aim", "event_type": "bullet impact", "material": "wood", "after": "shoot",
                              "eta_s": 1.0, "confidence": 0.9}])
        queued = set(srv.engine.demand) | set(srv.engine.cache) | {k for k, _v in srv.engine.inflight}
    finally:
        _stop(srv)
    assert CacheKey("bullet impact", "wood", "normal") in queued
    assert not any(k.event_type == "reload" and k.material == "wood" for k in queued)
    assert ack["queued"] >= 1


def test_pool_fix_requeues_until_every_variant_is_planned():
    def eng(flag):
        e = AffordanceEngine(MockGenerator(), realtime=True, variants_per_key=3, vram_budget_kb=200_000)
        e.pool_fix = flag
        e.register_predictions([FusedProposal(CacheKey("a", "b", "normal"), 5.0, 0.9, ("m",))])
        planned = []
        for _ in range(6):
            batch, _stat = e.plan_batch(max_items=1)
            if not batch:
                break
            planned += batch
            e.commit_clips(batch, e.gen.generate_batch(batch), "demand_gen")
        return planned, len(e.cache[CacheKey("a", "b", "normal")])

    head, n_head = eng(False)
    fixed, n_fixed = eng(True)
    assert n_head == 1 and [v for _k, v in head] == [0]
    assert n_fixed == 3 and [v for _k, v in fixed] == [0, 1, 2]


def test_predict_fixes_skip_the_idle_between_demand_items():
    def run(flag):
        gen = SlowGen(slow_s=0.0)
        srv = _srv(gen, predict_fixes=flag, deepen=True)
        try:
            with SfxClient(*srv.server_address) as c:
                t0 = time.perf_counter()
                c.predict([{"source": "m", "event_type": "e%d" % i, "material": "wood", "eta_s": 5.0}
                           for i in range(6)])
                deadline = t0 + 3.0
                while time.perf_counter() < deadline:
                    with engine_lock:
                        n = sum(1 for k in srv.engine.cache if k.event_type.startswith("e"))
                    if n >= 6:
                        break
                    time.sleep(0.002)
                return time.perf_counter() - t0
        finally:
            _stop(srv)

    slow = run(False)
    fast = run(True)
    assert slow > 0.2
    assert fast < slow * 0.6


def test_sibling_on_miss_serves_a_cached_neighbour_and_queues_the_exact_key():
    srv = _srv(sibling_on_miss=True)
    try:
        with SfxClient(*srv.server_address) as c:
            m0, x0 = c.event("impact", "metal", "normal")
            m1, x1 = c.event("impact", "metal", "hard")
            m2, _ = c.event("splash", "water")
        assert m0["served"] == "realtime"
        assert m1["served"] == "sibling" and np.array_equal(x1, x0)
        assert m2["served"] == "realtime"
        assert CacheKey("impact", "metal", "hard") in srv.engine.demand_set
        assert srv.engine.stats["sibling"] == 1
    finally:
        _stop(srv)


def test_no_sibling_by_default():
    srv = _srv()
    try:
        with SfxClient(*srv.server_address) as c:
            c.event("impact", "metal", "normal")
            m1, _ = c.event("impact", "metal", "hard")
        assert m1["served"] == "realtime" and "sibling" not in srv.engine.stats
    finally:
        _stop(srv)


def test_keep_warm_runs_tiny_jobs_only_while_a_client_is_connected():
    gen = SlowGen(slow_s=0.0)
    srv = _srv(gen, keep_warm_ms=40.0)
    try:
        time.sleep(0.25)
        assert srv.warm_jobs == 0
        with SfxClient(*srv.server_address) as c:
            c.ping()
            time.sleep(0.3)
            busy = srv.warm_jobs
        time.sleep(0.15)
        after = srv.warm_jobs
        time.sleep(0.25)
        assert busy >= 2
        assert srv.warm_jobs == after
        assert all(k.startswith("warmup|") for k, _v in gen.calls)
        assert not any(k.event_type == "warmup" for k in srv.engine.cache)
    finally:
        srv.keep_warm_s = 0.0
        _stop(srv)


def test_sibling_on_miss_wins_over_concurrent_gen_when_a_sibling_exists():
    srv = _srv(sibling_on_miss=True, concurrent_gen=True)
    try:
        with SfxClient(*srv.server_address) as c:
            m0, _ = c.event("impact", "metal", "normal")
            m1, _ = c.event("impact", "metal", "hard")
        assert m0["served"] == "realtime" and m1["served"] == "sibling"
        assert not srv.engine.inflight
    finally:
        _stop(srv)


PANE = {"a": {"shape": "box", "ext": [1.6, 2.0, 0.08], "dyn": 0, "name": "window"}, "kind": "break"}
BALL = {"shape": "sphere", "ext": [0.6, 0.6, 0.6], "m": 1.0, "dyn": 1, "bnc": 1.0}


def _tile(mat):
    return {"a": BALL, "b": {"shape": "box", "ext": [10, 0.2, 10], "dyn": 0, "tag": mat}, "vn": 9.0, "n": [0, 1, 0]}


def _queued(srv):
    with engine_lock:
        return set(srv.engine.demand) | set(srv.engine.cache) | {k for k, _v in srv.engine.inflight}


def test_world_predictions_and_intent_without_phys_take_the_last_fired_bucket():
    srv = _srv(world=True)
    try:
        with SfxClient(*srv.server_address) as c:
            m0, _ = c.event("break", "window", "normal", phys=PANE)
            ack = c.predict([{"source": "aim", "event_type": "break", "material": "window", "intensity": "hard",
                              "eta_s": 0.5}])
            for _ in range(3):
                c.event("shoot", "pistol")
                c.event("break", "window", "normal", phys=PANE)
            c.event("shoot", "pistol")
        queued = _queued(srv)
        pending = [k for k, _p in srv.intent_pending]
    finally:
        _stop(srv)
    b = m0["phys_key"]
    assert b.startswith("break;") and "phys_keys" not in ack
    assert CacheKey("break", "window", "hard", "default", b) in queued
    assert not any(k.event_type == "break" and not k.phys for k in queued)
    assert CacheKey("break", "window", "normal", "default", b) in pending


def test_world_off_predictions_stay_plain():
    srv = _srv()
    try:
        with SfxClient(*srv.server_address) as c:
            c.event("break", "window", "normal", phys=PANE)
            c.predict([{"source": "aim", "event_type": "break", "material": "window", "intensity": "hard",
                        "eta_s": 0.5}])
        queued = _queued(srv)
    finally:
        _stop(srv)
    assert CacheKey("break", "window", "hard") in queued
    assert not any(k.phys for k in queued)


def test_world_sibling_needs_the_same_physics_bucket():
    srv = _srv(world=True, sibling_on_miss=True)
    try:
        with SfxClient(*srv.server_address) as c:
            m0, x0 = c.event("impact", "metal", "hard", phys=_tile("metal"))
            m1, _ = c.event("impact", "glass", "hard", phys=_tile("glass"))
            m2, x2 = c.event("impact", "metal", "normal", phys=_tile("metal"))
    finally:
        _stop(srv)
    assert m0["served"] == "realtime" and m1["served"] == "realtime"
    assert m2["served"] == "sibling" and np.array_equal(x2, x0)


def test_a_sibling_reply_always_carries_pcm():
    srv = _srv(sibling_on_miss=True)
    try:
        with SfxClient(*srv.server_address) as c:
            m0, _ = c.event("impact", "metal", "normal")
            m1, x1 = c.event("impact", "metal", "hard", have=[m0["variant_id"]])
    finally:
        _stop(srv)
    assert m1["served"] == "sibling" and m1["variant_id"] == m0["variant_id"]
    assert "meta_only" not in m1 and len(x1) > 0


def test_alias_needs_the_stem_at_the_end_and_skips_empty_names():
    from types import SimpleNamespace
    from sarge.sidecar.server import _alias
    s = SimpleNamespace(fired_at={"landmine trigger": 3, "iron boot landing": 2, "shoot": 4})
    assert _alias(s, "land") == "iron boot landing"
    assert _alias(s, "") == ""
    assert _alias(SimpleNamespace(fired_at={"landmine": 5}), "land") == "land"
    assert _alias(SimpleNamespace(fired_at={"iron boot footsteps": 1}), "footstep") == "iron boot footsteps"


def test_a_failed_event_log_write_turns_the_log_off_and_keeps_the_connection(tmp_path):
    class Broken:
        def write(self, s):
            raise OSError("disk full")

        def flush(self):
            pass

    srv = _srv(event_log=str(tmp_path / "e.jsonl"))
    real = srv.event_log
    srv.event_log = Broken()
    try:
        with SfxClient(*srv.server_address) as c:
            m1, _ = c.event("fast", "wood")
            m2, _ = c.event("fast", "wood")
    finally:
        _stop(srv)
        real.close()
    assert m1["served"] == "realtime" and m2["served"] == "cached"
    assert srv.event_log is None


def test_event_log_gen_ms_belongs_to_the_served_clip(tmp_path):
    class Timed(SlowGen):
        def generate(self, key, *, variant=0):
            clip = super().generate(key, variant=variant)
            clip.gen_ms = 150.0 if key.event_type == "slow" else 5.0
            return clip

    log = tmp_path / "e.jsonl"
    srv = _srv(Timed(), concurrent_gen=True, event_log=str(log))
    try:
        def slow():
            with SfxClient(*srv.server_address) as c2:
                c2.event("slow", "metal")

        t = threading.Thread(target=slow)
        t.start()
        time.sleep(0.03)
        with SfxClient(*srv.server_address) as c:
            c.event("fast", "wood")
        t.join()
    finally:
        _stop(srv)
        srv.event_log.close()
    rows = {r["key"]: r for r in (json.loads(x) for x in log.read_text().splitlines())}
    assert rows["slow|metal|normal|default"]["gen_ms"] == 150.0
    assert rows["fast|wood|normal|default"]["gen_ms"] == 5.0


def test_concurrent_gen_records_a_slow_miss_in_arrival_order():
    srv = _srv(SlowGen(slow_s=0.15), concurrent_gen=True, predict_fixes=True)
    try:
        def slow():
            with SfxClient(*srv.server_address) as c2:
                c2.event("slow", "rifle")

        t = threading.Thread(target=slow)
        t.start()
        time.sleep(0.03)
        with SfxClient(*srv.server_address) as c:
            c.event("fast", "metal")
        t.join()
    finally:
        _stop(srv)
    assert srv.fired_at["slow"] < srv.fired_at["fast"]
    assert srv.intent.history == ["slow", "fast"]


def test_pool_fix_skips_variants_already_cached_or_in_flight():
    e = AffordanceEngine(MockGenerator(), realtime=True, variants_per_key=3, vram_budget_kb=200_000)
    e.pool_fix = True
    k = CacheKey("a", "b", "normal")
    assert e.claim_realtime(k)
    e.register_predictions([FusedProposal(k, 5.0, 0.9, ("m",))])
    planned = []
    for _ in range(2):
        b, _stat = e.plan_batch(max_items=1)
        planned += b
        e.commit_clips(b, e.gen.generate_batch(b), "demand_gen")
    e.complete_realtime(k, e.gen.generate(k, variant=0))
    rest, _stat = e.plan_batch(max_items=1)
    assert planned == [(k, 1), (k, 2)] and rest == []
    assert sorted(c.variant_id for c in e.cache[k]) == [0, 1, 2]
