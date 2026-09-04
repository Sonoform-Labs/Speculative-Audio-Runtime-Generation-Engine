import pytest

from sarge.affordance import AffordanceEngine
from sarge.fusion import FusedProposal
from sarge.generators import MockGenerator
from sarge.sidecar import serve, SfxClient
from sarge.types import CacheKey


def test_register_predictions_queues_and_lead_gates():
    eng = AffordanceEngine(MockGenerator(), variants_per_key=2, gen_cost_ms=393.0,
                           vram_budget_kb=100_000, gpu_ms_budget=1e9)
    good = FusedProposal(CacheKey("footstep", "grass", "normal"), 0.6, 0.9, ("motion",))
    late = FusedProposal(CacheKey("footstep", "metal", "normal"), 0.05, 0.9, ("motion",))
    res = eng.register_predictions([good, late], now=0.0)
    assert res["queued"] == 1
    assert res["lead_gated"] == 1
    assert eng.drain_demand()
    assert CacheKey("footstep", "grass", "normal") in eng.cache
    assert CacheKey("footstep", "metal", "normal") not in eng.cache


@pytest.fixture
def server():
    srv = serve(generator="mock", host="127.0.0.1", port=0,
                vram_budget_kb=50_000, gpu_ms_budget=10_000_000, variants_per_key=2)
    yield srv
    srv.shutdown()
    srv.server_close()


def test_predict_message_queues_and_acks(server):
    with SfxClient(*server.server_address) as c:
        c.context(0.0, None, ["grass"], "field")
        ack = c.predict([
            {"source": "motion", "event_type": "footstep", "material": "wood",
             "intensity": "normal", "eta_s": 0.6, "confidence": 0.9},
            {"source": "motion", "event_type": "footstep", "material": "stone",
             "intensity": "normal", "eta_s": 0.05, "confidence": 0.9},
        ], now=0.0)
    assert ack["queued"] == 1
    assert ack["lead_gated"] == 1


def test_predict_then_background_warms_cache(server):
    import time
    server.deepening = True
    import threading
    import sarge.sidecar.server as S
    t = threading.Thread(target=S._deepen_loop, args=(server,), daemon=True)
    t.start()
    try:
        with SfxClient(*server.server_address) as c:
            c.context(0.0, None, ["grass"], "field")
            c.predict([{"source": "motion", "event_type": "footstep", "material": "wood",
                        "eta_s": 0.6, "confidence": 0.9}], now=0.0)
            k = CacheKey("footstep", "wood", "normal")
            deadline = time.time() + 3.0
            while time.time() < deadline and k not in server.engine.cache:
                time.sleep(0.02)
            assert k in server.engine.cache
            meta, _ = c.event("footstep", "wood", "normal")
            assert meta["served"] == "cached"
    finally:
        server.deepening = False
