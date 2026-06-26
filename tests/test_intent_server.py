import pytest

from predictive_sfx.sidecar import serve, SfxClient


@pytest.fixture
def server():
    srv = serve(generator="mock", host="127.0.0.1", port=0,
                vram_budget_kb=50_000, gpu_ms_budget=10_000_000, variants_per_key=2)
    yield srv
    srv.shutdown()
    srv.server_close()


def test_intent_learns_pattern_and_surfaces_next(server):
    with SfxClient(*server.server_address) as c:
        c.context(0.0, "gun", ["grass"], "combat")
        for _ in range(8):
            c.event("shoot", "metal", "normal", "combat")
            c.event("shoot", "metal", "normal", "combat")
            c.event("reload", "metal", "normal", "combat")
        c.event("shoot", "metal", "normal", "combat")
        meta, _ = c.event("shoot", "metal", "normal", "combat")
    assert meta.get("intent", "").startswith("reload")


def test_intent_calibration_receives_feedback(server):
    with SfxClient(*server.server_address) as c:
        c.context(0.0, "gun", ["grass"], "combat")
        for _ in range(8):
            c.event("shoot", "metal", "normal", "combat")
            c.event("reload", "metal", "normal", "combat")
    cal = server.calib.cals.get("intent")
    assert cal is not None and cal.total > 0


def test_footsteps_do_not_drive_intent(server):
    with SfxClient(*server.server_address) as c:
        c.context(0.0, None, ["grass"], "field")
        for _ in range(6):
            c.event("footstep", "grass", "normal", "field")
    assert "footstep" not in server.intent.vocab


def test_existing_event_response_unaffected(server):
    with SfxClient(*server.server_address) as c:
        c.context(0.0, None, ["grass"], "field")
        meta, samples = c.event("explosion", "metal", "hard")
        assert meta["served"] == "fallback"
        assert len(samples) > 0
