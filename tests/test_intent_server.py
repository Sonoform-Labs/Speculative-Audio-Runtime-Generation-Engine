import pytest

from sarge.sidecar import serve, SfxClient


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def server(clock):
    srv = serve(generator="mock", host="127.0.0.1", port=0,
                vram_budget_kb=50_000, gpu_ms_budget=10_000_000, variants_per_key=2, clock=clock)
    yield srv
    srv.shutdown()
    srv.server_close()


def _fire(c, clock, gap, *ev):
    clock.t += gap
    return c.event(*ev)


def test_intent_learns_pattern_and_surfaces_next(server, clock):
    with SfxClient(*server.server_address) as c:
        c.context(0.0, "gun", ["grass"], "combat")
        for _ in range(8):
            _fire(c, clock, 1.0, "shoot", "metal", "normal", "combat")
            _fire(c, clock, 1.0, "shoot", "metal", "normal", "combat")
            _fire(c, clock, 1.0, "reload", "metal", "normal", "combat")
        _fire(c, clock, 1.0, "shoot", "metal", "normal", "combat")
        meta, _ = _fire(c, clock, 1.0, "shoot", "metal", "normal", "combat")
    assert meta.get("intent", "").startswith("reload")


def test_intent_calibration_receives_feedback(server, clock):
    with SfxClient(*server.server_address) as c:
        c.context(0.0, "gun", ["grass"], "combat")
        for _ in range(8):
            _fire(c, clock, 1.0, "shoot", "metal", "normal", "combat")
            _fire(c, clock, 1.0, "reload", "metal", "normal", "combat")
    cal = server.calib.cals.get("intent")
    assert cal is not None and cal.total > 0


def test_rapid_footsteps_do_not_drive_intent(server, clock):
    with SfxClient(*server.server_address) as c:
        c.context(0.0, None, ["grass"], "field")
        for _ in range(6):
            _fire(c, clock, 0.45, "Footstep_Grass", "grass", "normal", "field")
    assert server.intent.unigram["field"] == {"Footstep_Grass": 1}
    assert not any("Footstep_Grass" in hist for (_ctx, hist), d in server.intent.ngram.items()
                   if "Footstep_Grass" in d)


def test_any_name_firing_faster_than_the_rate_is_left_out_and_slow_ones_count(server, clock):
    with SfxClient(*server.server_address) as c:
        for _ in range(5):
            _fire(c, clock, 0.3, "dog_bark", "dog", "normal", "yard")
        for _ in range(3):
            _fire(c, clock, 2.0, "door_creak", "wood", "normal", "yard")
    u = server.intent.unigram["yard"]
    assert u == {"dog_bark": 1, "door_creak": 3}


def test_rate_filter_off_counts_every_event(clock):
    srv = serve(generator="mock", host="127.0.0.1", port=0, vram_budget_kb=50_000, rapid_hz=0.0, clock=clock)
    try:
        with SfxClient(*srv.server_address) as c:
            for _ in range(4):
                _fire(c, clock, 0.1, "footstep", "grass", "normal", "field")
        assert srv.intent.unigram["field"] == {"footstep": 4}
    finally:
        srv.shutdown()
        srv.server_close()


def test_existing_event_response_unaffected(server):
    with SfxClient(*server.server_address) as c:
        c.context(0.0, None, ["grass"], "field")
        meta, samples = c.event("explosion", "metal", "hard")
        assert meta["served"] == "fallback"
        assert len(samples) > 0
