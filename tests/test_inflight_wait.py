import threading
import time

from sarge.affordance import AffordanceEngine
from sarge.generators.mock import MockGenerator
from sarge.sidecar import serve, SfxClient


class SlowBatchGen(MockGenerator):
    def __init__(self):
        super().__init__()
        self.batch_started = threading.Event()
        self.batches = 0
        self.singles = 0

    def generate(self, key, *, variant=0):
        self.singles += 1
        return super().generate(key, variant=variant)

    def generate_batch(self, items):
        self.batches += 1
        self.batch_started.set()
        time.sleep(0.25)
        return [super(SlowBatchGen, self).generate(k, variant=v) for k, v in items]


def test_event_for_a_key_being_generated_waits_for_it_and_never_regenerates():
    gen = SlowBatchGen()
    eng = AffordanceEngine(gen, realtime=True, vram_budget_kb=200_000, gen_cost_ms=50.0)
    server = serve(engine=eng, port=0, warmup=False, deepen=True, deepen_batch=2)
    try:
        host, port = server.server_address
        with SfxClient(host, port) as c:
            c.hello()
            ack = c.predict([{"event_type": "footstep", "material": "metal", "eta_s": 1.0,
                              "confidence": 0.9, "source": "motion"}], now=0.0)
            assert ack["queued"] == 1
            assert gen.batch_started.wait(3.0)
            body, samples = c.event("footstep", "metal")
        assert body["served"] == "cached"
        assert gen.singles == 0
        assert len(samples) > 0
    finally:
        server.deepening = False
        server.shutdown()
        server.server_close()
