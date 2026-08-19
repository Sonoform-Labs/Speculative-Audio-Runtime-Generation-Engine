import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from sage.generators import MockGenerator, get_generator
from sage.types import CacheKey

ROOT = Path(__file__).resolve().parent.parent


def _start_server(backend="mock"):
    p = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "model_server.py"),
         "--backend", backend, "--port", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT), text=True)
    port = None
    deadline = time.time() + 60
    while time.time() < deadline:
        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                break
            continue
        if "READY" in line and ":" in line:
            port = int(line.strip().split(":")[-1])
            break
    if port is None:
        p.terminate()
        pytest.fail("model-server did not start")
    return p, port


def _stop(p):
    p.terminate()
    try:
        p.wait(timeout=5)
    except Exception:
        p.kill()


def test_remote_generator_matches_local_mock():
    p, port = _start_server("mock")
    try:
        rg = get_generator("remote", port=port)
        rg.ensure_ready()
        key = CacheKey("footstep", "wood", "normal")
        clip = rg.generate(key, variant=3)
        ref = MockGenerator().generate(key, variant=3)
        assert clip.sample_rate == ref.sample_rate
        assert clip.samples.shape == ref.samples.shape
        assert np.allclose(clip.samples, ref.samples, atol=1e-6)
        rg.close()
    finally:
        _stop(p)


def test_remote_generator_drives_the_engine():
    from sage.affordance import AffordanceEngine, GameContext
    p, port = _start_server("mock")
    try:
        eng = AffordanceEngine(get_generator("remote", port=port),
                               variants_per_key=2, vram_budget_kb=100_000, gpu_ms_budget=1e9)
        eng.on_context_change(GameContext(0.0, None, ["grass"], "field"))
        assert eng.on_event(CacheKey("footstep", "grass", "normal")) == "cached"
    finally:
        _stop(p)
