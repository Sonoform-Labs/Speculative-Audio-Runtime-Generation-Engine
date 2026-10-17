import pytest


@pytest.fixture(autouse=True)
def _no_startup_side_effects(monkeypatch):
    monkeypatch.setenv("SARGE_PREFETCH", "0")
    monkeypatch.setenv("SARGE_WOOSH_EARLY", "0")
    monkeypatch.setenv("SARGE_WEIGHT_CACHE", "off")
