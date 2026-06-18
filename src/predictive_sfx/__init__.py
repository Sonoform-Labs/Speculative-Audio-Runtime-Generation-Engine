"""Predictive Neural SFX — speculative prefetching for sound.

The core idea: neural audio generation is far too slow to call the instant a
sound must play (tens of ms to seconds, vs a ~10 ms audio-callback budget).
So instead of generating on the hot path, we:

    1. PREDICT which sound events are about to fire (from game state),
    2. PRE-GENERATE context-specific variations ahead of time,
    3. CACHE them keyed by (event, material, intensity, context),
    4. FALL BACK to authored assets instantly on a miss.

This package is a standalone simulation/MVP harness for validating the two
risks that decide the whole idea: prediction hit-rate and audio quality.
"""

from .types import AudioClip, CacheKey, GameEvent, Intensity, Material, SAMPLE_RATE
from .cache import VariantCache, CacheStats
from .predictor import Predictor, Prediction, Tier1DeterministicPredictor, NoisyPredictor
from .pipeline import PredictivePipeline, SimResult, EventOutcome
from .metrics import summarize, report

__version__ = "0.1.0"

__all__ = [
    "AudioClip", "CacheKey", "GameEvent", "Intensity", "Material", "SAMPLE_RATE",
    "VariantCache", "CacheStats",
    "Predictor", "Prediction", "Tier1DeterministicPredictor", "NoisyPredictor",
    "PredictivePipeline", "SimResult", "EventOutcome",
    "summarize", "report",
    "__version__",
]
