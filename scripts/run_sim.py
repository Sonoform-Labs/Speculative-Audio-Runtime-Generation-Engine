"""Run the predictive-SFX simulation. Thin wrapper over predictive_sfx.cli.

Examples:
    python scripts/run_sim.py
    python scripts/run_sim.py --sweep
    python scripts/run_sim.py --predictor noisy --recall 0.8 --accuracy 0.85 --prebake
"""
import _bootstrap  # noqa: F401  (adds src/ to path)

from predictive_sfx.cli import main_sim

if __name__ == "__main__":
    raise SystemExit(main_sim())
