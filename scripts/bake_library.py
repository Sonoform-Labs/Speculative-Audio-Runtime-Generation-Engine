"""Bake an offline library of SFX variants. Thin wrapper over predictive_sfx.cli.

Examples:
    python scripts/bake_library.py                       # mock, all materials
    python scripts/bake_library.py --generator elevenlabs --variants 10
"""
import _bootstrap  # noqa: F401  (adds src/ to path)

from predictive_sfx.cli import main_bake

if __name__ == "__main__":
    raise SystemExit(main_bake())
