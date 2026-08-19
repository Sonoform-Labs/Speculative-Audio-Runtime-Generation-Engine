import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sage.benchmark import compare_predictors, footstep_stream, gait_scenario, surface_seam

_COLS = ["hit_rate", "fallback_rate", "cold_hit_rate", "warm_hit_rate",
         "ready_in_time_among_predicted", "speculation_precision", "immediate_repeat_rate"]


def _row(name, s):
    cells = []
    for c in _COLS:
        v = s.get(c, 0.0)
        cells.append(f"{v*100:6.1f}")
    return f"{name:<10}" + "".join(f"{x:>9}" for x in cells)


def _table(title, results):
    print("\n" + title)
    print("-" * (10 + 9 * len(_COLS)))
    head = "predictor " + "".join(f"{c.split('_')[0][:8]:>9}" for c in _COLS)
    print(head)
    for name, s in results.items():
        print(_row(name, s))


def main():
    print("=" * 72)
    print(" PREDICTIVE NEURAL SFX - BENCHMARK  (values in %, gen latency 120ms)")
    print("=" * 72)

    walk = footstep_stream(n=80, surfaces=("grass",))
    _table("A. Steady footstep stream (single surface)", compare_predictors(walk))

    mixed = footstep_stream(n=96, surfaces=("grass", "metal", "wood"))
    _table("B. Footstep stream across 3 surfaces (recurring keys)", compare_predictors(mixed))

    uniq = footstep_stream(n=96, surfaces=("grass", "metal", "wood"), unique=True)
    _table("B2. Diverse/cold keys (every step a new context)", compare_predictors(uniq))

    seam = gait_scenario(speed=1.4, dur=16.0, surface_fn=surface_seam)
    _table("C. Gait predictor, straight walk crossing a grass/metal seam",
           {"predicted": seam["predicted"], "none": seam["none"]})

    print("\nD. Lead must exceed gen latency (gait predictor, SA3 latency = 393 ms)")
    print("-" * 48)
    print(f"{'lead_ms':>9}{'hit_rate':>12}{'ready_in_time':>16}")
    for lead in (250, 400, 600, 900):
        s = gait_scenario(speed=1.4, dur=14.0, lead_ms=lead, gen_latency_ms=393.0)["predicted"]
        print(f"{lead:>9}{s['hit_rate']*100:>11.0f}%{s['ready_in_time_among_predicted']*100:>15.0f}%")

    print("\nlegend: hit=served-from-cache  cold=first-occurrence  warm=steady-state")
    print("        ready=gen finished before the event  specul=fraction of gens actually used")
    print("        D shows hit-rate stays propped up by cache warming, but genuine")
    print("        anticipation (ready_in_time) needs lead > gen latency.")


if __name__ == "__main__":
    main()
