import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predictive_sfx.multigame import compare


def main():
    print("=" * 60)
    print(" CROSS-GAME GENERALIZATION  (warm-phase cache hit-rate, %)")
    print(" learner = auto-derived affordances, ZERO authoring")
    print(" authored = hand-authored ITEM/SURFACE tables")
    print("=" * 60)
    r = compare(rounds=8)
    print(f"{'game':<12}{'learner':>10}{'authored':>10}")
    for g, d in r.items():
        print(f"{g:<12}{d['learned']['warm_hit_rate']*100:>9.0f}%{d['authored']['warm_hit_rate']*100:>9.0f}%")
    print("-" * 32)
    learner_avg = sum(d["learned"]["warm_hit_rate"] for d in r.values()) / len(r)
    authored_avg = sum(d["authored"]["warm_hit_rate"] for d in r.values()) / len(r)
    print(f"{'AVERAGE':<12}{learner_avg*100:>9.0f}%{authored_avg*100:>9.0f}%")
    print("\nLearner generalizes to any game from play (zero per-game authoring);")
    print("hand-authored tables only cover games matching their predefined vocabulary.")


if __name__ == "__main__":
    main()
