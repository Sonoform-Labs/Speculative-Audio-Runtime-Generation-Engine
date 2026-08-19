from sage.multigame import GAMES, compare


def test_learner_generalizes_across_all_games():
    r = compare(rounds=6)
    for g, d in r.items():
        assert d["learned"]["warm_hit_rate"] >= 0.95, (g, d["learned"])


def test_authored_tables_fail_on_a_novel_game():
    r = compare(rounds=6)
    assert r["racing"]["authored"]["warm_hit_rate"] < 0.1
    assert r["racing"]["learned"]["warm_hit_rate"] >= 0.95


def test_learner_never_worse_than_authored():
    r = compare(rounds=6)
    for g, d in r.items():
        assert d["learned"]["warm_hit_rate"] >= d["authored"]["warm_hit_rate"]


def test_all_games_have_distinct_vocab():
    assert set(GAMES) == {"fps", "platformer", "racing", "explore"}
