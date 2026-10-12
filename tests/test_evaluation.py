"""Analyse, Szenario und Verdikte; Randfälle der Regler."""

import itertools

import pytest

import rga_aggregate as A
import rga_constants as C
import rga_evaluation as ev
from rga_scenario import Settings, make_profile

CASES = [Settings(n=n, m=m, noise=noise, k=k, phi=0.7, seed=seed)
         for n, m, noise, k, seed in itertools.product((4, 6, 8), (1, 2, 5, 9), ("turnier", "mallows"), (1, 4), (3, 4))]


def test_analysis_invariants_over_many_settings():
    for s in CASES:
        a = ev.analyse(s)
        n = s.n
        for name, order in a.orders.items():
            assert sorted(order) == list(range(n)), (s, name)
            assert a.costs[name] >= a.kemeny_cost, (s, name)               # Kemeny hat die kleinsten Gesamtkosten
            assert a.costs[name] == A.total_cost(order, a.data.profile)
            assert a.dist_truth[name] == A.kendall_distance(order, a.data.truth)
        assert a.costs["Kemeny"] == a.kemeny_cost and a.kemeny_optima >= 1
        assert (a.pooled_dist is None) == (s.noise == "mallows")
        assert a.cycles >= 0 and (a.winner is None or 0 <= a.winner < n)


def test_one_voter_returns_the_input_for_every_method():
    for noise in ("turnier", "mallows"):
        a = ev.analyse(Settings(n=6, m=1, noise=noise, seed=5))
        only = a.data.profile[0]
        assert all(o == only for o in a.orders.values())
        assert a.kemeny_cost == 0 and a.kemeny_optima == 1


def test_condorcet_winner_excludes_cycles_at_the_top():
    for s in CASES:
        a = ev.analyse(s)
        if a.winner is not None:
            top = a.winner
            assert all(a.prefer[top][j] > a.prefer[j][top] for j in range(s.n) if j != top)


def test_unknown_noise_raises():
    with pytest.raises(ValueError):
        make_profile(Settings(noise="chaos"))


def test_analysis_is_cached_and_deterministic():
    s = Settings(n=6, m=5, seed=11)
    assert ev.analyse(s) is ev.analyse(s)
    assert make_profile(s) == make_profile(Settings(n=6, m=5, seed=11))
    assert make_profile(s).profile != make_profile(Settings(n=6, m=5, seed=12)).profile


def test_names_are_a_permutation_of_letters_independent_of_strength():
    d = make_profile(Settings(n=8, seed=3))
    assert len(set(d.names)) == 8
    assert d.names != tuple(sorted(d.names)) or make_profile(Settings(n=8, seed=4)).names != tuple(sorted(d.names))


def test_verdict_single_voter_and_cycle_cases():
    level, text = ev.verdict_analysis(ev.analyse(Settings(n=6, m=1, seed=5)))
    assert level == "info" and "Ein einzelner Bewerter" in text
    cyc = ev.analyse(Settings(**C.PRESETS["Kein Mehrheitssieger"]))
    level, text = ev.verdict_analysis(cyc)
    assert level == "warning" and "keinen Mehrheitssieger" in text


def test_verdict_names_the_cheapest_and_the_closest_method():
    a = ev.analyse(Settings(**C.PRESETS["Viele ungenaue Bewerter"]))
    level, text = ev.verdict_analysis(a)
    assert "Kleinste Gesamtkosten" in text and "am nächsten" in text
    assert a.kemeny_cost == min(a.costs.values())


def test_sweeps_have_consistent_shapes():
    rows = ev.sweep_methods(ms=(1, 3), seeds=C.SWEEP_SEEDS[:20])
    assert [r["m"] for r in rows] == [1, 3]
    for r in rows:
        assert all(r[name] >= 0 for name in C.METHODS) and r["pooled"] is not None
    assert rows[0]["Borda"] == rows[0]["Kemeny"] == rows[0]["Copeland"] == rows[0]["Footrule-optimal"] == pytest.approx(rows[0]["single"])
    pairs = ev.sweep_pairs(ms=(3,), seeds=C.SWEEP_SEEDS[:20])
    assert {r["noise"] for r in pairs} == {"turnier", "mallows"}
    for r in pairs:
        assert r["better"] + r["equal"] + r["worse"] == pytest.approx(1.0)
    q = ev.sweep_quality(ms=(3,), seeds=C.SWEEP_SEEDS[:20])
    assert 1.0 <= q[0]["ratio_mean"] <= q[0]["ratio_max"] <= 2.0


def test_decimal_comma_helpers():
    assert ev.de(0.5) == "0,50" and ev.de(3.14159, 3) == "3,142" and ev.pct1(0.0526) == "5,3"


def test_time_kemeny_returns_one_row_per_size_and_growing_state_counts():
    rows = ev.time_kemeny(ns=(6, 12), m=3)
    assert [r["n"] for r in rows] == [6, 12] and rows[0]["states"] == 384 and rows[1]["states"] == 49152     # 2^n · n
    assert all(r["seconds"] >= 0 for r in rows)
