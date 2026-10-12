"""Einheitentests der Rauschmodelle und Profile."""

import itertools
import math

import numpy as np
import pytest

import rga_aggregate as A
import rga_noise as N
import turnier_generator as T


def test_truth_order_puts_the_strongest_first():
    assert N.truth_order(4) == (3, 2, 1, 0)


def test_turnier_profile_is_deterministic_permutations_with_independent_voters():
    p = N.turnier_profile(6, 5, 1, 0.3, 77)
    assert p == N.turnier_profile(6, 5, 1, 0.3, 77)
    assert len(p) == 5 and all(sorted(o) == list(range(6)) for o in p)
    assert len(set(p)) > 1                                  # die Bewerter sehen verschiedene Turniere
    assert p != N.turnier_profile(6, 5, 1, 0.3, 78)


def test_many_games_per_voter_bring_the_tables_close_to_the_truth():
    truth = N.truth_order(6)
    many = np.mean([A.kendall_distance(o, truth) for o in N.turnier_profile(6, 10, 40, 0.3, 5)])
    one = np.mean([A.kendall_distance(o, truth) for o in N.turnier_profile(6, 10, 1, 0.3, 5)])
    assert many < one and many < 2.0


def test_pooled_table_uses_the_sum_of_the_wins_of_the_same_tournaments():
    n, m, k, delta, seed = 6, 4, 1, 0.3, 11
    wins = N.turnier_wins(n, m, k, delta, seed)
    total = np.sum(wins, axis=0)
    pooled = N.pooled_table(n, m, k, delta, seed)
    assert sorted(pooled) == list(range(n))
    assert all(total[pooled[j]] >= total[pooled[j + 1]] for j in range(n - 1))      # absteigend nach Siegen
    assert total.sum() == m * n * (n - 1) / 2                                         # jede Partie hat genau einen Sieger


def expected_mallows_distance(n, phi):
    """Geschlossene Formel für den erwarteten Kendall-Abstand zur Referenz: n·φ/(1-φ) - Σ_j j·φ^j/(1-φ^j) (für φ < 1)."""
    return n * phi / (1 - phi) - sum(j * phi ** j / (1 - phi ** j) for j in range(1, n + 1))


def test_mallows_mean_distance_matches_the_closed_form_and_phi_one_is_uniform():
    rng = T.SplitMix64(3)
    for phi in (0.2, 0.5, 0.8):
        draws = [A.kendall_distance(N.mallows_sample(6, phi, rng), tuple(range(6))) for _ in range(1500)]
        assert np.mean(draws) == pytest.approx(expected_mallows_distance(6, phi), abs=0.2), phi
    far = [A.kendall_distance(N.mallows_sample(6, 1.0, rng), tuple(range(6))) for _ in range(600)]
    assert np.mean(far) == pytest.approx(7.5, abs=0.5)                                # gleichverteilt: Mittel n(n-1)/4 = 7,5


def test_mallows_sampler_matches_the_exact_distribution_for_n4():
    """Häufigkeit jeder der 24 Ordnungen gegen phi^K / Z (Chi-Quadrat nur grob: größte Abweichung unter 0,01)."""
    phi = 0.6
    ref = tuple(range(4))
    perms = list(itertools.permutations(range(4)))
    w = np.array([phi ** A.kendall_distance(p, ref) for p in perms])
    exact = w / w.sum()
    rng = T.SplitMix64(9)
    draws = 20000
    counts = {p: 0 for p in perms}
    for _ in range(draws):
        counts[N.mallows_sample(4, phi, rng)] += 1
    freq = np.array([counts[p] / draws for p in perms])
    assert np.abs(freq - exact).max() < 0.01


def test_mallows_profile_is_centered_on_the_truth_and_validates_phi():
    prof = N.mallows_profile(6, 50, 0.3, 4)
    assert all(sorted(o) == list(range(6)) for o in prof)
    first_places = [o[0] for o in prof]
    assert first_places.count(5) > 25                         # die Wahrheit (Index 5 zuerst) ist die häufigste Spitze
    with pytest.raises(ValueError):
        N.mallows_sample(5, 0.0, T.SplitMix64(1))
    with pytest.raises(ValueError):
        N.mallows_sample(5, 1.5, T.SplitMix64(1))


def test_random_profile_cycle_shortcut_equals_the_general_computation():
    r1, r2 = T.SplitMix64(5), T.SplitMix64(5)
    for _ in range(200):
        fast = N.random_profile_has_cycle3(7, r1)
        slow = A.majority_cycles(N.random_profile(3, 7, r2), 3) > 0
        assert fast == slow


def test_exact_cycle_probability_for_three_voters_is_one_in_eighteen():
    perms = list(itertools.permutations(range(3)))
    cyc = sum(A.majority_cycles(list(p), 3) > 0 for p in itertools.product(perms, repeat=3))
    assert (cyc, len(perms) ** 3) == (12, 216)
    assert math.isclose(cyc / 216, 1 / 18)
