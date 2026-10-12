"""Einheitentests der Verfahren gegen die von Hand gerechneten Mini-Instanzen (conftest.py)."""

import numpy as np
import pytest

import rga_aggregate as A
import turnier_generator as T


def rng(seed=1):
    return T.SplitMix64(seed)


def test_prefer_matrix_of_the_cycle(cycle3):
    W = A.prefer_matrix(cycle3, 3)
    assert W.tolist() == [[0, 2, 1], [1, 0, 2], [2, 1, 0]]       # A vor B 2, B vor C 2, C vor A 2


def test_majority_cycle_and_winner(cycle3, clear_winner):
    assert A.majority_cycles(cycle3, 3) == 1 and A.condorcet_winner(cycle3, 3) is None
    assert A.majority_cycles(clear_winner, 4) == 0 and A.condorcet_winner(clear_winner, 4) == 0


def test_condorcet_winner_is_found_at_every_index_including_the_last():
    for w in range(5):
        rest = [x for x in range(5) if x != w]
        profile = [tuple([w] + rest), tuple([w] + rest[::-1]), tuple([w] + rest)]
        assert A.condorcet_winner(profile, 5) == w, w
        assert A.majority_cycles(profile, 5) == 0


def test_copeland_hand_instance_with_four_voters():
    """Von Hand: 0 schlägt 1, 2 (je 3:1) und 3 (4:0); 1 schlägt 2 (3:1) und 3 (4:0); 2 schlägt 3 (3:1): Copeland-Wertung (3, 2, 1, 0)."""
    profile = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (2, 0, 1, 3)]
    assert A.prefer_matrix(profile, 4).tolist() == [[0, 3, 3, 4], [1, 0, 3, 4], [1, 1, 0, 3], [0, 0, 1, 0]]
    assert A.copeland(profile, 4, rng(1)) == (0, 1, 2, 3)
    assert A.condorcet_winner(profile, 4) == 0 and A.majority_cycles(profile, 4) == 0


def test_kemeny_dp_cycle_has_three_optima_with_cost_four(cycle3):
    cost, order, count = A.kemeny_dp(cycle3, 3)
    assert cost == 4 and count == 3
    assert A.total_cost(order, cycle3) == 4
    assert A.total_cost((0, 2, 1), cycle3) == 5            # eine nicht optimale Ordnung, von Hand: 1 + 3 + 1


def test_kemeny_dp_clear_winner(clear_winner):
    cost, order, count = A.kemeny_dp(clear_winner, 4)
    assert (cost, order, count) == (2, (0, 1, 2, 3), 1)


def test_borda_ties_in_the_cycle_and_is_exact_on_the_clear_profile(cycle3, clear_winner):
    orders = {A.borda(cycle3, 3, rng(s)) for s in range(40)}
    assert len(orders) == 6                                  # alle Gleichstände: jede der 6 Ordnungen kommt vor
    assert A.borda(clear_winner, 4, rng(3)) == (0, 1, 2, 3)


def test_copeland_ties_in_the_cycle_and_follows_the_majority_otherwise(cycle3, clear_winner):
    assert len({A.copeland(cycle3, 3, rng(s)) for s in range(40)}) == 6
    assert A.copeland(clear_winner, 4, rng(3))[0] == 0


def test_footrule_cost_matrix_of_the_cycle(cycle3):
    C = A.footrule_cost_matrix(cycle3, 3)
    # von Hand: Plätze von A = (0, 2, 1), B = (1, 0, 2), C = (2, 1, 0) bei den drei Bewertern; Platz j = 0, 1, 2 kostet je Teilnehmer 3, 2, 3
    assert C.tolist() == [[3.0, 2.0, 3.0], [3.0, 2.0, 3.0], [3.0, 2.0, 3.0]]
    cost, count = A.footrule_optimal_brute(cycle3, 3)
    assert cost == 8.0 and count == 6


def test_footrule_optimal_matches_the_brute_force_cost(clear_winner):
    order = A.footrule_optimal(clear_winner, 4, rng(5))
    cost_matrix = A.footrule_cost_matrix(clear_winner, 4)
    achieved = sum(cost_matrix[order[j]][j] for j in range(4))
    assert achieved == A.footrule_optimal_brute(clear_winner, 4)[0]


def test_every_method_returns_a_permutation_and_one_voter_returns_its_input():
    single = [(3, 1, 0, 2)]
    res = A.run_all(single, 4, rng(9))
    assert res["kemeny_cost"] == 0 and res["kemeny_optima"] == 1
    for name, order in res["orders"].items():
        assert order == (3, 1, 0, 2), name


def test_total_cost_and_kendall_distance():
    assert A.kendall_distance((0, 1, 2), (2, 1, 0)) == 3
    assert A.kendall_distance((0, 1, 2), (0, 1, 2)) == 0
    assert A.total_cost((0, 1, 2), [(0, 1, 2), (2, 1, 0), (1, 0, 2)]) == 0 + 3 + 1


def test_order_by_score_descending_ascending_and_lot():
    assert A.order_by_score([1.0, 3.0, 2.0], rng(1)) == (1, 2, 0)
    assert A.order_by_score([1.0, 3.0, 2.0], rng(1), descending=False) == (0, 2, 1)
    tied = {A.order_by_score([2.0, 2.0, 1.0], rng(s)) for s in range(30)}
    assert tied == {(0, 1, 2), (1, 0, 2)}


def test_kemeny_random_relabel_removes_the_index_preference():
    """Bei drei gleich guten Optima soll nicht immer dasselbe zurückkommen (sonst entschiede der Index, der hier mit der Stärke korreliert)."""
    cycle = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    seen = {A.kemeny(cycle, 3, rng(s))[1] for s in range(60)}
    assert len(seen) == 3


def test_kemeny_dp_agrees_with_enumeration_on_a_noisy_profile():
    profile = [(0, 2, 1, 3), (1, 0, 2, 3), (0, 1, 3, 2), (2, 0, 1, 3), (0, 1, 2, 3)]
    cost, order, count = A.kemeny_dp(profile, 4)
    import itertools
    costs = [A.total_cost(p, profile) for p in itertools.permutations(range(4))]
    assert cost == min(costs) and count == costs.count(min(costs))
    assert A.total_cost(order, profile) == cost


def test_majority_cycles_count_triples_not_cyclic_chains():
    # 4 Teilnehmer: 0>1>2>0 ist ein Zyklus, 3 verliert gegen alle: genau eine zyklische Dreiergruppe
    W_profile = [(0, 1, 2, 3), (1, 2, 0, 3), (2, 0, 1, 3)]
    assert A.majority_cycles(W_profile, 4) == 1
    assert np.array_equal(A.prefer_matrix(W_profile, 4)[3], [0, 0, 0, 0])
