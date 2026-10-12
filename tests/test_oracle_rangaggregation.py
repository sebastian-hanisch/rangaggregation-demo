"""Orakel-Tests rangaggregation-demo: Verfahren und angezeigte Kennzahlen gegen einen ANDEREN Rechenweg (DEMO-PLAYBOOK §12).

1. Kemeny-Kosten und Zahl der Optima gegen die Aufzählung aller n! Ordnungen.
2. Footrule-optimal: Kosten gegen die Aufzählung aller Zuordnungen; Kemeny-Kostenverhältnis höchstens 2 (aus K <= F <= 2K).
3. Borda gegen eine Neuberechnung aus den Plätzen; Copeland gegen die Aufzählung der paarweisen Mehrheiten.
4. Mehrheitszyklen gegen die Aufzählung aller Dreiergruppen; exakte Wahrscheinlichkeit 12/216 für 3 Teilnehmer und 3 Bewerter.
5. Symmetrie: Umbenennung der Teilnehmer benennt das Ergebnis um (bei eindeutiger Lage).
"""

import itertools
import random

import pytest

import rga_aggregate as A
import turnier_generator as T


def random_profile(n, m, rnd):
    out = []
    for _ in range(m):
        o = list(range(n))
        rnd.shuffle(o)
        out.append(tuple(o))
    return out


def brute_kendall(a, b):
    pa = {x: j for j, x in enumerate(a)}
    pb = {x: j for j, x in enumerate(b)}
    return sum(1 for x, y in itertools.combinations(a, 2) if (pa[x] < pa[y]) != (pb[x] < pb[y]))


# --- 1. Kemeny -----------------------------------------------------------------------------------------------------------------------

def test_kemeny_cost_and_optima_count_match_enumeration():
    rnd = random.Random(2026)
    for _ in range(150):
        n = rnd.randint(2, 6)
        profile = random_profile(n, rnd.randint(1, 6), rnd)
        costs = [sum(brute_kendall(p, o) for o in profile) for p in itertools.permutations(range(n))]
        cost, order, count = A.kemeny_dp(profile, n)
        assert cost == min(costs)
        assert count == costs.count(min(costs))
        assert sum(brute_kendall(order, o) for o in profile) == cost
        assert sorted(order) == list(range(n))


def test_kemeny_with_relabel_has_the_same_cost_and_count():
    rnd = random.Random(5)
    for s in range(60):
        n = rnd.randint(3, 7)
        profile = random_profile(n, rnd.randint(2, 7), rnd)
        c0, _, k0 = A.kemeny_dp(profile, n)
        c1, order, k1 = A.kemeny(profile, n, T.SplitMix64(s))
        assert (c1, k1) == (c0, k0) and sum(brute_kendall(order, o) for o in profile) == c0


# --- 2. Footrule -----------------------------------------------------------------------------------------------------------------------

def test_footrule_optimal_cost_matches_enumeration_and_scipy_assignment():
    pytest.importorskip("scipy.optimize")
    rnd = random.Random(7)
    for s in range(80):
        n = rnd.randint(2, 6)
        profile = random_profile(n, rnd.randint(1, 6), rnd)
        order = A.footrule_optimal(profile, n, T.SplitMix64(s))
        pos = {x: j for j, x in enumerate(order)}
        achieved = sum(abs(p.index(x) - pos[x]) for p in profile for x in range(n))
        best, _ = A.footrule_optimal_brute(profile, n)
        assert achieved == pytest.approx(best)


def test_footrule_optimal_kemeny_cost_is_at_most_twice_the_optimum():
    rnd = random.Random(11)
    for s in range(150):
        n = rnd.randint(2, 7)
        profile = random_profile(n, rnd.randint(1, 7), rnd)
        cost, _, _ = A.kemeny_dp(profile, n)
        order = A.footrule_optimal(profile, n, T.SplitMix64(s))
        assert A.total_cost(order, profile) <= 2 * cost


# --- 3. Borda und Copeland ----------------------------------------------------------------------------------------------------------

def test_borda_orders_by_the_sum_of_places_with_ties_only_between_equal_sums():
    rnd = random.Random(13)
    for s in range(100):
        n = rnd.randint(2, 8)
        profile = random_profile(n, rnd.randint(1, 8), rnd)
        sums = [sum(o.index(x) for o in profile) for x in range(n)]
        order = A.borda(profile, n, T.SplitMix64(s))
        assert all(sums[order[j]] <= sums[order[j + 1]] for j in range(n - 1))


def test_copeland_orders_by_pairwise_majority_wins():
    rnd = random.Random(17)
    for s in range(100):
        n = rnd.randint(2, 7)
        m = rnd.randint(1, 8)
        profile = random_profile(n, m, rnd)

        def wins(i):
            total = 0.0
            for j in range(n):
                if j == i:
                    continue
                a = sum(1 for o in profile if o.index(i) < o.index(j))
                b = m - a
                total += 1.0 if a > b else (0.5 if a == b else 0.0)
            return total

        order = A.copeland(profile, n, T.SplitMix64(s))
        assert all(wins(order[j]) >= wins(order[j + 1]) for j in range(n - 1))


# --- 4. Mehrheitszyklen --------------------------------------------------------------------------------------------------------------

def brute_cycles(profile, n):
    def maj(i, j):
        a = sum(1 for o in profile if o.index(i) < o.index(j))
        return a > len(profile) - a
    return sum(1 for x, y, z in itertools.combinations(range(n), 3)
               if (maj(x, y) and maj(y, z) and maj(z, x)) or (maj(x, z) and maj(z, y) and maj(y, x)))


def test_majority_cycles_match_enumeration_and_the_exact_three_voter_probability():
    rnd = random.Random(19)
    for _ in range(150):
        n = rnd.randint(3, 7)
        profile = random_profile(n, rnd.randint(1, 8), rnd)
        assert A.majority_cycles(profile, n) == brute_cycles(profile, n)
    perms = list(itertools.permutations(range(3)))
    cycles = sum(1 for p in itertools.product(perms, repeat=3) if brute_cycles(list(p), 3) > 0)
    assert (cycles, len(perms) ** 3) == (12, 216)


# --- 5. Symmetrie --------------------------------------------------------------------------------------------------------------------

def test_relabelling_the_participants_relabels_borda_and_copeland_when_the_result_is_unique():
    rnd = random.Random(23)
    checked = 0
    for s in range(120):
        n = rnd.randint(3, 6)
        profile = random_profile(n, 7, rnd)
        perm = list(range(n))
        rnd.shuffle(perm)
        sums = [sum(o.index(x) for o in profile) for x in range(n)]
        if len(set(sums)) < n:
            continue
        mapped = [tuple(perm[x] for x in o) for o in profile]
        expected = tuple(perm[x] for x in A.borda(profile, n, T.SplitMix64(1)))
        assert A.borda(mapped, n, T.SplitMix64(1)) == expected
        checked += 1
    assert checked > 10
