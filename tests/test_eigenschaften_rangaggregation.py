"""Eigenschaftstests (metamorph) der Rangaggregation (ORACLE-PLAYBOOK §5.2).

Beziehungen zwischen Eingabe und Ergebnis, an je 25 bis 40 Zufallsprofilen (SplitMix64, fester Seed; Rauschen aus `rga_noise`, damit
auch schiefe Profile mit Mehrheitssieger vorkommen) plus Randfällen. Ergänzt test_oracle_rangaggregation.py (Aufzählung, Neuberechnung):

1. Einstimmiges Profil: jedes Verfahren gibt genau diese Rangfolge (Kemeny: Kosten 0, ein Optimum).
2. Umnummerieren der Teilnehmer ist äquivariant (Kosten, Zahl der Optima und, wo das Ergebnis eindeutig ist, die Rangfolge selbst).
3. Alle Eingaben umkehren: Borda-Platzsummen m(n-1) - s, Copeland-Wertung (n-1) - w, Kemeny-/Footrule-Kosten gleich, Ergebnis umgekehrt
   (wo eindeutig).
4. Kemeny-Kosten sind kleiner oder gleich den Kosten jeder anderen Rangfolge (Borda, Copeland, Footrule-optimal, jede Eingabe) und
   mindestens die Summe der Minderheiten je Paar; bei transitiver Mehrheit gleich dieser Summe.
5. Condorcet-Sieger steht bei Copeland und Kemeny vorn, Condorcet-Verlierer hinten (Borda hat diese Eigenschaft nicht und wird
   deshalb nicht geprüft).
6. Profil verdoppeln oder die Bewerter umsortieren ändert kein Verfahren.
7. Eine weitere Eingabe gleich dem Kemeny-Ergebnis lässt dessen Kosten unverändert und das Ergebnis optimal.
8. Footrule-Zuordnung: minimale Footrule-Kosten unter allen Kandidaten, und Kendall-Kosten höchstens das Doppelte des Kemeny-Optimums.
9. Randfälle: ein Teilnehmer, zwei Teilnehmer, ein Bewerter, zwei gegenläufige Bewerter (alles Gleichstand), phi -> 0.
"""

import itertools
import math

import numpy as np
import pytest

import rga_aggregate as A
import rga_noise as N
import turnier_generator as T

N_INST = 30


# --- Zufallsprofile ----------------------------------------------------------------------------------------------------------------------

def profile_of(i, nmin=3, nmax=7, odd_m=False):
    """(profile, n, m): wechselnde Größe, mal gleichverteilt zufällig (impartial culture), mal Mallows mit phi 0,4 bis 0,8 um die Wahrheit."""
    n = nmin + (i * 3) % (nmax - nmin + 1)
    m = 2 + (i * 5) % 6
    if odd_m and m % 2 == 0:
        m += 1
    if i % 3 == 0:
        return N.random_profile(n, m, T.SplitMix64(1000 + i)), n, m
    phi = (0.4, 0.6, 0.8)[i % 3]
    return N.mallows_profile(n, m, phi, 2000 + i), n, m


def rng(seed=5):
    return T.SplitMix64(seed)


def borda_points(profile, n):
    return sum(A.positions(o) for o in profile)


def copeland_scores(profile, n):
    W = A.prefer_matrix(profile, n)
    return (W > W.T).sum(axis=1) + 0.5 * ((W == W.T).sum(axis=1) - 1)


def relabel(profile, mapping):
    return [tuple(int(mapping[x]) for x in o) for o in profile]


def apply(order, mapping):
    return tuple(int(mapping[x]) for x in order)


def footrule_cost(order, profile):
    """Summe der Footrule-Abstände von `order` zu allen Eingaben (eigener Rechenweg über die Plätze)."""
    pos = A.positions(order)
    return int(sum(np.abs(pos - A.positions(o)).sum() for o in profile))


def unique_orders(profile, n):
    """Welche Verfahren haben ein eindeutiges Ergebnis? (Borda-Summen, Copeland-Wertungen paarweise verschieden.)"""
    b = borda_points(profile, n)
    c = copeland_scores(profile, n)
    return len(set(b.tolist())) == n, len(set(c.tolist())) == n


# --- 1. Einstimmig -----------------------------------------------------------------------------------------------------------------------

def test_unanimous_profile_returns_exactly_that_ranking_for_every_method():
    for i in range(N_INST):
        n = 1 + i % 8
        m = 1 + i % 5
        pi = tuple(int(x) for x in T.SplitMix64(300 + i).permutation(n))
        profile = [pi] * m
        assert A.borda(profile, n, rng(i)) == pi
        assert A.copeland(profile, n, rng(i)) == pi
        cost, order, count = A.kemeny(profile, n, rng(i))
        assert (cost, order, count) == (0, pi, 1)
        assert A.footrule_optimal(profile, n, rng(i)) == pi
        assert A.majority_cycles(profile, n) == 0
        assert A.condorcet_winner(profile, n) == (pi[0] if n > 1 else 0)


# --- 2. Umnummerieren ----------------------------------------------------------------------------------------------------------------

def test_relabeling_candidates_is_equivariant_for_every_method():
    unique_checked = [0, 0, 0, 0]
    for i in range(N_INST):
        profile, n, m = profile_of(i, nmax=6)
        sigma = tuple(int(x) for x in T.SplitMix64(7000 + i).permutation(n))
        prof2 = relabel(profile, sigma)
        # Mehrheitsmatrix, Zyklen, Condorcet-Sieger
        W, W2 = A.prefer_matrix(profile, n), A.prefer_matrix(prof2, n)
        s = np.array(sigma)
        assert (W2[np.ix_(s, s)] == W).all()
        assert A.majority_cycles(prof2, n) == A.majority_cycles(profile, n)
        cw, cw2 = A.condorcet_winner(profile, n), A.condorcet_winner(prof2, n)
        assert (cw is None and cw2 is None) or sigma[cw] == cw2
        # Kemeny: Kosten und Zahl der Optima gleich, bei einem Optimum die Rangfolge selbst
        c1, o1, k1 = A.kemeny(profile, n, rng(i))
        c2, o2, k2 = A.kemeny(prof2, n, rng(i))
        assert c1 == c2 and k1 == k2
        assert A.total_cost(o2, prof2) == c2
        if k1 == 1:
            assert o2 == apply(o1, sigma)
            unique_checked[0] += 1
        # Borda, Copeland
        ub, uc = unique_orders(profile, n)
        if ub:
            assert A.borda(prof2, n, rng(i)) == apply(A.borda(profile, n, rng(i)), sigma)
            unique_checked[1] += 1
        if uc:
            assert A.copeland(prof2, n, rng(i)) == apply(A.copeland(profile, n, rng(i)), sigma)
            unique_checked[2] += 1
        # Footrule-Zuordnung: Kosten gleich, Ergebnis gleich, wenn das Optimum eindeutig ist
        f1, f2 = A.footrule_optimal(profile, n, rng(i)), A.footrule_optimal(prof2, n, rng(i))
        assert footrule_cost(f1, profile) == footrule_cost(f2, prof2)
        if A.footrule_optimal_brute(profile, n)[1] == 1:
            assert f2 == apply(f1, sigma)
            unique_checked[3] += 1
    assert min(unique_checked) >= 3, unique_checked


# --- 3. Alle Eingaben umkehren --------------------------------------------------------------------------------------------------------

def test_reversing_every_input_reverses_the_result():
    checked = [0, 0, 0, 0]
    for i in range(N_INST):
        profile, n, m = profile_of(i, nmax=6)
        rev = [tuple(reversed(o)) for o in profile]
        # Borda-Platzsummen und Copeland-Wertung spiegeln sich
        assert (borda_points(rev, n) == m * (n - 1) - borda_points(profile, n)).all()
        assert copeland_scores(rev, n) == pytest.approx((n - 1) - copeland_scores(profile, n))
        ub, uc = unique_orders(profile, n)
        if ub:
            assert A.borda(rev, n, rng(i)) == tuple(reversed(A.borda(profile, n, rng(i))))
            checked[0] += 1
        if uc:
            assert A.copeland(rev, n, rng(i)) == tuple(reversed(A.copeland(profile, n, rng(i))))
            checked[1] += 1
        c1, o1, k1 = A.kemeny(profile, n, rng(i))
        c2, o2, k2 = A.kemeny(rev, n, rng(i))
        assert c1 == c2 and k1 == k2
        if k1 == 1:
            assert o2 == tuple(reversed(o1))
            checked[2] += 1
        f1, f2 = A.footrule_optimal(profile, n, rng(i)), A.footrule_optimal(rev, n, rng(i))
        assert footrule_cost(f1, profile) == footrule_cost(f2, rev)
        if A.footrule_optimal_brute(profile, n)[1] == 1:
            assert f2 == tuple(reversed(f1))
            checked[3] += 1
    assert min(checked) >= 3, checked


# --- 4. Kemeny ist am billigsten ------------------------------------------------------------------------------------------------------

def test_kemeny_cost_is_a_lower_bound_for_every_other_ranking_and_at_least_the_sum_of_minorities():
    transitive = 0
    for i in range(N_INST):
        profile, n, m = profile_of(i, nmax=7)
        cost, order, count = A.kemeny(profile, n, rng(i))
        assert A.total_cost(order, profile) == cost and count >= 1
        others = [A.borda(profile, n, rng(i)), A.copeland(profile, n, rng(i)), A.footrule_optimal(profile, n, rng(i)), *profile]
        for o in others:
            assert A.total_cost(o, profile) >= cost
        W = A.prefer_matrix(profile, n)
        minorities = int(sum(min(W[a, b], W[b, a]) for a in range(n) for b in range(a + 1, n)))
        assert cost >= minorities
        if A.majority_cycles(profile, n) == 0 and all(W[a, b] != W[b, a] for a in range(n) for b in range(a + 1, n)):
            transitive += 1
            assert cost == minorities                                 # transitive Mehrheit: jedes Paar zahlt genau seine Minderheit
            assert count == 1
            assert order == A.copeland(profile, n, rng(i))            # und Kemeny = Copeland = Mehrheitsordnung
    assert transitive >= 3, "der Zweig mit transitiver Mehrheit wurde kaum erreicht"


# --- 5. Condorcet ----------------------------------------------------------------------------------------------------------------------

def test_condorcet_winner_leads_and_condorcet_loser_trails_in_copeland_and_kemeny():
    winners = losers = 0
    for i in range(40):
        profile, n, m = profile_of(i, nmax=7, odd_m=True)
        W = A.prefer_matrix(profile, n)
        cw = A.condorcet_winner(profile, n)
        cl = next((x for x in range(n) if all(W[j, x] > W[x, j] for j in range(n) if j != x)), None)
        cop = A.copeland(profile, n, rng(i))
        _, kem, _ = A.kemeny(profile, n, rng(i))
        if cw is not None:
            winners += 1
            assert cop[0] == cw and kem[0] == cw
        if cl is not None:
            losers += 1
            assert cop[-1] == cl and kem[-1] == cl
    assert winners >= 12 and losers >= 12


def test_condorcet_winner_leads_every_copeland_draw_whatever_the_lot():
    profile = [(2, 0, 1, 3), (2, 1, 3, 0), (0, 2, 3, 1)]       # 2 schlägt jeden mit 2:1, die anderen bilden eine Mischung
    assert A.condorcet_winner(profile, 4) == 2
    for seed in range(20):
        assert A.copeland(profile, 4, rng(seed))[0] == 2
        assert A.kemeny(profile, 4, rng(seed))[1][0] == 2


# --- 6. Verdoppeln, Umsortieren ---------------------------------------------------------------------------------------------------------

def test_duplicating_the_profile_or_reordering_the_voters_changes_no_method():
    for i in range(N_INST):
        profile, n, m = profile_of(i, nmax=6)
        doubled = profile + profile
        shuffled = [profile[j] for j in T.SplitMix64(60 + i).permutation(m)]
        base = A.run_all(profile, n, rng(i))
        for other, scale in ((doubled, 2), (shuffled, 1)):
            res = A.run_all(other, n, rng(i))
            assert res["orders"]["Borda"] == base["orders"]["Borda"]
            assert res["orders"]["Copeland"] == base["orders"]["Copeland"]
            assert res["orders"]["Kemeny"] == base["orders"]["Kemeny"]
            assert res["orders"]["Footrule-optimal"] == base["orders"]["Footrule-optimal"]
            assert res["kemeny_cost"] == scale * base["kemeny_cost"]
            assert res["kemeny_optima"] == base["kemeny_optima"]
        assert A.majority_cycles(doubled, n) == A.majority_cycles(profile, n)


# --- 7. Eine Stimme für das Ergebnis ----------------------------------------------------------------------------------------------------

def test_adding_a_voter_who_equals_the_kemeny_result_keeps_it_optimal_at_the_same_cost():
    for i in range(N_INST):
        profile, n, m = profile_of(i, nmax=7)
        cost, order, count = A.kemeny(profile, n, rng(i))
        extended = profile + [order]
        cost2, order2, count2 = A.kemeny(extended, n, rng(i))
        assert cost2 == cost                                   # neue Eingabe kostet 0, kein Optimum wird billiger
        assert A.total_cost(order, extended) == cost2          # das alte Ergebnis ist weiterhin optimal
        assert 1 <= count2 <= count                            # Optima des neuen Profils sind Optima des alten
        if count == 1:
            assert order2 == order


# --- 8. Footrule-Zuordnung ----------------------------------------------------------------------------------------------------------

def test_footrule_assignment_is_footrule_optimal_and_within_twice_the_kemeny_cost():
    for i in range(N_INST):
        profile, n, m = profile_of(i, nmax=7)
        fo = A.footrule_optimal(profile, n, rng(i))
        fcost = footrule_cost(fo, profile)
        candidates = [A.borda(profile, n, rng(i)), A.copeland(profile, n, rng(i)), A.kemeny(profile, n, rng(i))[1], *profile]
        assert all(footrule_cost(c, profile) >= fcost for c in candidates)
        kcost = A.kemeny(profile, n, rng(i))[0]
        assert kcost <= A.total_cost(fo, profile) <= 2 * kcost       # K <= F <= 2K, angewandt auf die Summen


# --- 9. Randfälle -----------------------------------------------------------------------------------------------------------------------

def test_single_candidate_and_two_candidates():
    assert A.borda([(0,)], 1, rng()) == (0,)
    assert A.kemeny([(0,), (0,)], 1, rng()) == (0, (0,), 1)
    assert A.copeland([(0,)], 1, rng()) == (0,)
    assert A.footrule_optimal([(0,)], 1, rng()) == (0,)
    profile = [(0, 1), (0, 1), (1, 0)]
    for fn in (A.borda, A.copeland, A.footrule_optimal):
        assert fn(profile, 2, rng()) == (0, 1)
    assert A.kemeny(profile, 2, rng()) == (1, (0, 1), 1)
    assert A.condorcet_winner(profile, 2) == 0


def test_one_voter_returns_its_input_and_two_opposite_voters_tie_everything():
    for i in range(10):
        n = 2 + i % 6
        pi = tuple(int(x) for x in T.SplitMix64(40 + i).permutation(n))
        assert A.kemeny([pi], n, rng(i))[1:] == (pi, 1)
        assert A.borda([pi], n, rng(i)) == pi and A.copeland([pi], n, rng(i)) == pi
        both = [pi, tuple(reversed(pi))]
        cost, _, count = A.kemeny(both, n, rng(i))
        assert cost == n * (n - 1) // 2 and count == math.factorial(n)          # jede Rangfolge kostet jedes Paar einmal
        assert len(set(borda_points(both, n).tolist())) == 1                    # alle Platzsummen gleich
        assert A.condorcet_winner(both, n) is None
        assert A.majority_cycles(both, n) == 0                                  # Gleichstand ist keine Mehrheitskante
        # verschiedene Lose ergeben verschiedene, aber gültige Rangfolgen
        orders = {A.borda(both, n, rng(s)) for s in range(12)}
        assert all(sorted(o) == list(range(n)) for o in orders)
        if n >= 3:
            assert len(orders) > 1


def test_mallows_with_tiny_phi_returns_the_reference_and_every_draw_is_a_permutation():
    for n in range(1, 9):
        assert N.mallows_sample(n, 1e-12, rng(n)) == tuple(range(n))
    for i in range(N_INST):
        n = 1 + i % 9
        phi = (0.1, 0.5, 0.9, 1.0)[i % 4]
        assert sorted(N.mallows_sample(n, phi, rng(i))) == list(range(n))
    for phi in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            N.mallows_sample(3, phi, rng())
