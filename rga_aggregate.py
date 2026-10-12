"""Aggregationsverfahren (numpy, scipy, Standardbibliothek). Ein Profil ist eine Liste von m Rangfolgen (Tupel der Teilnehmerindizes, bester zuerst)
über n Teilnehmer. Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b); Zufall nur über einen SplitMix64.

Gleichstände entscheiden Borda und Copeland durch das Los; Kemeny und die Footrule-Zuordnung laufen nach einer zufälligen Umbenennung der
Teilnehmer, sonst entschiede der Index (hier korreliert er mit der Stärke) und verzerrte den Vergleich."""

import itertools
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

import turnier_generator as T


def positions(order) -> np.ndarray:
    order = np.asarray(order, dtype=int)
    pos = np.empty(len(order), dtype=int)
    pos[order] = np.arange(len(order))
    return pos


def kendall_distance(a, b) -> int:
    pa, pb = positions(a), positions(b)
    iu = np.triu_indices(len(pa), 1)
    return int((np.sign(pa[iu[0]] - pa[iu[1]]) != np.sign(pb[iu[0]] - pb[iu[1]])).sum())


def prefer_matrix(profile, n: int) -> np.ndarray:
    """W[i, j] = Zahl der Bewerter, die Teilnehmer i vor j setzen."""
    W = np.zeros((n, n), dtype=int)
    for order in profile:
        pos = positions(order)
        W += (pos[:, None] < pos[None, :]).astype(int)
    return W


def total_cost(order, profile) -> int:
    """Summe der Kendall-Distanzen von `order` zu allen Eingaben: die Zielfunktion von Kemeny."""
    return sum(kendall_distance(order, o) for o in profile)


def order_by_score(score, rng: T.SplitMix64, descending: bool = True) -> tuple:
    """Rangfolge nach Wertung; Gleichstand entscheidet das Los."""
    score = np.asarray(score, dtype=float)
    lots = rng.uniform(len(score))
    sign = -1.0 if descending else 1.0
    return tuple(int(i) for i in sorted(range(len(score)), key=lambda i: (sign * score[i], lots[i])))


def borda(profile, n: int, rng: T.SplitMix64) -> tuple:
    """Borda: Summe der Plätze (0 = bester) über alle Bewerter, kleinste Summe zuerst."""
    pts = np.zeros(n)
    for order in profile:
        pts += positions(order)
    return order_by_score(pts, rng, descending=False)


def copeland(profile, n: int, rng: T.SplitMix64) -> tuple:
    """Copeland: je Teilnehmer die Zahl der paarweisen Mehrheitssiege (Gleichstand zählt halb), meiste zuerst."""
    W = prefer_matrix(profile, n)
    score = (W > W.T).sum(axis=1) + 0.5 * ((W == W.T).sum(axis=1) - 1)
    return order_by_score(score, rng, descending=True)


def _relabel(profile, perm):
    return [tuple(int(perm[i]) for i in order) for order in profile]


def _unmap(order, perm):
    inverse = {int(perm[i]): i for i in range(len(perm))}
    return tuple(inverse[x] for x in order)


def kemeny_dp(profile, n: int) -> tuple:
    """Exaktes Kemeny per Teilmengen-DP über alle 2^n Präfixmengen: (kleinste Gesamtkosten, eine optimale Rangfolge, Zahl der optimalen Rangfolgen).
    Kosten, wenn x hinter die Menge S gesetzt wird: Zahl der Bewerter, die x vor ein Element von S setzen."""
    W = prefer_matrix(profile, n).tolist()
    inf = 10 ** 12
    full = (1 << n) - 1
    best = [inf] * (full + 1)
    count = [0] * (full + 1)
    parent = [(-1, -1)] * (full + 1)
    best[0], count[0] = 0, 1
    for S in range(full + 1):
        if best[S] >= inf:
            continue
        members = [s for s in range(n) if S >> s & 1]
        for x in range(n):
            if S >> x & 1:
                continue
            cost = best[S] + sum(W[x][s] for s in members)
            nxt = S | (1 << x)
            if cost < best[nxt]:
                best[nxt], count[nxt], parent[nxt] = cost, count[S], (S, x)
            elif cost == best[nxt]:
                count[nxt] += count[S]
    order, S = [], full
    while S:
        S0, x = parent[S]
        order.append(x)
        S = S0
    return best[full], tuple(order[::-1]), count[full]


def kemeny(profile, n: int, rng: T.SplitMix64) -> tuple:
    """Kemeny mit zufälliger Umbenennung gegen Index-Artefakte bei Gleichstand. Rückgabe (Kosten, Rangfolge, Zahl der Optima)."""
    perm = np.argsort(rng.uniform(n), kind="stable")
    cost, order, count = kemeny_dp(_relabel(profile, perm), n)
    return cost, _unmap(order, perm), count


def footrule_cost_matrix(profile, n: int) -> np.ndarray:
    """C[i, j] = Summe über Bewerter von |Platz von i beim Bewerter - j|: die Kosten, Teilnehmer i auf Platz j zu setzen."""
    cost = np.zeros((n, n))
    for order in profile:
        cost += np.abs(positions(order)[:, None] - np.arange(n)[None, :])
    return cost


def footrule_optimal(profile, n: int, rng: T.SplitMix64) -> tuple:
    """Footrule-optimale Aggregation als Zuordnungsproblem (Dwork, Kumar, Naor, Sivakumar 2001), gelöst per scipy.optimize.linear_sum_assignment."""
    perm = np.argsort(rng.uniform(n), kind="stable")
    cost = footrule_cost_matrix(_relabel(profile, perm), n)
    rows, cols = linear_sum_assignment(cost)
    order = [0] * n
    for i, j in zip(rows, cols):
        order[j] = int(i)
    return _unmap(tuple(order), perm)


def footrule_optimal_brute(profile, n: int) -> tuple:
    """Gegenprobe: kleinste Footrule-Kosten durch Aufzählung aller Zuordnungen (nur kleine n). Rückgabe (Kosten, Zahl der Optima)."""
    cost = footrule_cost_matrix(profile, n)
    best, count = math.inf, 0
    for perm in itertools.permutations(range(n)):
        c = sum(cost[perm[j]][j] for j in range(n))
        if c < best - 1e-9:
            best, count = c, 1
        elif abs(c - best) <= 1e-9:
            count += 1
    return best, count


def majority_cycles(profile, n: int) -> int:
    """Zahl der Dreiergruppen mit zyklischer Mehrheit (a schlägt b, b schlägt c, c schlägt a): Spur der dritten Potenz der Mehrheitsmatrix durch 3."""
    W = prefer_matrix(profile, n)
    B = (W > W.T).astype(np.int64)
    return int(np.trace(B @ B @ B) // 3)


def condorcet_winner(profile, n: int):
    """Der Teilnehmer, der jeden anderen paarweise mit strenger Mehrheit schlägt, sonst None."""
    W = prefer_matrix(profile, n)
    for i in range(n):
        if all(W[i, j] > W[j, i] for j in range(n) if j != i):
            return i
    return None


def run_all(profile, n: int, rng: T.SplitMix64) -> dict:
    """Alle Verfahren auf ein Profil: Name -> Rangfolge; dazu Kemeny-Kosten und Zahl der Optima."""
    k_cost, k_order, k_count = kemeny(profile, n, rng)
    return {"orders": {"Borda": borda(profile, n, rng), "Copeland": copeland(profile, n, rng), "Kemeny": k_order,
                       "Footrule-optimal": footrule_optimal(profile, n, rng)},
            "kemeny_cost": k_cost, "kemeny_optima": k_count}
