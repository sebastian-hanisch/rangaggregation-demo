"""Literatur-Referenzwerte zur Rangaggregation (ORACLE-PLAYBOOK §5.1). Einsicht jeweils am 2026-10-17 über WebFetch (Seitentext).

Q1  Wikipedia, "Kemeny-Young method" (https://en.wikipedia.org/wiki/Kemeny%E2%80%93Young_method), Abschnitt Beispiel (Hauptstädte von
    Tennessee): 42 % Memphis>Nashville>Chattanooga>Knoxville, 26 % Nashville>Chattanooga>Knoxville>Memphis, 15 % Chattanooga>Knoxville>
    Nashville>Memphis, 17 % Knoxville>Chattanooga>Nashville>Memphis; paarweise Zählungen (Memphis gegen jeden anderen 42:58, Nashville
    gegen Chattanooga und Knoxville 68:32, Chattanooga gegen Knoxville 83:17); die Tabelle der Rangscores aller 24 Rangfolgen
    (Rangscore = Summe der paarweisen Zählungen, die mit der Rangfolge übereinstimmen), Höchstwert 393 für Nashville>Chattanooga>
    Knoxville>Memphis. Die Demo zählt Kosten (Uneinigkeiten): bei 100 Bewertern und 6 Paaren ist Kosten = 600 - Rangscore.
Q2  Wikipedia, "Borda count" (https://en.wikipedia.org/wiki/Borda_count), Abschnitt Beispiel: dieselben Wähler, Punkte 3/2/1/0:
    Memphis 126, Nashville 194, Knoxville 107, Chattanooga 173 ("Nashville is elected"); zweites Beispiel mit 3 Bewertern und
    4 Kandidaten (U und V: A>B>C>D, W: B>C>D>A): A 6, B 7, C 4, D 1, B gewählt. Die Demo zählt Platzsummen (0 = bester Platz), also
    Punkte = m (n-1) - Platzsumme.
Q3  Wikipedia, "Copeland's method" (https://en.wikipedia.org/wiki/Copeland%27s_method), Beispiele 1 und 3: Tennessee-Profil:
    Copeland-Wertung (1/½/0) Nashville 3, Chattanooga 2, Knoxville 1, Memphis 0. Beispiel 3 (5 Bewerter: 2x A>B>C>D, 2x B>C>D>A,
    1x D>A>B>C): paarweise A:B 3:2, A:C 3:2, A:D 2:3, B:C 5:0, B:D 4:1, C:D 4:1; Wertung A 2, B 2, C 1, D 1, kein Condorcet-Sieger;
    Fußnote: Borda-Punkte A 8, B 11, C 6, D 5. (Beispiel 2 mit unvollständigen Stimmzetteln hat in der Demo kein Gegenstück: Teillisten
    sind nicht Teil des Modells.)
Q4  Wikipedia, "Condorcet paradox" (https://en.wikipedia.org/wiki/Condorcet_paradox), Abschnitt über die Wahrscheinlichkeit bei
    gleichverteilten Rangfolgen und drei Kandidaten: 3 Wähler 5,556 %, 201 Wähler 8,732 %, Grenzwert 8,77 % (arcsin(√6/9)/π).

Nicht eingesehen und deshalb ohne Test: Kemeny (1959), Young/Levenglick (1978), Dwork et al. (2001, Footrule-Aggregation), Mallows
(1957), Doignon et al. (2004). Für die Footrule-optimale Aggregation, den Mallows-Sampler und die Kemeny-Rechenzeit liegt kein Zahlenwert
einer eingesehenen Quelle vor.
"""

import itertools
import math

import numpy as np
import pytest

import rga_aggregate as A
import rga_noise as N
import turnier_generator as T

M_, N_, C_, K_ = 0, 1, 2, 3          # Memphis, Nashville, Chattanooga, Knoxville
LETTER = {"M": M_, "N": N_, "C": C_, "K": K_}

TENNESSEE = (42 * [(M_, N_, C_, K_)] + 26 * [(N_, C_, K_, M_)] + 15 * [(C_, K_, N_, M_)] + 17 * [(K_, C_, N_, M_)])

# Q1: Rangscores aller 24 Rangfolgen (Reihenfolge der Quelle)
KEMENY_YOUNG_SCORES = {
    "MNCK": 345, "MNKC": 279, "MCNK": 309, "MCKN": 273, "MKNC": 243, "MKCN": 207,
    "NMCK": 361, "NMKC": 295, "NCMK": 377, "NCKM": 393, "NKMC": 311, "NKCM": 327,
    "CMNK": 325, "CMKN": 289, "CNMK": 341, "CNKM": 357, "CKMN": 305, "CKNM": 321,
    "KMNC": 259, "KMCN": 223, "KNMC": 275, "KNCM": 291, "KCMN": 239, "KCNM": 255,
}


def rng(seed=3):
    return T.SplitMix64(seed)


def order_of(word):
    return tuple(LETTER[c] for c in word)


# --- Q1: Kemeny-Young ---------------------------------------------------------------------------------------------------------------------

def test_transcribed_table_is_complete_and_consistent():
    """Lesekontrolle der abgeschriebenen Tabelle: 24 verschiedene Rangfolgen, und jedes Paar wird in genau der Hälfte der Rangfolgen in
    jeder Richtung gezählt, also Summe der Scores = 12 * (6 Paare * 100)."""
    assert set(KEMENY_YOUNG_SCORES) == {"".join(p) for p in itertools.permutations("MNCK")}
    assert sum(KEMENY_YOUNG_SCORES.values()) == 12 * 600


def test_pairwise_counts_of_the_tennessee_profile():
    W = A.prefer_matrix(TENNESSEE, 4)
    for other in (N_, C_, K_):
        assert (W[M_, other], W[other, M_]) == (42, 58)
    assert (W[N_, C_], W[C_, N_]) == (68, 32)
    assert (W[N_, K_], W[K_, N_]) == (68, 32)
    assert (W[C_, K_], W[K_, C_]) == (83, 17)


def test_cost_of_every_ranking_is_600_minus_the_ranking_score_of_the_source():
    for word, score in KEMENY_YOUNG_SCORES.items():
        assert A.total_cost(order_of(word), TENNESSEE) == 600 - score


def test_kemeny_returns_nashville_chattanooga_knoxville_memphis_with_score_393():
    for seed in range(8):
        cost, order, count = A.kemeny(TENNESSEE, 4, rng(seed))
        assert order == order_of("NCKM") and cost == 600 - 393 and count == 1
    assert A.kemeny_dp(TENNESSEE, 4) == (207, order_of("NCKM"), 1)


# --- Q2: Borda ------------------------------------------------------------------------------------------------------------------------------

def test_borda_points_and_winner_of_the_tennessee_example():
    sums = sum(A.positions(o) for o in TENNESSEE)
    points = 100 * 3 - sums
    assert points.tolist() == [126, 194, 173, 107]              # Memphis, Nashville, Chattanooga, Knoxville
    for seed in range(5):
        assert A.borda(TENNESSEE, 4, rng(seed)) == order_of("NCMK")      # 194 > 173 > 126 > 107


def test_borda_three_voter_example_elects_brian_although_andrew_is_the_condorcet_winner():
    A_, B_, C2, D_ = 0, 1, 2, 3
    profile = [(A_, B_, C2, D_), (A_, B_, C2, D_), (B_, C2, D_, A_)]
    points = 3 * 3 - sum(A.positions(o) for o in profile)
    assert points.tolist() == [6, 7, 4, 1]
    assert A.borda(profile, 4, rng())[0] == B_
    assert A.condorcet_winner(profile, 4) == A_                  # A schlägt jeden 2:1 (Handrechnung zur Quelle)
    assert A.copeland(profile, 4, rng())[0] == A_


# --- Q3: Copeland -----------------------------------------------------------------------------------------------------------------------------

def test_copeland_on_the_tennessee_profile():
    W = A.prefer_matrix(TENNESSEE, 4)
    wins = (W > W.T).sum(axis=1)
    assert wins.tolist() == [0, 3, 2, 1]                        # Memphis, Nashville, Chattanooga, Knoxville
    for seed in range(5):
        assert A.copeland(TENNESSEE, 4, rng(seed)) == order_of("NCKM")
    assert A.condorcet_winner(TENNESSEE, 4) == N_
    assert A.majority_cycles(TENNESSEE, 4) == 0


def test_copeland_example_with_a_condorcet_cycle_among_four_candidates():
    a, b, c, d = 0, 1, 2, 3
    profile = 2 * [(a, b, c, d)] + 2 * [(b, c, d, a)] + [(d, a, b, c)]
    W = A.prefer_matrix(profile, 4)
    expected = {(a, b): 3, (a, c): 3, (a, d): 2, (b, c): 5, (b, d): 4, (c, d): 4}
    for (x, y), v in expected.items():
        assert (W[x, y], W[y, x]) == (v, 5 - v)
    assert A.condorcet_winner(profile, 4) is None
    # Handrechnung aus den Zählungen der Quelle: Zyklen A>B>D>A und A>C>D>A, also 2 (= C(4,3) - C(2,2) - C(2,2) - 0 - 0)
    assert A.majority_cycles(profile, 4) == 2
    points = 5 * 3 - sum(A.positions(o) for o in profile)
    assert points.tolist() == [8, 11, 6, 5]                     # Borda-Punkte der Fußnote
    tops, bottoms = set(), set()
    for seed in range(30):
        order = A.copeland(profile, 4, rng(seed))
        tops.add(order[:2])
        bottoms.add(order[2:])
    assert {frozenset(t) for t in tops} == {frozenset({a, b})}   # A und B teilen sich Platz 1 und 2 (Wertung 2)
    assert {frozenset(t) for t in bottoms} == {frozenset({c, d})}
    assert len(tops) == 2 and len(bottoms) == 2                  # das Los entscheidet beide Gleichstände


# --- Q4: Condorcet-Paradox -------------------------------------------------------------------------------------------------------------

def test_condorcet_cycle_probability_for_three_candidates_and_three_voters_is_5_556_percent():
    all_orders = list(itertools.permutations(range(3)))
    cycles = sum(A.majority_cycles(list(profile), 3) > 0 for profile in itertools.product(all_orders, repeat=3))
    assert cycles == 12 and len(all_orders) ** 3 == 216
    assert round(100 * cycles / 216, 3) == 5.556


def test_condorcet_cycle_probability_for_201_voters_is_close_to_8_732_percent():
    """Monte-Carlo (feste Seeds, plattformunabhängiger Strom): 12000 Profile; Standardfehler 0,26 Prozentpunkte, geprüft wird 5 Sigma."""
    trials = 12_000
    r = T.SplitMix64(2026)
    hits = sum(N.random_profile_has_cycle3(201, r) for _ in range(trials))
    p = 0.08732
    sigma = math.sqrt(p * (1 - p) / trials)
    assert abs(hits / trials - p) < 5 * sigma
    # und die Grenzformel arcsin(sqrt(6)/9)/pi ist 8,77 %
    assert math.asin(math.sqrt(6) / 9) / math.pi == pytest.approx(0.0877, abs=5e-5)
