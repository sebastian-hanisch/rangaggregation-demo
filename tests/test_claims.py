"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, nie ein einzelner Seed).
Der Generator hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig; Schwellen trotzdem mit Abstand zur Messung."""

import pytest

import rga_constants as C
import rga_evaluation as ev


@pytest.fixture(scope="module")
def methods():
    return {r["m"]: r for r in ev.sweep_methods()}


@pytest.fixture(scope="module")
def pairs():
    return {(r["noise"], r["m"]): r for r in ev.sweep_pairs()}


@pytest.fixture(scope="module")
def cycles():
    return ev.sweep_cycles()


@pytest.fixture(scope="module")
def quality():
    return {r["m"]: r for r in ev.sweep_quality()}


# --- Experiment 1 ------------------------------------------------------------------------------------------------------------------------

def test_a_single_table_is_far_from_the_truth_and_all_methods_agree_for_one_voter(methods):
    assert methods[15]["single"] == pytest.approx(6.47, abs=0.006)
    one = methods[1]
    assert one["Borda"] == one["Copeland"] == one["Kemeny"] == one["Footrule-optimal"] == pytest.approx(one["single"])


def test_distance_to_the_truth_with_fifteen_voters(methods):
    m = methods[15]
    assert m["Borda"] == pytest.approx(1.47, abs=0.006) and m["Copeland"] == pytest.approx(1.76, abs=0.006)
    assert m["Kemeny"] == pytest.approx(1.76, abs=0.006) and m["Footrule-optimal"] == pytest.approx(1.81, abs=0.006)
    assert m["Borda"] < m["Kemeny"] < m["Footrule-optimal"]


def test_pooling_all_games_beats_every_aggregation_and_more_voters_always_help(methods):
    for m in (3, 5, 9, 15):
        assert methods[m]["pooled"] < min(methods[m][name] for name in C.METHODS), m
    assert methods[15]["pooled"] == pytest.approx(1.10, abs=0.006)
    borda = [methods[m]["Borda"] for m in (1, 3, 5, 9, 15)]
    assert borda == sorted(borda, reverse=True)


# --- Experiment 2 ------------------------------------------------------------------------------------------------------------------------

def test_kemeny_is_farther_from_the_truth_than_borda_under_tournament_noise(pairs):
    r = pairs[("turnier", 15)]
    assert r["mean"] == pytest.approx(0.29, abs=0.006) and r["se"] == pytest.approx(0.06, abs=0.006)
    assert r["better"] == pytest.approx(0.19, abs=0.006) and r["worse"] == pytest.approx(0.40, abs=0.006)
    assert all(pairs[("turnier", m)]["mean"] > 2 * pairs[("turnier", m)]["se"] for m in (5, 9, 15))


def test_under_mallows_noise_the_two_are_indistinguishable(pairs):
    for m in (5, 9, 15):
        r = pairs[("mallows", m)]
        assert abs(r["mean"]) < 2 * r["se"], m
    assert pairs[("mallows", 15)]["mean"] == pytest.approx(-0.03, abs=0.006)
    assert pairs[("mallows", 15)]["se"] == pytest.approx(0.08, abs=0.006)


# --- Experiment 3 ------------------------------------------------------------------------------------------------------------------------

def test_cycles_among_eight_participants_and_the_rarer_missing_winner(cycles):
    rows = {r["m"]: r for r in cycles["turnier"]}
    assert rows[3]["cycle"] == pytest.approx(0.40, abs=0.006) and rows[15]["cycle"] == pytest.approx(0.15, abs=0.006)
    assert 1 - rows[3]["winner"] == pytest.approx(0.09, abs=0.006) and 1 - rows[15]["winner"] == pytest.approx(0.03, abs=0.006)
    assert all(1 - r["winner"] < r["cycle"] for r in rows.values())


def test_random_rankings_of_three_candidates_approach_the_limit(cycles):
    assert cycles["exact"] == {"cycles": 12, "profiles": 216}
    rows = {r["m"]: r["cycle"] for r in cycles["random"]}
    assert rows[3] == pytest.approx(12 / 216, abs=0.01) and rows[201] == pytest.approx(0.0877, abs=0.012)
    assert rows[3] < rows[11] < rows[201] + 0.005


# --- Experiment 4 ------------------------------------------------------------------------------------------------------------------------

def test_kemeny_is_often_not_unique_and_borda_often_not_cost_optimal(quality):
    assert quality[3]["nonunique"] == pytest.approx(0.23, abs=0.006) and quality[15]["nonunique"] == pytest.approx(0.09, abs=0.006)
    assert quality[3]["borda_not_optimal"] == pytest.approx(0.73, abs=0.006) and quality[15]["borda_not_optimal"] == pytest.approx(0.67, abs=0.006)
    assert quality[5]["borda_not_optimal"] == pytest.approx(0.76, abs=0.006)
    assert all(0.67 <= q["borda_not_optimal"] <= 0.76 + 0.006 for q in quality.values())


def test_footrule_assignment_stays_close_to_the_kemeny_cost_and_below_the_guarantee(quality):
    assert quality[3]["ratio_mean"] == pytest.approx(1.10, abs=0.006) and quality[15]["ratio_mean"] == pytest.approx(1.02, abs=0.006)
    assert quality[3]["ratio_max"] == pytest.approx(1.47, abs=0.006) and quality[15]["ratio_max"] == pytest.approx(1.09, abs=0.006)
    assert all(q["ratio_max"] <= 2.0 for q in quality.values())


# --- Texte -------------------------------------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_values(methods, pairs, cycles, quality):
    level, text = ev.verdict_methods(list(methods.values()))
    assert "Borda 1,47" in text and "1,10" in text
    level, text = ev.verdict_pairs(list(pairs.values()))
    assert level == "warning" and "0,29" in text and "-0,03" in text.replace("−", "-")
    level, text = ev.verdict_cycles(cycles)
    assert "12 von 216" in text and "5,6 %" in text
    level, text = ev.verdict_cost(list(quality.values()), ev.time_kemeny())
    assert "Garantie: höchstens 2" in text and "2ⁿ" in text
