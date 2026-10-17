"""Kennzahlen, Verdikt-Logik und Experimente (über die festen Instanzen SWEEP_SEEDS)."""

import functools
import itertools
import math
import time
from dataclasses import dataclass

import numpy as np

import rga_aggregate as A
import rga_constants as C
import rga_noise as N
import turnier_generator as T
from rga_scenario import ProfileData, Settings, make_profile

_RUN_SALT = 0xA66


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    data: ProfileData
    orders: dict            # Verfahren -> Rangfolge
    costs: dict             # Verfahren -> Summe der Kendall-Distanzen zu den Eingaben
    dist_truth: dict        # Verfahren -> Kendall-Distanz zur wahren Ordnung
    top1: dict              # Verfahren -> steht der Stärkste auf Platz 1?
    kemeny_cost: int
    kemeny_optima: int
    cycles: int             # zyklische Dreiergruppen der Mehrheitsrelation
    triples: int
    winner: int | None      # Condorcet-Sieger oder None
    single_mean_dist: float  # mittlerer Abstand der Eingaben zur Wahrheit
    pooled_dist: int | None
    prefer: np.ndarray      # W[i, j]: Bewerter, die i vor j setzen


@functools.lru_cache(maxsize=8192)
def analyse(settings: Settings) -> Analysis:
    """Gleiche Reglerwerte, gleiche Analyse: der Zwischenspeicher lässt die Experimente dieselben Turniere teilen (Settings ist unveränderlich)."""
    data = make_profile(settings)
    n = settings.n
    res = A.run_all(data.profile, n, T.SplitMix64(settings.seed ^ _RUN_SALT))
    orders = res["orders"]
    truth = data.truth
    return Analysis(
        settings=settings, data=data, orders=orders,
        costs={name: A.total_cost(o, data.profile) for name, o in orders.items()},
        dist_truth={name: A.kendall_distance(o, truth) for name, o in orders.items()},
        top1={name: o[0] == truth[0] for name, o in orders.items()},
        kemeny_cost=res["kemeny_cost"], kemeny_optima=res["kemeny_optima"],
        cycles=A.majority_cycles(data.profile, n), triples=math.comb(n, 3), winner=A.condorcet_winner(data.profile, n),
        single_mean_dist=float(np.mean([A.kendall_distance(o, truth) for o in data.profile])),
        pooled_dist=None if data.pooled is None else A.kendall_distance(data.pooled, truth),
        prefer=A.prefer_matrix(data.profile, n),
    )


def de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def pct1(x: float) -> str:
    return f"{100 * x:.1f}".replace(".", ",")


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else float("nan")


def _se(values) -> float:
    v = np.asarray(values, dtype=float)
    return float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0


def _runs(make_settings, seeds):
    return [analyse(make_settings(s)) for s in seeds]


# --- Experiment 1: wer führt am besten zusammen? ---------------------------------------------------------------------------------------

def sweep_methods(ms=C.EXP1_MS, n=C.EXP_N, noise="turnier", k=1, phi=C.DEFAULT_PHI, seeds=C.SWEEP_SEEDS) -> list[dict]:
    rows = []
    for m in ms:
        runs = _runs(lambda s: Settings(n=n, m=m, noise=noise, k=k, phi=phi, seed=s), seeds)
        row = {"m": m, "single": _mean([r.single_mean_dist for r in runs])}
        for name in C.METHODS:
            row[name] = _mean([float(r.dist_truth[name]) for r in runs])
            row[name + "_top1"] = _mean([float(r.top1[name]) for r in runs])
        pooled = [r.pooled_dist for r in runs if r.pooled_dist is not None]
        row["pooled"] = _mean([float(x) for x in pooled]) if pooled else None
        rows.append(row)
    return rows


# --- Experiment 2: zusammenfassen oder schätzen? ---------------------------------------------------------------------------------------

def sweep_pairs(ms=C.EXP2_MS, n=C.EXP_N, phi=C.EXP2_PHI, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Gepaarter Vergleich Kemeny gegen Borda im Abstand zur Wahrheit: Differenz (Kemeny minus Borda) je Profil."""
    rows = []
    for noise in ("turnier", "mallows"):
        for m in ms:
            # phi wirkt nur bei Mallows; bei Turnier-Tabellen den Standardwert lassen, damit Experiment 1 und 2 dieselben Analysen teilen
            runs = _runs(lambda s: Settings(n=n, m=m, noise=noise, k=1, phi=phi if noise == "mallows" else C.DEFAULT_PHI, seed=s), seeds)
            d = np.array([r.dist_truth["Kemeny"] - r.dist_truth["Borda"] for r in runs], dtype=float)
            rows.append({"noise": noise, "m": m, "mean": float(d.mean()), "se": _se(d), "better": float((d < 0).mean()),
                         "equal": float((d == 0).mean()), "worse": float((d > 0).mean()),
                         "borda_optimal": _mean([float(r.costs["Borda"] == r.kemeny_cost) for r in runs])})
    return rows


# --- Experiment 3: wann gibt es keinen Mehrheitssieger? ----------------------------------------------------------------------------------

def exact_cycle_probability_3x3() -> tuple[int, int]:
    """3 Teilnehmer, 3 Bewerter, alle 6 Rangfolgen gleich wahrscheinlich: (Zahl der Profile mit Mehrheitszyklus, Zahl aller Profile) durch Aufzählung."""
    perms = list(itertools.permutations(range(3)))
    cyc = 0
    for profile in itertools.product(perms, repeat=3):
        cyc += A.majority_cycles(list(profile), 3) > 0
    return cyc, len(perms) ** 3


def sweep_cycles(ms=C.EXP3_MS, ms_random=C.EXP3_RANDOM_MS, n=C.EXP_N, draws=C.EXP3_RANDOM_DRAWS, seeds=C.SWEEP_SEEDS) -> dict:
    turnier = []
    for m in ms:
        runs = _runs(lambda s: Settings(n=n, m=m, noise="turnier", k=1, seed=s), seeds)
        turnier.append({"m": m, "cycle": _mean([float(r.cycles > 0) for r in runs]), "winner": _mean([float(r.winner is not None) for r in runs])})
    rng = T.SplitMix64(2026)
    random_rows = []
    for m in ms_random:
        hits = sum(N.random_profile_has_cycle3(m, rng) for _ in range(draws))
        random_rows.append({"m": m, "cycle": hits / draws, "draws": draws})
    cyc, total = exact_cycle_probability_3x3()
    return {"turnier": turnier, "random": random_rows, "exact": {"cycles": cyc, "profiles": total}}


# --- Experiment 4: was kostet das Exakte? -----------------------------------------------------------------------------------------------

def sweep_quality(ms=C.EXP3_MS, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Kemeny nicht eindeutig, Borda nicht kostenoptimal, Kostenverhältnis der Footrule-Näherung zu Kemeny (die Garantie ist 2)."""
    rows = []
    for m in ms:
        runs = _runs(lambda s: Settings(n=n, m=m, noise="turnier", k=1, seed=s), seeds)
        ratios = [r.costs["Footrule-optimal"] / r.kemeny_cost for r in runs if r.kemeny_cost > 0]
        rows.append({"m": m, "nonunique": _mean([float(r.kemeny_optima > 1) for r in runs]),
                     "borda_not_optimal": _mean([float(r.costs["Borda"] > r.kemeny_cost) for r in runs]),
                     "ratio_mean": _mean(ratios), "ratio_max": max(ratios) if ratios else float("nan")})
    return rows


def time_kemeny(ns=C.EXP4_NS, m=9) -> list[dict]:
    """Rechenzeit des exakten Kemeny (Teilmengen-DP) gegen n. Zeiten hängen vom Rechner ab: nur Größenordnung und Wachstum zählen."""
    rows = []
    for n in ns:
        profile = N.turnier_profile(n, m, 1, C.DELTA, 5)
        t0 = time.perf_counter()
        A.kemeny_dp(profile, n)
        rows.append({"n": n, "seconds": time.perf_counter() - t0, "states": (1 << n) * n})
    return rows


# --- Verdikte (live aus den Messwerten) ---------------------------------------------------------------------------------------------------

def verdict_analysis(a: Analysis) -> tuple[str, str]:
    """Kernfrage der Primäransicht: Welches Verfahren liegt wo, und gibt es einen Mehrheitssieger?"""
    n, m = a.settings.n, a.settings.m
    if m == 1:
        return "info", "Ein einzelner Bewerter: Alle Verfahren geben seine Rangfolge unverändert zurück, die Aggregation hat nichts zu tun."
    best_cost = min(a.costs.values())
    cheapest = [x for x in C.METHODS if a.costs[x] == best_cost]
    closest = min(C.METHODS, key=lambda x: (a.dist_truth[x], C.METHODS.index(x)))
    text = (f"Kleinste Gesamtkosten zu den {m} Eingaben: **{', '.join(cheapest)}** ({best_cost}); der Wahrheit am nächsten liegt "
            f"**{closest}** (Kendall-Distanz {a.dist_truth[closest]} von höchstens {math.comb(n, 2)}). ")
    if a.winner is None:
        text += f"Es gibt **keinen Mehrheitssieger**; {a.cycles} von {a.triples} Dreiergruppen laufen im Kreis. "
    else:
        text += "Es gibt einen Mehrheitssieger (Condorcet). "
    if a.kemeny_optima > 1:
        text += f"Kemeny hat {a.kemeny_optima} gleich gute Rangfolgen; eine wurde zufällig gewählt."
    level = "warning" if a.winner is None or a.kemeny_optima > 1 else "info"
    return level, text.strip()


def verdict_methods(rows) -> tuple[str, str]:
    r = rows[-1]
    parts = ", ".join(f"{name} {de(r[name])}" for name in C.METHODS)
    text = (f"Mit {r['m']} Bewertern liegt der Abstand zur Wahrheit bei {parts}; eine einzelne Tabelle liegt bei {de(r['single'])}.")
    if r.get("pooled") is not None:
        text += (f" Alle Partien in eine Tabelle zu legen ergibt {de(r['pooled'])}: Jede Aggregation von Rangfolgen verliert Information, die in "
                 "den Rohdaten steckt.")
    return "info", text


def verdict_pairs(rows) -> tuple[str, str]:
    t = next(r for r in rows if r["noise"] == "turnier" and r["m"] == max(x["m"] for x in rows))
    ml = next(r for r in rows if r["noise"] == "mallows" and r["m"] == max(x["m"] for x in rows))
    return "warning", (f"Kemeny minimiert den Abstand zu den **Eingaben**, nicht zur Wahrheit. Bei Turnier-Tabellen liegt er im Mittel um "
                       f"{de(t['mean'])} ± {de(t['se'])} Paare hinter Borda (Kemeny besser in {pct(t['better'])} %, schlechter in {pct(t['worse'])} %); "
                       f"unter Mallows-Rauschen, für das er der Maximum-Likelihood-Schätzer sein soll, liegt der Unterschied bei {de(ml['mean'])} ± "
                       f"{de(ml['se'])}: in dieser Stichprobe kein erkennbarer Vorteil, aber auch zu wenig Auflösung für einen kleinen "
                       "(mit sehr viel mehr Profilen liegt Kemeny hier leicht vorn). Der Befund hängt am Rauschmodell.")


def verdict_cycles(res) -> tuple[str, str]:
    r3 = res["turnier"][0]
    exact = res["exact"]
    last = res["random"][-1]
    return "warning", (f"Bei {r3['m']} Bewertern und 8 Teilnehmern hat {pct(r3['cycle'])} % der Profile irgendwo einen Mehrheitszyklus, aber nur in "
                       f"{pct(1 - r3['winner'])} % fehlt ein Mehrheitssieger (der Zyklus betrifft die übrigen Teilnehmer, im Mittel nur leicht hinter der Mitte). Bei zufälligen Rangfolgen von "
                       f"3 Teilnehmern sind es mit 3 Bewertern exakt {exact['cycles']} von {exact['profiles']} ({pct1(exact['cycles'] / exact['profiles'])} %) "
                       f"und mit {last['m']} Bewertern {pct1(last['cycle'])} %, im Grenzwert rund 8,8 %: Mehr Bewerter beseitigen den Zyklus nicht.")


def verdict_cost(quality, timing) -> tuple[str, str]:
    q = quality[0]
    t_last, t_first = timing[-1], timing[0]
    return "info", (f"Das exakte Kemeny braucht auf diesem Rechner bei {t_last['n']} Teilnehmern {de(t_last['seconds'], 2)} s (bei {t_first['n']}: "
                    f"{de(t_first['seconds'], 3)} s). Der Aufwand wächst mit 2ⁿ · n, die Rechenzeit je zwei Teilnehmer mehr etwa auf das Vier- bis Fünffache; "
                    f"die Zeiten selbst hängen vom Rechner ab. Die Footrule-Zuordnung kostet im Mittel nur das {de(q['ratio_mean'], 2)}-Fache des "
                    f"Kemeny-Optimums (größter Wert {de(q['ratio_max'], 2)}, Garantie: höchstens 2). Borda ist bei {pct(q['borda_not_optimal'])} % der "
                    f"Profile mit {q['m']} Bewertern nicht kostenoptimal, Kemeny bei {pct(q['nonunique'])} % nicht eindeutig.")
