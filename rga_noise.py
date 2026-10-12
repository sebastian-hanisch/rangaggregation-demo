"""Wie entstehen die Rangfolgen der Bewerter? Zwei Rauschmodelle um dieselbe wahre Ordnung (Generator der Wurzel, Version 1):
Tabellen aus je einem eigenen kleinen Turnier, oder zufällige Abweichungen nach dem Mallows-Modell. Zufall nur über SplitMix64."""

import numpy as np

import rga_aggregate as A
import turnier_generator as T


def truth_order(n: int) -> tuple:
    """Wahre Ordnung des Generators: Index n-1 ist der Stärkste."""
    return tuple(range(n - 1, -1, -1))


def turnier_wins(n: int, m: int, k: int, delta: float, seed: int) -> list:
    """Siegvektor je Bewerter: ein eigenes vollständiges Turnier (k Partien je Paar), gleiche Stärken, verschiedene Partien."""
    out = []
    for j in range(m):
        t = T.build_turnier(n, delta, k, "vollstaendig", 1, 0.5, 0.0, "ausgewogen", seed * 1000 + j)
        out.append(t.wins.sum(axis=1).astype(float))
    return out


def turnier_profile(n: int, m: int, k: int, delta: float, seed: int) -> list:
    """Je Bewerter die Tabelle nach Siegen seines Turniers, Gleichstand durch das Los."""
    profile = []
    for j, wins in enumerate(turnier_wins(n, m, k, delta, seed)):
        profile.append(A.order_by_score(wins, T.SplitMix64((seed * 1000 + j) ^ 0xBEEF)))
    return profile


def pooled_table(n: int, m: int, k: int, delta: float, seed: int) -> tuple:
    """Gegenprobe: alle Partien der m Turniere in EINE Tabelle (Summe der Siege), Gleichstand durch das Los."""
    total = np.sum(turnier_wins(n, m, k, delta, seed), axis=0)
    return A.order_by_score(total, T.SplitMix64(seed ^ 0xB00D))


def mallows_sample(n: int, phi: float, rng: T.SplitMix64) -> tuple:
    """Eine Ziehung aus dem Mallows-Modell um die Referenz (0, 1, ..., n-1) per wiederholtem Einfügen (Doignon, Pekeč, Regenwetter 2004):
    Element i wird an Stelle j (0 = ganz vorn) mit Wahrscheinlichkeit proportional zu phi^(i - j) eingefügt."""
    if not 0.0 < phi <= 1.0:
        raise ValueError("phi muss in (0, 1] liegen")
    order = []
    u = rng.uniform(n)
    for i in range(n):
        w = np.array([phi ** (i - j) for j in range(i + 1)])
        cdf = np.cumsum(w / w.sum())
        j = int(np.searchsorted(cdf, u[i], side="right"))
        order.insert(min(j, i), i)
    return tuple(order)


def mallows_profile(n: int, m: int, phi: float, seed: int) -> list:
    """m Ziehungen um die wahre Ordnung (Stärkster zuerst)."""
    rng = T.SplitMix64(seed ^ 0x11AA)
    return [tuple(n - 1 - x for x in mallows_sample(n, phi, rng)) for _ in range(m)]


def random_profile_has_cycle3(m: int, rng: T.SplitMix64) -> bool:
    """Schnelle Fassung für 3 Teilnehmer: m gleichverteilt zufällige Rangfolgen, hat die Mehrheitsrelation einen Zyklus? (Wie `random_profile`
    plus `majority_cycles`, aber ohne Schleife über die Bewerter.)"""
    u = rng.uniform(3 * m).reshape(m, 3)
    pos = np.argsort(np.argsort(u, axis=1, kind="stable"), axis=1, kind="stable")      # pos[v, i] = Platz von Teilnehmer i beim Bewerter v
    W = (pos[:, :, None] < pos[:, None, :]).sum(axis=0)
    beat = W > W.T
    return bool((beat[0, 1] and beat[1, 2] and beat[2, 0]) or (beat[0, 2] and beat[2, 1] and beat[1, 0]))


def random_profile(n: int, m: int, rng: T.SplitMix64) -> list:
    """m gleichverteilt zufällige Rangfolgen (alle n! gleich wahrscheinlich, „impartial culture“)."""
    return [tuple(int(i) for i in np.argsort(rng.uniform(n), kind="stable")) for _ in range(m)]
