"""Szenario: Reglerwerte (Settings) und das daraus erzeugte Profil der Bewerter."""

from dataclasses import dataclass

import rga_constants as C
import rga_noise as N
import turnier_generator as T


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    noise: str = "turnier"
    k: int = C.DEFAULT_K
    phi: float = C.DEFAULT_PHI
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class ProfileData:
    profile: tuple         # m Rangfolgen (Tupel von Teilnehmerindizes, bester zuerst)
    truth: tuple           # wahre Ordnung
    names: tuple           # Anzeigename je Teilnehmerindex (zufällig vergeben, damit der Index die Stärke nicht verrät)
    pooled: tuple | None   # alle Partien in eine Tabelle (nur Rauschmodell "turnier")


def make_profile(s: Settings) -> ProfileData:
    """Deterministisch aus den Reglerwerten: gleiche Eingabe, gleiches Profil."""
    perm = T.SplitMix64(s.seed ^ 0xA11CE).permutation(s.n)
    names = tuple(T.LABELS[int(perm[i])] for i in range(s.n))
    truth = N.truth_order(s.n)
    if s.noise == "turnier":
        profile = N.turnier_profile(s.n, s.m, s.k, C.DELTA, s.seed)
        pooled = N.pooled_table(s.n, s.m, s.k, C.DELTA, s.seed)
    elif s.noise == "mallows":
        profile, pooled = N.mallows_profile(s.n, s.m, s.phi, s.seed), None
    else:
        raise ValueError(f"unbekanntes Rauschmodell: {s.noise}")
    return ProfileData(tuple(profile), truth, names, pooled)
