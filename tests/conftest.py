"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def cycle3():
    """3 Teilnehmer A=0, B=1, C=2 und 3 Bewerter: (A, B, C), (B, C, A), (C, A, B).
    Handrechnung: Mehrheiten A vor B 2:1, B vor C 2:1, C vor A 2:1: ein Zyklus, kein Mehrheitssieger. Platzsummen (0 = bester): A = 0+2+1 = 3,
    B = 1+0+2 = 3, C = 2+1+0 = 3: Borda-Gleichstand. Copeland: je ein Mehrheitssieg, Gleichstand. Kemeny: die drei zyklischen Verschiebungen
    (A,B,C), (B,C,A), (C,A,B) haben Kosten 0+2+2 = 4 und sind die Optima (3 Stück); jede andere Ordnung kostet mehr (z. B. (A,C,B): 1+3+1 = 5).
    Footrule: Kosten je Zuordnung 8, alle 6 Zuordnungen gleich."""
    return [(0, 1, 2), (1, 2, 0), (2, 0, 1)]


@pytest.fixture
def clear_winner():
    """4 Teilnehmer, 3 Bewerter, fast einig: A=0 schlägt alle (3 von 3 Bewertern gegen jeden). (0,1,2,3), (0,1,3,2), (0,2,1,3).
    Handrechnung: Condorcet-Sieger 0. Platzsummen: 0: 0+0+0 = 0; 1: 1+1+2 = 4; 2: 2+3+1 = 6; 3: 3+2+3 = 8: Borda-Reihenfolge 0, 1, 2, 3.
    Kemeny: (0,1,2,3) kostet 0+1+1 = 2 und ist das Optimum."""
    return [(0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3)]
