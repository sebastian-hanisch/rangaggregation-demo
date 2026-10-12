"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 4, 12, 8             # n <= 12: das exakte Kemeny (Teilmengen-DP) rechnet dann unter 0,1 s
M_MIN, M_MAX, DEFAULT_M = 1, 15, 9             # Zahl der Bewerter
K_MIN, K_MAX, DEFAULT_K = 1, 10, 1             # Partien je Paar im Turnier jedes Bewerters (nur Rauschmodell "turnier")
PHI_MIN, PHI_MAX, PHI_STEP, DEFAULT_PHI = 0.3, 0.95, 0.05, 0.8   # Mallows-Parameter (nur Rauschmodell "mallows")
SEED_MAX, DEFAULT_SEED = 999999, 35
DELTA = 0.3                                    # Stärkeabstand im Turnier jedes Bewerters (Standardinstanz der Wurzel)

NOISE_LABELS = {"turnier": "Tabellen aus kleinen Turnieren", "mallows": "Mallows-Rauschen um die wahre Ordnung"}
METHODS = ("Borda", "Copeland", "Kemeny", "Footrule-optimal")

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100300))
EXP_N = 8                          # Experimente 1 bis 3: feste Instanz mit 8 Teilnehmern
EXP1_MS = (1, 3, 5, 9, 15)
EXP2_MS = (5, 9, 15)
EXP2_PHI = 0.7
EXP3_MS = (3, 5, 9, 15)
EXP3_RANDOM_MS = (3, 5, 11, 51, 201)
EXP3_RANDOM_DRAWS = 3000
EXP4_NS = (6, 8, 10, 12)

COLOR_MAIN = "#2F6B65"
COLOR_BORDA = "#c0392b"
COLOR_COPELAND = "#e08e0b"
COLOR_KEMENY = "#4c78a8"
COLOR_FOOTRULE = "#8e6bb0"
COLOR_POOLED = "#222222"
COLOR_SINGLE = "rgba(150,150,150,0.8)"

PRESETS = {
    "Ein Bewerter allein": {"n": DEFAULT_N, "m": 1, "noise": "turnier", "k": DEFAULT_K, "phi": DEFAULT_PHI, "seed": DEFAULT_SEED},
    "Viele ungenaue Bewerter": {"n": DEFAULT_N, "m": 15, "noise": "turnier", "k": DEFAULT_K, "phi": DEFAULT_PHI, "seed": DEFAULT_SEED},
    "Kein Mehrheitssieger": {"n": 4, "m": 3, "noise": "turnier", "k": DEFAULT_K, "phi": DEFAULT_PHI, "seed": 7},
    "Mallows-Rauschen": {"n": DEFAULT_N, "m": 9, "noise": "mallows", "k": DEFAULT_K, "phi": 0.7, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Ein Bewerter allein": "Nur eine Tabelle aus einem kleinen Turnier: Alle Verfahren geben sie unverändert zurück, die Aggregation hat nichts zu tun.",
    "Viele ungenaue Bewerter": "15 Tabellen aus je einem kleinen Turnier: Die Verfahren führen sie zusammen, am nächsten an der Wahrheit liegt oft Borda.",
    "Kein Mehrheitssieger": "3 Bewerter, 4 Teilnehmer, eine Mehrheit, die im Kreis läuft: kein Teilnehmer schlägt alle anderen (Condorcet-Paradoxon).",
    "Mallows-Rauschen": "9 Rangfolgen, jede eine zufällige Abweichung von der wahren Ordnung nach dem Mallows-Modell (φ = 0,7).",
}
