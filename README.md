# Borda, Copeland, Kemeny: viele Rangfolgen zu einer – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-rangaggregation-demo.streamlit.app/)**

Drittes Stück der **Ranking-Linie** (Konzepte-Demo), Nachfolger von `rangdistanz-demo`: Mehrere Bewerter liefern je eine Rangfolge, vier Verfahren führen
sie zu einer zusammen, vier Experimente messen, was das bringt. Das Vehikel ist dasselbe neutrale, simulierte Turnier wie in den ersten beiden Stücken:
Jeder Bewerter bringt die Tabelle eines **eigenen kleinen Turniers** mit (oder eine Zufallsabweichung nach dem Mallows-Modell) um dieselbe wahre
Ordnung. Alles ist simuliert, es gibt keinen echten Datensatz.

## Kernfrage

Aus vielen Rangfolgen eine zu machen heißt, ein **Ziel** zu wählen. **Borda** summiert die Plätze, **Copeland** zählt die paarweisen Mehrheiten,
**Kemeny** minimiert den Gesamtabstand (Summe der Kendall-Distanzen) zu den Eingaben, die **Footrule-Zuordnung** ist seine billige Näherung. Die
Eingaben am besten zusammenzufassen ist nicht dasselbe, wie der Wahrheit am nächsten zu kommen; und keine Aggregation ersetzt die Rohdaten.

## Modell

- n Teilnehmer (höchstens 12, damit das exakte Kemeny unter 0,1 s bleibt), m Bewerter (1 bis 15).
- **Rauschmodell Turnier-Tabellen:** Jeder Bewerter spielt ein eigenes vollständiges Turnier (k Partien je Paar, Generator der Wurzel, Version 1,
  wortgleich) und gibt dessen Tabelle nach Siegen ab, Gleichstand durch das Los.
- **Rauschmodell Mallows:** Jede Rangfolge ist eine Ziehung aus dem Mallows-Modell (Parameter φ) um die wahre Ordnung, per wiederholtem Einfügen.
- **Verfahren:** Borda, Copeland, Kemeny (exakt per Teilmengen-DP), Footrule-optimal (Zuordnung mit `scipy.optimize.linear_sum_assignment`).
  Gleichstände entscheidet bei Borda und Copeland das Los; Kemeny und die Zuordnung laufen nach einer zufälligen Umbenennung der Teilnehmer, damit der
  Index (der hier mit der Stärke korreliert) nichts entscheidet.
- **Referenz:** „alle Partien in eine Tabelle“ (die Siege aller Bewerter-Turniere summiert).
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 300 feste Profile (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt, auf einer festen Instanz
  mit 8 Teilnehmern, unabhängig vom Regler. Gepaarte Vergleiche auf denselben Profilen mit Standardfehler.
- **Abstand zur Wahrheit:** Kendall-Distanz zur wahren Ordnung (höchstens 28 bei 8 Teilnehmern).
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom; das Ergebnis hängt nur von Seed und Position ab.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 8 Teilnehmer, Turnier-Tabellen mit einer Partie je Paar, 300 feste Profile, wenn nicht anders genannt.

| Frage | Befund | Test |
|---|---|---|
| Wie gut führen die Verfahren 15 Tabellen zusammen? | Eine einzelne Tabelle liegt im Mittel bei 6,47 (von höchstens 28). **Borda 1,47**, Copeland 1,76, Kemeny 1,76, Footrule-optimal 1,81. | `test_distance_to_the_truth_with_fifteen_voters` |
| Hilft es, die Rohdaten zu behalten? | **Alle Partien in eine Tabelle: 1,10**, besser als jede Aggregation bei jeder Zahl von Bewertern (3, 5, 9, 15). Aggregation verliert Information. | `test_pooling_all_games_beats_every_aggregation_and_more_voters_always_help` |
| Ist die genaueste Zusammenfassung (Kemeny) auch die der Wahrheit nächste? | **Nein, nicht bei Turnier-Tabellen:** Kemeny minus Borda im Abstand zur Wahrheit +0,29 ± 0,06 (15 Bewerter, gepaart); Kemeny besser in 19 %, schlechter in 40 % der Profile. | `test_kemeny_is_farther_from_the_truth_than_borda_under_tournament_noise` |
| Und unter Mallows-Rauschen, für das Kemeny der Maximum-Likelihood-Schätzer sein soll (φ = 0,7)? | **Kein erkennbarer Unterschied:** −0,03 ± 0,08 (15 Bewerter); bei 5 und 9 Bewertern ebenfalls innerhalb von zwei Standardfehlern. | `test_under_mallows_noise_the_two_are_indistinguishable` |
| Wie oft läuft die Mehrheit im Kreis? | Bei 8 Teilnehmern hat 40 % der Profile mit 3 Bewertern irgendwo einen Mehrheitszyklus, 15 % mit 15 Bewertern; ein **Mehrheitssieger fehlt aber nur in 9 % bzw. 3 %**. Bei zufälligen Rangfolgen von 3 Teilnehmern: **12 von 216 Profilen (5,6 %) mit 3 Bewertern**, rund 9 % mit 201. | `test_cycles_among_eight_participants_and_the_rarer_missing_winner`, `test_random_rankings_of_three_candidates_approach_the_limit` |
| Wie oft ist Kemeny nicht eindeutig? | In 23 % (3 Bewerter) bis 9 % (15 Bewerter) der Profile gibt es mehrere gleich gute Rangfolgen. | `test_kemeny_is_often_not_unique_and_borda_often_not_cost_optimal` |
| Ist Borda als Zusammenfassung kostenoptimal? | In 67 % bis 76 % der Profile **nicht**: Borda liegt der Wahrheit näher, hat aber höhere Gesamtkosten zu den Eingaben. | `test_kemeny_is_often_not_unique_and_borda_often_not_cost_optimal` |
| Wie nah kommt die Footrule-Zuordnung an Kemeny? | Kostenverhältnis im Mittel 1,10 (3 Bewerter) bis 1,02 (15), größter Wert 1,47, unter der Garantie 2 (aus K ≤ F ≤ 2K). | `test_footrule_assignment_stays_close_to_the_kemeny_cost_and_below_the_guarantee` |
| Was kostet das exakte Kemeny? | Der Aufwand der Teilmengen-DP wächst mit 2ⁿ · n (49 152 Schritte bei n = 12 gegen 384 bei n = 6); die Rechenzeit hängt vom Rechner ab und steht in der App. | `test_time_kemeny_returns_one_row_per_size_and_growing_state_counts` |

## Befunde und Korrekturen gegenüber dem Plan

- **Erste Messung verzerrt.** Der Kemeny-DP löste Gleichstände nach Index, und der Index korreliert hier mit der Stärke: Das hätte Kemeny scheinbar
  benachteiligt. Seitdem läuft er nach zufälliger Umbenennung der Teilnehmer; der Befund „Borda näher“ blieb bestehen.
- **„Mehrheitszyklus“ und „kein Mehrheitssieger“ sind verschiedene Dinge.** Die Vorab-Messreihe nannte nur den Zyklus (46 % bei 3 Bewertern, 14 % bei 15);
  ein Mehrheitssieger fehlt in viel weniger Profilen. Die Demo nennt beide Zahlen.
- **Zahlen aus der Vorab-Messreihe (Phase 0)** wichen im Rauschen ab (z. B. Borda 1,38 statt 1,47 bei 15 Bewertern): andere feste Profile.

## Abgrenzung

- [Ungarische Methode](https://sebastianhanisch-hungarian-demo.streamlit.app/) (Matching) behandelt die Zuordnung als Verfahren. Hier wird sie
  **angewendet**, um die footrule-optimale Aggregation zu berechnen; kein Nachbau.
- [Siegtabelle](https://sebastianhanisch-siegtabelle-demo.streamlit.app/) und
  [Rangvergleiche](https://sebastianhanisch-rangdistanz-demo.streamlit.app/) liefern die Tabelle aus einem Turnier und die Abstandsmaße, auf denen
  dieses Stück aufbaut.
- [Dynamische Programmierung](https://sebastianhanisch-dynamic-programming-demo.streamlit.app/) (Exakte Suche) zeigt die Teilmengen-/Zustandsrekursion am
  Rucksack; hier dasselbe Muster für eine lineare Ordnung.

## Ehrliche Grenzen

- **Der Befund „Borda näher an der Wahrheit“ hängt am Rauschmodell** (Turnier-Tabellen); unter Mallows sind beide gleichauf. Eine Allgemeinaussage wäre
  falsch.
- **Kleine Instanzen:** n ≤ 12 wegen der exakten DP, m ≤ 15; ein Stärkeabstand.
- **Die „wahre Ordnung“ ist eine Konstruktion** des Generators; echte Bewerter haben sie nicht.
- **Gleichstände** bei Borda, Copeland und Kemeny werden zufällig aufgelöst; das ändert Einzelergebnisse.
- **Alle Bewerter sind gleich gut** und ehrlich; ungleich gute oder manipulierende Bewerter sind nicht Teil dieses Stücks.

## Quellen

- Dwork, C., Kumar, R., Naor, M., Sivakumar, D. (2001): Rank aggregation methods for the web. *WWW10*, 613–622 (DOI 10.1145/371920.372165). Die
  Zuordnungsformulierung der Footrule-Aggregation nach der gängigen Darstellung dieser Arbeit; das Original wurde im Wortlaut nicht eingesehen.
- Young, H. P., Levenglick, A. (1978): A consistent extension of Condorcet's election principle. *SIAM Journal on Applied Mathematics* 35 (Seitenzahlen nicht
  geprüft).
- Kemeny, J. G. (1959): Mathematics without numbers. *Daedalus* 88, 577–591 (Angabe nicht geprüft).
- Mallows, C. L. (1957): Non-null ranking models. I. *Biometrika* 44 (1/2), 114–130.
- Doignon, J.-P., Pekeč, A., Regenwetter, M. (2004): The repeated insertion model for rankings: Missing link between two subset choice models.
  *Psychometrika* 69 (1), 33–54 (DOI 10.1007/BF02295838).
- Diaconis, P., Graham, R. L. (1977): Spearman's footrule as a measure of disarray. *JRSS B* 39 (2), 262–268.
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472.

## Tests

85 Tests, rund 30 Sekunden: Einheiten gegen die von Hand gerechneten Mini-Instanzen (`test_aggregate.py`, `test_noise.py`), Generator (`test_generator.py`,
wortgleich mit der Wurzel), **Orakel** (`test_oracle_rangaggregation.py`: Kemeny gegen die Aufzählung aller Ordnungen, Footrule gegen die Aufzählung aller
Zuordnungen, Kostenverhältnis höchstens 2, Borda und Copeland gegen Neuberechnung, Mehrheitszyklen gegen die Aufzählung, exakt 12/216, Symmetrie),
Mallows-Sampler gegen die exakte Verteilung und die geschlossene Formel, Presets, ausblendbare Regler, Analyse und Verdikte (`test_evaluation.py`),
alle Zahlen dieses README (`test_claims.py`) und AppTest-Rauchtests.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie) |
| `rga_aggregate.py` | Borda, Copeland, Kemeny (DP), Footrule-Zuordnung, Mehrheitsmatrix, Zyklen |
| `rga_noise.py` | Turnier-Tabellen, Mallows-Sampler, Zufallsprofile, gemeinsame Tabelle |
| `rga_scenario.py` | Reglerwerte (`Settings`) und das Profil daraus |
| `rga_evaluation.py` | Analyse, Experimente über die festen Profile, Verdikte |
| `rga_presets.py`, `rga_constants.py` | Permalink, Presets, ausblendbare Regler, Grenzen |
| `rga_visualization.py`, `rga_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Ungleich gute oder manipulierende Bewerter:** eigenes Stück der Linie („Die Bewerter sind unsicher“).
- **Modelle für die Stärke** (Bradley-Terry, Elo): eigene Stücke der Linie; hier nur die Gegenprobe „alle Partien in eine Tabelle“.
- **Weitere Aggregationsregeln** (Schulze, Ranked Pairs, Young): nicht umgesetzt.
- **Mehr als 12 Teilnehmer:** das exakte Kemeny wäre zu langsam; Näherungen (lokale Verbesserung) nicht umgesetzt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, scipy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html).
