"""Borda, Copeland, Kemeny: viele Rangfolgen zu einer - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Drittes Stück der Ranking-Linie: m Bewerter liefern je eine Rangfolge, vier Verfahren führen sie zusammen, vier Experimente messen, was das bringt.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import rga_aggregate as AG
import rga_constants as C
import rga_evaluation as ev
from rga_pdf_export import build_pdf
from rga_presets import (
    apply_preset,
    bounds,
    current_value,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    remember,
    seed_widget,
    sync_query_params,
)
from rga_scenario import Settings
from rga_visualization import build_cost, build_cycles, build_majority, build_methods, build_pairs, build_profile

st.set_page_config(page_title="Viele Rangfolgen zu einer – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Profile …")
def _methods():
    return ev.sweep_methods()


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Profile …")
def _pairs():
    return ev.sweep_pairs()


@st.cache_data(show_spinner="Experiment rechnet über 300 feste Profile …")
def _cycles():
    return ev.sweep_cycles()


@st.cache_data(show_spinner=False)
def _quality():
    return ev.sweep_quality()


@st.cache_data(show_spinner=False)
def _timing():
    return ev.time_kemeny()


def verdict(level, text):
    SHOW[level](text)


st.title("🗳️ Borda, Copeland, Kemeny: viele Rangfolgen zu einer")
st.markdown(
    """
Mehrere Bewerter liefern je eine Rangfolge; wie wird daraus eine? Diese Demo ist das **dritte Stück der Ranking-Linie** und die **erste Demo der Reihe
mit mehreren Eingaben**. **Borda** summiert die Plätze, **Copeland** zählt die paarweisen Mehrheiten, **Kemeny** sucht die Rangfolge mit der kleinsten
Summe der Abstände zu den Eingaben, und die **Footrule-Zuordnung** ist seine billige Näherung. Das Ziel ist die Wahl: Die Eingaben **am besten
zusammenzufassen** ist nicht dasselbe, wie **der Wahrheit am nächsten zu kommen**.

Weiter unten: vier **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So führt man viele Rangfolgen zusammen", expanded=True):
    st.markdown(
        """
1. **Eingaben:** m Bewerter, jeder mit einer Rangfolge über dieselben n Teilnehmer. Hier ist jede Rangfolge die Tabelle eines eigenen kleinen
   Turniers (oder eine Zufallsabweichung nach Mallows) um dieselbe wahre Ordnung.
2. **Borda:** Jeder Platz zählt als Punktzahl; wer die kleinste Summe hat, steht vorn. Einfach, nutzt die ganze Rangfolge.
3. **Copeland:** Für jedes Paar entscheidet die Mehrheit der Bewerter; gezählt werden die Paar-Siege. Kann im Kreis laufen (Condorcet-Paradoxon).
4. **Kemeny:** Die Rangfolge, die die Summe der Kendall-Distanzen zu allen Eingaben minimiert. Exakt aber schwer (der Aufwand wächst mit 2ⁿ).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Bewerter**")
    n = st.slider("Teilnehmer n", *bounds("n_slider"), key="n_slider", help="Wie viele Teilnehmer bewertet werden (höchstens 12: das exakte Kemeny rechnet sonst zu lange).")
    m = st.slider("Zahl der Bewerter m", *bounds("m_slider"), key="m_slider", help="Wie viele Rangfolgen zusammengeführt werden.")
    noise = st.selectbox("Wie entstehen die Rangfolgen?", list(C.NOISE_LABELS), format_func=C.NOISE_LABELS.get, key="noise_select")
    if noise == "turnier":
        seed_widget("k_slider")
        k = st.slider("Partien je Paar im Turnier jedes Bewerters", *bounds("k_slider"), key="k_slider",
                      help="Wie oft jedes Paar im kleinen Turnier eines Bewerters spielt. Mehr Partien, genauere Eingaben.")
        remember("k_slider", k)
    else:
        seed_widget("phi_slider")
        phi = st.slider("Mallows-Parameter φ", *bounds("phi_slider"), step=C.PHI_STEP, key="phi_slider",
                        help="Wie stark die Rangfolgen von der wahren Ordnung abweichen: klein = nah dran, nahe 1 = fast zufällig.")
        remember("phi_slider", phi)
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Bewerter würfeln", width="stretch", on_click=randomize_seed)

k_value = int(current_value("k_slider"))
phi_value = round(float(current_value("phi_slider")), 2)
sync_query_params({"n_slider": int(n), "m_slider": int(m), "noise_select": noise, "k_slider": k_value, "phi_slider": phi_value, "seed_input": int(seed)})

settings = Settings(int(n), int(m), noise, k_value, phi_value, int(seed))
a = _analysis(settings)
data = a.data

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Eingaben und Ergebnisse")
st.plotly_chart(build_profile(a), width="stretch", key="profile_chart")
st.caption("Die Zahl in jeder Zelle ist der Platz des Teilnehmers (1 = Spitze). Zeilen: Teilnehmer in der wahren Reihenfolge, Spalten: die Bewerter "
           "und rechts die vier Verfahren. Die wahre Ordnung ist hier bekannt, weil alles simuliert ist.")

rows = []
for method in C.METHODS:
    rows.append({"Verfahren": method, "Rangfolge": " > ".join(data.names[i] for i in a.orders[method]), "Gesamtkosten zu den Eingaben": a.costs[method],
                 "Abstand zur Wahrheit": a.dist_truth[method], "Stärkster auf Platz 1": "ja" if a.top1[method] else "nein"})
rows.append({"Verfahren": "Wahre Ordnung", "Rangfolge": " > ".join(data.names[i] for i in data.truth),
             "Gesamtkosten zu den Eingaben": AG.total_cost(data.truth, data.profile), "Abstand zur Wahrheit": 0, "Stärkster auf Platz 1": "ja"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Kleinste Gesamtkosten (Kemeny)", a.kemeny_cost, delta=f"{a.kemeny_optima} optimale Rangfolge(n)", delta_color="off",
          help="Kleinste mögliche Summe der Kendall-Distanzen zu den Eingaben.")
c2.metric("Mehrheitssieger", "keiner" if a.winner is None else data.names[a.winner], delta=f"{a.cycles} von {a.triples} Dreiergruppen zyklisch",
          delta_color="off", help="Condorcet-Sieger: schlägt jeden anderen mit strenger Mehrheit.")
c3.metric("Eine Eingabe im Mittel", f"{a.single_mean_dist:.2f}".replace(".", ","), delta="Abstand zur Wahrheit", delta_color="off")
c4.metric("Alle Partien in einer Tabelle", "–" if a.pooled_dist is None else a.pooled_dist, delta="Abstand zur Wahrheit (nur Turnier)", delta_color="off",
          help="Referenz: die Siege aller Bewerter-Turniere summiert, statt Rangfolgen zusammenzuführen.")
st.download_button("📄 PDF der aktuellen Konfiguration", data=build_pdf(a), file_name="rangaggregation.pdf", mime="application/pdf", key="pdf_download")

st.markdown("**Paarweise Mehrheiten** (Zeile i gegen Spalte j: Zahl der Bewerter, die i vor j setzen)")
st.plotly_chart(build_majority(a), width="stretch", key="majority_chart")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Welches Verfahren führt zusammen, und gibt es einen Mehrheitssieger?")
verdict(*ev.verdict_analysis(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Wer führt am besten zusammen?")
st.caption("8 Teilnehmer (feste Instanz, unabhängig vom Regler), jeder Bewerter mit einem eigenen kleinen Turnier (1 Partie je Paar), Mittel über 300 feste "
           "Profile. Kendall-Abstand zur wahren Ordnung, höchstens 28.")
rows1 = _methods()
st.plotly_chart(build_methods(rows1), width="stretch", key="exp1_chart")
verdict(*ev.verdict_methods(rows1))

st.subheader("🔬 Experiment 2: Zusammenfassen oder schätzen?")
st.caption("Gepaarter Vergleich auf denselben Profilen: Kemeny minus Borda im Abstand zur Wahrheit (über 0: Borda näher), Fehlerbalken ± 2 Standardfehler. "
           "Zwei Rauschmodelle: Turnier-Tabellen und Mallows (φ = 0,7).")
rows2 = _pairs()
st.plotly_chart(build_pairs(rows2), width="stretch", key="exp2_chart")
verdict(*ev.verdict_pairs(rows2))

st.subheader("🔬 Experiment 3: Wann gibt es keinen Mehrheitssieger?")
st.caption("Links: 8 Teilnehmer, Turnier-Tabellen. Rechts: 3 Teilnehmer, jede der 6 Rangfolgen gleich wahrscheinlich (3000 Ziehungen je Punkt).")
res3 = _cycles()
st.plotly_chart(build_cycles(res3), width="stretch", key="exp3_chart")
verdict(*ev.verdict_cycles(res3))

st.subheader("🔬 Experiment 4: Was kostet das Exakte?")
st.caption("Links: Rechenaufwand der Teilmengen-DP gegen n. Darunter: wie nah die Footrule-Zuordnung an die Kemeny-Kosten kommt und wie oft Kemeny nicht "
           "eindeutig ist.")
quality, timing = _quality(), _timing()
st.plotly_chart(build_cost(timing), width="stretch", key="exp4_chart")
verdict(*ev.verdict_cost(quality, timing))
st.dataframe(pd.DataFrame([{"Bewerter": r["m"], "Kemeny nicht eindeutig": f"{ev.pct(r['nonunique'])} %",
                            "Borda nicht kostenoptimal": f"{ev.pct(r['borda_not_optimal'])} %",
                            "Footrule/Kemeny Kosten (Mittel)": round(r["ratio_mean"], 3), "(größter Wert)": round(r["ratio_max"], 3)} for r in quality]),
             hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Alle Bewerter sind gleich gut** | Hier haben alle dieselbe Genauigkeit. Mit ungleich guten oder manipulierenden Bewertern führen Borda, Copeland und Kemeny zu verzerrten Ergebnissen. | Die Bewerter sind unsicher |
| **Eine Rangfolge ist das Ziel** | Ein Mehrheitszyklus (Condorcet) zeigt, dass es keine konsistente Mehrheitsordnung geben muss; jedes Verfahren löst das anders auf. | – |
| **„Näher an der Wahrheit“ gilt allgemein** | Der Vorsprung von Borda vor Kemeny ist an das Rauschmodell gebunden: Unter Mallows-Rauschen hat Borda keinen Vorsprung (Kemeny ist hier gleichauf oder leicht besser). | – |
| **Die Eingaben sind alles, was es gibt** | Wer die Rohdaten (alle Partien) hat, sollte sie nicht in Rangfolgen pressen: Die gemeinsame Tabelle liegt näher an der Wahrheit. | Bradley-Terry |
| **Das exakte Kemeny ist machbar** | Die Teilmengen-DP wächst mit 2ⁿ; die Demo geht bis 12 Teilnehmer. Das Problem ist NP-schwer. | – |
| **Alles ist simuliert** | Die „wahre Ordnung“ ist eine Konstruktion des Generators, kein Messwert. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Profil.** $m$ Rangfolgen $\pi_1, \dots, \pi_m$ über $n$ Teilnehmer; $w_{ij}$ ist die Zahl der Bewerter, die $i$ vor $j$ setzen.

**Borda.** Punktzahl $b_i = \sum_v \pi_v(i)$ (Platz, 0 = bester); es gewinnt die kleinste Summe.

**Copeland.** $c_i = \#\{j : w_{ij} > w_{ji}\} + \tfrac12 \#\{j \ne i : w_{ij} = w_{ji}\}$; es gewinnt die größte Zahl.

**Kemeny.** $\pi^\star \in \arg\min_\pi \sum_v K(\pi, \pi_v)$ mit der Kendall-Distanz $K$; gleichbedeutend mit der Minimierung von
$\sum_{i \text{ vor } j \text{ in } \pi} w_{ji}$. Teilmengen-DP: $f(S \cup \{x\}) = \min_x \big( f(S) + \sum_{s \in S} w_{xs} \big)$ über alle $2^n$ Mengen.

**Footrule-optimal.** $\min_\pi \sum_v F(\pi, \pi_v)$ ist eine Zuordnung von Teilnehmern auf Plätze mit Kosten $c_{ij} = \sum_v |\pi_v(i) - j|$;
wegen $K \le F \le 2K$ ist ihre Kemeny-Kostenabweichung höchstens ein Faktor 2.

**Condorcet-Zyklus.** Mehrheitsmatrix $B_{ij} = [w_{ij} > w_{ji}]$; die Zahl zyklischer Dreiergruppen ist $\mathrm{tr}(B^3)/3$.

**Mallows-Modell.** $P(\pi) \propto \varphi^{K(\pi, \pi_0)}$ um die Referenz $\pi_0$; Ziehung per wiederholtem Einfügen.

**Literatur.**
- Dwork, C., Kumar, R., Naor, M., Sivakumar, D. (2001): Rank aggregation methods for the web. *Proceedings of the 10th International World Wide Web Conference
  (WWW10)*, 613–622 (DOI 10.1145/371920.372165). Die
  Zuordnungsformulierung der Footrule-Aggregation (Proposition 4, Abschnitt 4.2) und die Schranke $K \le F \le 2K$ samt 2-Näherung (Propositionen 1 und 2,
  Abschnitt 2.1.2) im Original eingesehen 2026-10-17 (PDF-Volltext, WWW10-Archiv).
- Young, H. P., Levenglick, A. (1978): A consistent extension of Condorcet's election principle. *SIAM Journal on Applied Mathematics* 35, 285–300 (Kemeny-Regel als
  einzige konsistente, neutrale Condorcet-Regel; Seitenzahlen nicht geprüft).
- Kemeny, J. G. (1959): Mathematics without numbers. *Daedalus* 88, 577–591 (Angabe nicht geprüft).
- Mallows, C. L. (1957): Non-null ranking models. I. *Biometrika* 44 (1/2), 114–130.
- Doignon, J.-P., Pekeč, A., Regenwetter, M. (2004): The repeated insertion model for rankings: Missing link between two subset choice models.
  *Psychometrika* 69 (1), 33–54.
- Diaconis, P., Graham, R. L. (1977): Spearman's footrule as a measure of disarray. *Journal of the Royal Statistical Society, Series B* 39 (2), 262–268.
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html)."
)
