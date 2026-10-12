"""PDF-Export der aktuellen Konfiguration und der Ergebnisse (fpdf2, Kernschrift: nur Zeichen aus Latin-1)."""

from fpdf import FPDF

import rga_constants as C
import rga_evaluation as ev


def _latin(text: str) -> str:
    for old, new in (("–", "-"), ("—", "-"), ("τ", "Tau"), ("φ", "phi"), ("≥", ">="), ("≤", "<="), ("→", "->"), ("**", ""), ("€", "EUR"),
                     ("ⁿ", "^n"), ("−", "-")):
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _latin("Borda, Copeland, Kemeny: viele Rangfolgen zu einer"))
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    noise = C.NOISE_LABELS[s.noise]
    detail = f"Partien je Paar: {s.k}" if s.noise == "turnier" else f"phi: {ev.de(s.phi)}"
    pdf.multi_cell(0, 5.5, _latin(f"Teilnehmer: {s.n}, Bewerter: {s.m}, Rauschmodell: {noise} ({detail}), Seed: {s.seed}."))
    pdf.ln(2)
    level, text = ev.verdict_analysis(a)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernaussage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _latin(text))
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Ergebnisse der Verfahren", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    names = a.data.names
    for method in C.METHODS:
        order = " > ".join(names[i] for i in a.orders[method])
        pdf.multi_cell(0, 5, _latin(f"{method}: {order}  (Gesamtkosten {a.costs[method]}, Abstand zur Wahrheit {a.dist_truth[method]})"),
                       new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 5, _latin("Wahre Ordnung: " + " > ".join(names[i] for i in a.data.truth)), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4.5, _latin(
        "Alles ist simuliert (Generator der Ranking-Linie). Die wahre Ordnung ist eine Konstruktion der Simulation. Kemeny minimiert die Summe der "
        "Kendall-Distanzen zu den Eingaben, nicht den Abstand zur Wahrheit. Quelle: Demo von Sebastian Hanisch, sebastianhanisch.net."))
    return bytes(pdf.output())
