"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import rga_constants as C

METHOD_COLORS = {"Borda": C.COLOR_BORDA, "Copeland": C.COLOR_COPELAND, "Kemeny": C.COLOR_KEMENY, "Footrule-optimal": C.COLOR_FOOTRULE}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_profile(a):
    """Plätze jedes Teilnehmers bei jedem Bewerter (Zeilen: Teilnehmer in wahrer Reihenfolge, Spalten: Bewerter); dazu die Ergebnisse der Verfahren."""
    n = a.settings.n
    truth = list(a.data.truth)
    cols = [f"Bewerter {j + 1}" for j in range(len(a.data.profile))] + list(C.METHODS)
    z = []
    for i in truth:
        row = [int(np.where(np.array(o) == i)[0][0]) + 1 for o in a.data.profile]
        row += [int(np.where(np.array(a.orders[mth]) == i)[0][0]) + 1 for mth in C.METHODS]
        z.append(row)
    y = [f"{a.data.names[i]}" for i in truth]
    fig = go.Figure(go.Heatmap(z=z, x=cols, y=y, text=z, texttemplate="%{text}", zmin=1, zmax=n, colorscale="Blues_r", showscale=False,
                               hovertemplate="%{y}, %{x}: Platz %{z}<extra></extra>", xgap=1, ygap=1))
    fig.update_xaxes(type="category", side="top")
    fig.update_yaxes(type="category", autorange="reversed", title_text="Teilnehmer (wahre Ordnung von oben)")
    return _base(fig, height=max(300, 32 * n + 100))


def build_majority(a):
    """Mehrheitsmatrix: Zelle (Zeile i, Spalte j) = Zahl der Bewerter, die i vor j setzen; rot, wenn i die Mehrheit gegen j hat."""
    n, m = a.settings.n, a.settings.m
    order = list(a.data.truth)
    names = [a.data.names[i] for i in order]
    W = a.prefer[np.ix_(order, order)]
    z = np.where(np.eye(n, dtype=bool), np.nan, W / m)
    text = [["" if r == c else str(int(W[r, c])) for c in range(n)] for r in range(n)]
    fig = go.Figure(go.Heatmap(z=z, x=names, y=names, text=text, texttemplate="%{text}", zmin=0, zmax=1, colorscale="RdBu", showscale=False,
                               hovertemplate="%{y} vor %{x}: %{text} von " + str(m) + " Bewertern<extra></extra>", xgap=1, ygap=1))
    fig.update_xaxes(type="category", side="top", title_text="gegen")
    fig.update_yaxes(type="category", autorange="reversed", title_text="Teilnehmer")
    return _base(fig, height=max(300, 34 * n + 100))


def build_methods(rows):
    xs = [str(r["m"]) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["single"] for r in rows], mode="lines+markers", name="einzelne Tabelle (Mittel)",
                             line=dict(color=C.COLOR_SINGLE, width=2, dash="dot"), hovertemplate="%{x} Bewerter: %{y:.2f}<extra></extra>"))
    for name in C.METHODS:
        fig.add_trace(go.Scatter(x=xs, y=[r[name] for r in rows], mode="lines+markers", name=name, line=dict(color=METHOD_COLORS[name], width=3),
                                 hovertemplate=name + ", %{x} Bewerter: %{y:.2f}<extra></extra>"))
    if rows and rows[0].get("pooled") is not None:
        fig.add_trace(go.Scatter(x=xs, y=[r["pooled"] for r in rows], mode="lines+markers", name="alle Partien in einer Tabelle",
                                 line=dict(color=C.COLOR_POOLED, width=3, dash="dash"), marker=dict(symbol="diamond", size=9),
                                 hovertemplate="alle Partien, %{x} Bewerter: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Zahl der Bewerter", type="category")
    fig.update_yaxes(title_text="Kendall-Abstand zur wahren Ordnung", rangemode="tozero")
    return _base(fig, height=400)


def build_pairs(rows):
    labels = [f"{'Turnier' if r['noise'] == 'turnier' else 'Mallows'}, m = {r['m']}" for r in rows]
    colors = [C.COLOR_MAIN if r["noise"] == "turnier" else C.COLOR_FOOTRULE for r in rows]
    fig = go.Figure(go.Bar(x=labels, y=[r["mean"] for r in rows], marker_color=colors, error_y=dict(type="data", array=[2 * r["se"] for r in rows]),
                           hovertemplate="%{x}: Kemeny minus Borda = %{y:.2f} (± 2 SE)<extra></extra>"))
    fig.add_hline(y=0.0, line_color="#333333")
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="Kemeny minus Borda im Abstand zur Wahrheit (über 0: Borda näher)")
    return _base(fig)


def build_cycles(res):
    fig = go.Figure()
    xs = [str(r["m"]) for r in res["turnier"]]
    fig.add_trace(go.Scatter(x=xs, y=[r["cycle"] for r in res["turnier"]], mode="lines+markers", name="8 Teilnehmer, Turnier-Tabellen",
                             line=dict(color=C.COLOR_MAIN, width=3), hovertemplate="%{x} Bewerter: %{y:.0%}<extra></extra>"))
    xr = [str(r["m"]) for r in res["random"]]
    fig.add_trace(go.Scatter(x=xr, y=[r["cycle"] for r in res["random"]], mode="lines+markers", name="3 Teilnehmer, zufällige Rangfolgen",
                             xaxis="x2", line=dict(color=C.COLOR_COPELAND, width=3), hovertemplate="%{x} Bewerter: %{y:.1%}<extra></extra>"))
    fig.update_layout(xaxis=dict(title="Bewerter (Turnier-Tabellen)", type="category", domain=[0, 0.45]),
                      xaxis2=dict(title="Bewerter (zufällige Rangfolgen)", type="category", domain=[0.55, 1.0], anchor="y"),
                      yaxis=dict(title="Anteil der Profile mit Mehrheitszyklus", tickformat=".0%", rangemode="tozero", fixedrange=True))
    fig.update_xaxes(fixedrange=True)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return fig


def build_cost(timing):
    xs = [f"n = {r['n']}" for r in timing]
    fig = go.Figure(go.Bar(x=xs, y=[r["states"] for r in timing], marker_color=C.COLOR_KEMENY,
                           hovertemplate="%{x}: %{y:,} Zustände mal Kandidaten (2ⁿ · n)<extra></extra>"))
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="Rechenschritte der Teilmengen-DP (2ⁿ · n)", type="log")
    return _base(fig)
