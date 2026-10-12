"""Invariante der ausblendbaren Regler (DEMO-PLAYBOOK §3, KEPT-Muster): ein nicht gezeichneter Regler bekommt nie einen Wert in seinem Widget-Zustand;
sein zuletzt gewählter Wert wird gemerkt und kommt beim nächsten Zeichnen zurück."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180


def fresh():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def slider_keys(at):
    return {s.key for s in at.slider}


def test_sliders_are_only_drawn_for_their_noise_model():
    at = fresh()
    assert at.selectbox(key="noise_select").value == "turnier"
    assert "k_slider" in slider_keys(at) and "phi_slider" not in slider_keys(at)
    assert "phi_slider" not in at.session_state
    at.selectbox(key="noise_select").select("mallows").run(timeout=TIMEOUT)
    assert "phi_slider" in slider_keys(at) and "k_slider" not in slider_keys(at)
    assert "k_slider" not in at.session_state
    assert not at.exception


def test_hidden_value_survives_switching_the_noise_model_away_and_back():
    at = fresh()
    at.slider(key="k_slider").set_value(6).run(timeout=TIMEOUT)
    at.selectbox(key="noise_select").select("mallows").run(timeout=TIMEOUT)
    assert "k_slider" not in at.session_state
    at.slider(key="phi_slider").set_value(0.5).run(timeout=TIMEOUT)
    at.selectbox(key="noise_select").select("turnier").run(timeout=TIMEOUT)
    assert at.slider(key="k_slider").value == 6 and "phi_slider" not in at.session_state
    at.selectbox(key="noise_select").select("mallows").run(timeout=TIMEOUT)
    assert abs(at.slider(key="phi_slider").value - 0.5) < 1e-9
    assert not at.exception


def test_preset_sets_the_hidden_slider_for_the_next_drawing():
    at = fresh()
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Mallows-Rauschen"]].click().run(timeout=TIMEOUT)
    assert at.selectbox(key="noise_select").value == "mallows"
    assert abs(at.slider(key="phi_slider").value - 0.7) < 1e-9 and at.slider(key="m_slider").value == 9
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Kein Mehrheitssieger"]].click().run(timeout=TIMEOUT)
    assert at.selectbox(key="noise_select").value == "turnier"
    assert at.slider(key="n_slider").value == 4 and at.slider(key="m_slider").value == 3
    assert "phi_slider" not in at.session_state
    assert not at.exception
