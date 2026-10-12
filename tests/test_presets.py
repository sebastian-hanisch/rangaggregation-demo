"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt, was seine Beschreibung verspricht."""

import dataclasses

import rga_constants as C
import rga_evaluation as ev
import rga_presets as P
from rga_scenario import Settings


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX, name
        assert p["noise"] in C.NOISE_LABELS, name
        assert C.K_MIN <= p["k"] <= C.K_MAX and C.PHI_MIN <= p["phi"] <= C.PHI_MAX, name
        assert 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    # die Voreinstellung der App ist nicht das erste Preset (das ist der Einzelfall), sondern die Standard-Settings
    assert Settings() == Settings(n=C.DEFAULT_N, m=C.DEFAULT_M, noise="turnier", k=C.DEFAULT_K, phi=C.DEFAULT_PHI, seed=C.DEFAULT_SEED)


def test_setting_specs_cover_every_preset_key_and_settings_fields_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert set(P.PRESET_KEYS) == {f.name for f in dataclasses.fields(Settings)}
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_kept_keys_are_exactly_the_hideable_sliders():
    assert set(P.KEPT_KEYS) == {"k_slider", "phi_slider"}


def test_each_preset_tells_its_story():
    def a(name):
        return ev.analyse(Settings(**C.PRESETS[name]))

    single = a("Ein Bewerter allein")
    assert single.kemeny_cost == 0 and len({tuple(o) for o in single.orders.values()}) == 1
    many = a("Viele ungenaue Bewerter")
    assert many.settings.m == 15 and many.dist_truth["Borda"] < many.single_mean_dist
    cycle = a("Kein Mehrheitssieger")
    assert cycle.winner is None and cycle.cycles >= 1
    mallows = a("Mallows-Rauschen")
    assert mallows.pooled_dist is None and mallows.data.truth == tuple(range(7, -1, -1))
