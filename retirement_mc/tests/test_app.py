"""Headless tests of the Streamlit app (no browser): set widgets, click buttons, inspect output."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


def run(**_) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def headline(at: AppTest) -> str:
    return at.metric[0].value


def test_default_run_renders_without_errors():
    at = run()
    assert at.title[0].value.endswith("Retirement Monte Carlo")
    assert headline(at).endswith("%")
    assert len(at.metric) == 5 and len(at.tabs) == 6
    assert not at.error


def test_app_matches_cli_engine_for_identical_inputs():
    """The web form's defaults equal configs/retiree.toml at n_sims=10,000, so the numbers must agree."""
    from retirement_mc import load_config, override, simulate
    from retirement_mc.analysis import summarize
    cfg = override(load_config(ROOT / "configs" / "retiree.toml"), "simulation.n_sims", 10_000)
    p = summarize(simulate(cfg))["p_success"]
    expected = "99%+" if p >= 0.995 else f"{p:.0%}"
    assert headline(run()) == expected


def test_invalid_inputs_show_message_and_stop_cleanly():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.number_input(key="retire_age").set_value(100).run()      # retire after the plan ends
    assert not at.exception
    assert at.error and "Check your inputs" in at.error[0].value
    assert len(at.metric) == 0


def test_preset_fills_numbers_and_editing_switches_to_custom():
    at = run()
    at.sidebar.selectbox(key="preset").select("Conservative").run()
    assert at.sidebar.number_input(key="s_mu").value == 5.0
    assert at.sidebar.number_input(key="b_mu").value == 1.0
    at.sidebar.number_input(key="s_mu").set_value(4.0).run()
    assert at.sidebar.selectbox(key="preset").value == "Custom"
    assert at.sidebar.number_input(key="s_mu").value == 4.0             # the edit is kept, not overwritten


def test_higher_spending_lowers_success():
    at = run()
    base = float(headline(at).rstrip("%+"))
    at.sidebar.number_input(key="spend").set_value(60_000).run()
    assert not at.exception
    assert float(headline(at).rstrip("%+<")) < base


@pytest.mark.parametrize("model", ["student_t", "bootstrap", "block_bootstrap"])
def test_every_return_model_runs(model):
    at = run()
    at.sidebar.selectbox(key="model").select(model).run()
    assert not at.exception and headline(at).endswith("%")


def test_guardrails_and_percent_rules_run_and_explain_themselves():
    at = run()
    at.sidebar.selectbox(key="rule").select("guardrails").run()
    assert not at.exception and any("adaptive spending rule" in i.value for i in at.info)
    at.sidebar.selectbox(key="rule").select("percent").run()
    assert not at.exception


def test_accumulator_scenario():
    at = run()
    at.sidebar.number_input(key="current_age").set_value(35).run()
    at.sidebar.number_input(key="contrib").set_value(20_000).run()
    assert not at.exception
    assert any("Median savings at retirement" in m.label for m in at.metric)


def test_expensive_analyses_are_gated_and_run_on_click():
    at = run()
    assert len(at.dataframe) == 0                                       # nothing heavy ran on load
    at.button(key="btn_tornado").click().run()
    assert not at.exception and len(at.dataframe) == 1
    at.button(key="btn_models").click().run()
    assert not at.exception
    at.button(key="btn_sequence").click().run()
    assert not at.exception
    at.button(key="btn_safe").click().run()
    assert not at.exception
    assert any("Max spending" in m.label for m in at.metric)


def test_stale_results_are_hidden_when_inputs_change():
    at = run()
    at.button(key="btn_tornado").click().run()
    assert len(at.dataframe) == 1
    at.sidebar.number_input(key="spend").set_value(45_000).run()
    assert len(at.dataframe) == 0
    assert any("inputs changed" in i.value for i in at.info)
