import numpy as np
import pytest

from retirement_mc import analysis
from retirement_mc.config import Asset, Config, load_config, override, override_many
from retirement_mc.simulate import simulate
from retirement_mc.strategies import FixedReal, Guardrails, PercentOfPortfolio


def test_percent_rule_never_fails_and_scales_with_balance(base_cfg):
    c = override_many(base_cfg, {"spending.rule": "percent", "spending.rate": 0.04})
    res = simulate(c)
    assert res.success.all()
    s = PercentOfPortfolio(0.04).spend(np.array([1e6, 5e5]), 0)
    np.testing.assert_allclose(s, [40_000, 20_000])
    assert PercentOfPortfolio(0.04, floor=30_000).spend(np.array([1e5]), 0)[0] == 30_000


def test_guardrails_cut_after_crash_and_raise_after_boom():
    g = Guardrails(amount=40_000, upper_band=0.2, lower_band=0.2, cut_pct=0.1, raise_pct=0.1)
    bal = np.array([1_000_000.0, 1_000_000.0, 1_000_000.0])
    assert (g.spend(bal, 0) == 40_000).all()                    # initial rate = 4%
    # year 1: balances -> 700k (rate 5.7% > 4.8%: cut), 1M (unchanged), 1.5M (rate 2.7% < 3.2%: raise)
    out = g.spend(np.array([700_000.0, 1_000_000.0, 1_500_000.0]), 1)
    np.testing.assert_allclose(out, [36_000, 40_000, 44_000])


def test_guardrails_reduce_failures_vs_fixed_at_same_start(base_cfg):
    fixed = simulate(override(base_cfg, "spending.annual", 50_000)).success.mean()
    guard = simulate(override_many(base_cfg, {"spending.rule": "guardrails", "spending.annual": 50_000})).success.mean()
    assert guard > fixed


def test_wilson_ci_properties():
    lo, hi = analysis.wilson_ci(50, 100)
    assert lo < 0.5 < hi
    lo0, hi0 = analysis.wilson_ci(100, 100)
    assert hi0 == pytest.approx(1.0) and lo0 > 0.9                             # unlike the naive interval, not degenerate at p=1
    l1, h1 = analysis.wilson_ci(500, 1000)
    assert (h1 - l1) < (hi - lo)                                # more data, tighter


def test_convergence_matches_binomial_theory(base_cfg):
    df = analysis.convergence(base_cfg, sizes=(500, 5_000), n_reps=40, seed=0)
    tab = analysis.convergence_table(df)
    ratio = tab["empirical_se"] / tab["theoretical_se"]
    assert ((ratio > 0.6) & (ratio < 1.5)).all()
    assert tab.loc[500, "empirical_se"] > 2 * tab.loc[5_000, "empirical_se"]   # ~sqrt(10) = 3.2x


def test_max_sustainable_spend_hits_target(base_cfg):
    r = analysis.max_sustainable_spend(base_cfg, target=0.9)
    assert r["p_success"] >= 0.9
    from retirement_mc.returns import generate_returns
    gross = generate_returns(base_cfg, np.random.default_rng(base_cfg.simulation.seed))
    above = simulate(override(base_cfg, "spending.annual", r["spend"] + 500), gross=gross).success.mean()
    assert above < 0.9 + 1e-9                                   # nudging spending up breaks the target
    assert analysis.max_sustainable_spend(base_cfg, 0.99)["spend"] < r["spend"] < analysis.max_sustainable_spend(base_cfg, 0.7)["spend"]


def test_sequence_risk_ordering(base_cfg):
    d = analysis.sequence_risk_demo(base_cfg).set_index("ordering")["p_success"]
    assert d["Best years first"] > d["Order as drawn (random)"] > d["Worst years first"]
    assert d["Best years first"] > 0.99 and d["Worst years first"] < 0.2


def test_tornado_directions(base_cfg):
    t = analysis.tornado(override(base_cfg, "simulation.n_sims", 4_000)).set_index("driver")
    assert (t["p_optimistic"] >= t["p_pessimistic"]).all()
    assert t["swing"].idxmax() in ("Annual spending", "Plan horizon (end age)", "Starting balance")


def test_model_comparison_runs_and_student_t_is_not_safer(base_cfg):
    d = analysis.compare_models(base_cfg).set_index("model")
    assert len(d) == 5 and d["p_success"].between(0, 1).all()


def test_summary_keys(base_cfg):
    s = analysis.summarize(simulate(base_cfg))
    assert 0 <= s["p_success_lo"] <= s["p_success"] <= s["p_success_hi"] <= 1
    assert s["p10_terminal"] <= s["median_terminal"] <= s["p90_terminal"]


@pytest.mark.parametrize("name", ["retiree", "accumulator", "guardrails"])
def test_shipped_configs_load_and_run(name):
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "configs" / f"{name}.toml"
    cfg = load_config(p)
    cfg = override(cfg, "simulation.n_sims", 500)
    for model in ("lognormal", "student_t", "bootstrap", "block_bootstrap"):
        res = simulate(override(cfg, "returns.model", model))
        assert np.isfinite(res.balance).all()


def test_config_validation_errors(base_cfg):
    with pytest.raises(ValueError):
        override(base_cfg, "portfolio.weights", [0.7, 0.7]).validate()
    with pytest.raises(ValueError):
        override(base_cfg, "returns.correlation", [[1, 1.2], [1.2, 1]]).validate()
    with pytest.raises(ValueError):
        override_many(base_cfg, {"returns.model": "student_t", "returns.df": 2.0}).validate()
    with pytest.raises(ValueError):
        override(base_cfg, "timeline.end_age", 60).validate()
    with pytest.raises(AttributeError):
        override(base_cfg, "spending.nonsense", 1)
