"""Engine tests: the strongest tests compare the simulator to a closed-form answer."""
import numpy as np
import pytest

from retirement_mc.config import override, override_many
from retirement_mc.simulate import simulate


def deterministic(cfg, mu=0.04, **extra):
    """One asset, zero volatility, no fees: every path is identical and analytically solvable."""
    from retirement_mc.config import Asset, Portfolio, Returns
    c = override(cfg, "assets", [Asset("a", mu, 0.0)])
    c = override(c, "portfolio", Portfolio(cfg.portfolio.starting_balance, [1.0], True, 0.0))
    c = override(c, "returns", Returns(model="lognormal", correlation=[[1.0]]))
    return override_many(c, extra) if extra else c


def test_matches_annuity_closed_form(base_cfg):
    """Start-of-year withdrawals S, growth g:  B_T = B0 g^T - S g (g^T - 1)/(g - 1)."""
    c = deterministic(base_cfg, mu=0.04, **{"spending.annual": 30_000, "simulation.n_sims": 10})
    res = simulate(c)
    g, T, B0, S = 1.04, 30, 1_000_000, 30_000
    expected = B0 * g**T - S * g * (g**T - 1) / (g - 1)
    assert expected > 0
    np.testing.assert_allclose(res.balance[:, -1], expected, rtol=1e-9)
    assert res.success.all()


def test_deterministic_ruin_year(base_cfg):
    """At 0% growth and $100k/yr on $1M, money lasts exactly 10 years and year 11 is short."""
    c = deterministic(base_cfg, mu=0.0, **{"spending.annual": 100_000, "simulation.n_sims": 5})
    res = simulate(c)
    assert not res.success.any()
    assert (res.depletion_age == 65 + 10).all()
    assert (res.shortfall[:, 10] == 100_000).all()
    assert (res.shortfall[:, 11:] == 100_000).all()            # keeps failing every later year
    assert (res.balance[:, 11:] == 0).all()


def test_rebalancing_vs_drift_without_flows(base_cfg):
    """No cash flows, sigma=0: rebalanced ends at B0*(sum w g)^T, drifting ends at B0*sum(w g^T)."""
    from retirement_mc.config import Asset
    c = override(base_cfg, "assets", [Asset("a", 0.08, 0.0), Asset("b", 0.01, 0.0)])
    c = override_many(c, {"returns.correlation": [[1, 0], [0, 1]], "portfolio.fee": 0.0,
                          "spending.annual": 0.0, "simulation.n_sims": 4})
    w, g, T, B0 = np.array([0.6, 0.4]), np.array([1.08, 1.01]), 30, 1_000_000
    reb = simulate(override(c, "portfolio.rebalance", True)).balance[:, -1]
    drift = simulate(override(c, "portfolio.rebalance", False)).balance[:, -1]
    np.testing.assert_allclose(reb, B0 * (w @ g) ** T, rtol=1e-9)
    np.testing.assert_allclose(drift, B0 * (w * g**T).sum(), rtol=1e-9)
    assert drift[0] > reb[0]                                    # letting the winner run beats rebalancing here


def test_fees_reduce_growth_exactly(base_cfg):
    c = deterministic(base_cfg, mu=0.05, **{"spending.annual": 0.0, "simulation.n_sims": 3})
    c_fee = override(c, "portfolio.fee", 0.01)
    r = simulate(c_fee).balance[:, -1]
    np.testing.assert_allclose(r, 1_000_000 * (1.05 * 0.99) ** 30, rtol=1e-9)


def test_accumulation_then_retirement_closed_form(base_cfg):
    """10 years of contributions C then 20 years of withdrawals S at constant growth g."""
    c = deterministic(base_cfg, mu=0.03, **{"timeline.current_age": 45, "timeline.retire_age": 55, "timeline.end_age": 75,
                                            "cashflow.annual_contribution": 10_000, "spending.annual": 20_000,
                                            "simulation.n_sims": 3, "portfolio.starting_balance": 100_000})
    g = 1.03
    B = 100_000.0
    for _ in range(10):
        B = (B + 10_000) * g
    for _ in range(20):
        B = max(B - 20_000, 0) * g
    np.testing.assert_allclose(simulate(c).balance[:, -1], B, rtol=1e-9)


def test_social_security_offsets_spending(base_cfg):
    c = deterministic(base_cfg, mu=0.0, **{"spending.annual": 50_000, "cashflow.social_security_age": 75,
                                           "cashflow.social_security_annual": 50_000, "simulation.n_sims": 3,
                                           "portfolio.starting_balance": 500_000})
    res = simulate(c)                                           # 10 yrs x $50k drains $500k; SS then covers the rest
    assert res.success.all()
    np.testing.assert_allclose(res.balance[:, -1], 0.0, atol=1e-6)
    assert (res.spending[:, 10:] == 50_000).all()


def test_invariants_on_random_paths(base_cfg):
    res = simulate(base_cfg)
    assert (res.balance >= 0).all()
    assert np.isfinite(res.balance).all()
    assert (res.shortfall >= 0).all()
    assert (res.spending <= base_cfg.spending.annual + 1e-9).all()
    # with fixed spending and no other income, a path that has run short is at exactly zero forever after
    failed_once = (res.shortfall > 0).cumsum(axis=1) > 0
    assert failed_once.any()
    assert (res.balance[:, 1:][failed_once] == 0).all()


def test_reproducible_and_seed_sensitive(base_cfg):
    a = simulate(base_cfg).balance
    b = simulate(base_cfg).balance
    c = simulate(override(base_cfg, "simulation.seed", 2)).balance
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_success_monotone_in_spending_on_common_paths(base_cfg):
    from retirement_mc.returns import generate_returns
    gross = generate_returns(base_cfg, np.random.default_rng(3))
    ps = [simulate(override(base_cfg, "spending.annual", s), gross=gross).success.mean()
          for s in range(20_000, 80_001, 10_000)]
    assert all(x >= y for x, y in zip(ps, ps[1:]))
    assert ps[0] > ps[-1]


def test_wrong_shape_raises(base_cfg):
    with pytest.raises(ValueError):
        simulate(base_cfg, gross=np.ones((3, 3, 3)))
