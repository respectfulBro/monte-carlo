"""The simulation engine.

Design: vectorize across SIMULATIONS (numpy), loop across YEARS (python).
Years are inherently sequential -- this year's balance depends on last year's -- but
the 10,000 paths are independent, so each year's update is one numpy operation on an
(n_sims, n_assets) array.

Timing convention inside each year t (age = current_age + t):
    1. cash flow at the START of the year:
         working  -> contribution goes in
         retired  -> (spending - social security) comes out
    2. rebalance (optional)
    3. the year's returns are earned, net of fees
A withdrawal larger than the balance is a SHORTFALL: the balance goes to zero and the
unfunded amount is recorded. A path "fails" if it ever has a shortfall.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .returns import generate_returns
from .strategies import make_strategy


@dataclass
class Result:
    cfg: Config
    balance: np.ndarray     # (n_sims, years+1)  real $, balance[:, t] = start of year t; last column = end of plan
    spending: np.ndarray    # (n_sims, years)    real $ actually spent (0 while working)
    shortfall: np.ndarray   # (n_sims, years)    real $ of net spending that could NOT be funded

    @property
    def ages(self) -> np.ndarray:
        t = self.cfg.timeline
        return np.arange(t.current_age, t.end_age + 1)

    @property
    def success(self) -> np.ndarray:
        """True for paths that funded every year of spending."""
        return ~(self.shortfall > 0).any(axis=1)

    @property
    def depletion_age(self) -> np.ndarray:
        """Age at which spending first went unfunded; NaN for paths that never fail."""
        failed = self.shortfall > 0
        first = failed.argmax(axis=1)
        return np.where(failed.any(axis=1), self.cfg.timeline.current_age + first, np.nan)


def simulate(cfg: Config, rng: np.random.Generator | None = None, gross: np.ndarray | None = None) -> Result:
    """Run the model.

    `gross` (n_sims, years, n_assets) can be passed in to reuse the SAME random paths
    across runs. That is the "common random numbers" trick: when you compare two
    settings on identical paths, the difference is signal, not sampling noise.
    """
    t, p, cf = cfg.timeline, cfg.portfolio, cfg.cashflow
    n, T, k = cfg.simulation.n_sims, t.years, cfg.n_assets
    if gross is None:
        rng = rng if rng is not None else np.random.default_rng(cfg.simulation.seed)
        gross = generate_returns(cfg, rng)
    if gross.shape != (n, T, k):
        raise ValueError(f"gross has shape {gross.shape}, expected {(n, T, k)}")

    w = np.asarray(p.weights, float)
    strat = make_strategy(cfg.spending)
    keep = 1.0 - p.fee

    holdings = np.tile(w * p.starting_balance, (n, 1))          # (n, k) dollars in each asset
    balance = np.empty((n, T + 1))
    balance[:, 0] = p.starting_balance
    spending = np.zeros((n, T))
    shortfall = np.zeros((n, T))

    for yr in range(T):
        age = t.current_age + yr
        total = holdings.sum(axis=1)

        # ---- 1. cash flow -------------------------------------------------
        if age < t.retire_age:
            inflow = np.full(n, cf.annual_contribution * (1.0 + cf.contribution_growth) ** yr)
            outflow = np.zeros(n)
        else:
            wanted = strat.spend(total, age - t.retire_age)
            income = cf.social_security_annual if age >= cf.social_security_age else 0.0
            net = wanted - income                                 # >0 : withdraw, <0 : surplus income
            outflow = np.maximum(net, 0.0)
            inflow = np.maximum(-net, 0.0)

        paid = np.minimum(outflow, total)
        short = outflow - paid
        remaining = total - paid
        new_total = remaining + inflow
        if age >= t.retire_age:
            spending[:, yr] = wanted - short
        shortfall[:, yr] = short

        # ---- 2. rebalance or drift ----------------------------------------
        if p.rebalance:
            holdings = new_total[:, None] * w
        else:
            # withdrawals are taken pro-rata; new money buys the target mix
            ratio = np.divide(remaining, total, out=np.zeros_like(total), where=total > 0)
            holdings = holdings * ratio[:, None] + inflow[:, None] * w

        # ---- 3. earn the year's return, net of fees -------------------------
        holdings = holdings * gross[:, yr, :] * keep
        balance[:, yr + 1] = holdings.sum(axis=1)

    return Result(cfg=cfg, balance=balance, spending=spending, shortfall=shortfall)
