"""Spending rules.

A strategy answers one question each retirement year, for ALL simulated paths at
once: "given each path's current balance, how much do I WANT to spend?"

    spend(balance: (n_sims,), k: int) -> (n_sims,)      # k = years since retirement

Strategies are stateful (guardrails remembers last year's spending per path), so a
fresh instance is created for every simulation run via `make_strategy`.
All amounts are real dollars, so "keep spending constant" = "keep up with inflation".
"""
from __future__ import annotations

import numpy as np

from .config import Spending


class FixedReal:
    """Spend the same inflation-adjusted amount every year, no matter what (the '4% rule' style)."""

    def __init__(self, amount: float):
        self.amount = amount

    def spend(self, balance: np.ndarray, k: int) -> np.ndarray:
        return np.full(balance.shape, self.amount)


class PercentOfPortfolio:
    """Spend a fixed fraction of the current balance. Can never run out -- but spending floats with markets."""

    def __init__(self, rate: float, floor: float | None = None, ceiling: float | None = None):
        self.rate, self.floor, self.ceiling = rate, floor, ceiling

    def spend(self, balance: np.ndarray, k: int) -> np.ndarray:
        s = self.rate * balance
        lo = -np.inf if self.floor is None else self.floor
        hi = np.inf if self.ceiling is None else self.ceiling
        return np.clip(s, lo, hi)


class Guardrails:
    """Simplified Guyton-Klinger guardrails.

    Start at `amount`. Each year compute the current withdrawal rate = last year's
    spending / current balance and compare to the INITIAL rate:
      * rate > initial * (1 + upper_band)  -> portfolio has shrunk: cut spending by cut_pct
      * rate < initial * (1 - lower_band)  -> portfolio has boomed:  raise spending by raise_pct
      * otherwise                          -> keep spending constant in real terms
    Trades some spending volatility for a much lower chance of running out of money.
    """

    def __init__(self, amount, upper_band, lower_band, cut_pct, raise_pct):
        self.amount, self.upper, self.lower = amount, upper_band, lower_band
        self.cut, self.raise_ = cut_pct, raise_pct
        self.prev: np.ndarray | None = None
        self.w0: np.ndarray | None = None

    def spend(self, balance: np.ndarray, k: int) -> np.ndarray:
        bal = np.maximum(balance, 0.0)
        if k == 0:
            self.prev = np.full(balance.shape, self.amount)
            with np.errstate(divide="ignore"):
                self.w0 = np.where(bal > 0, self.amount / np.where(bal > 0, bal, 1.0), np.inf)
            return self.prev.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(bal > 0, self.prev / np.where(bal > 0, bal, 1.0), np.inf)
        cut = rate > self.w0 * (1.0 + self.upper)
        boost = rate < self.w0 * (1.0 - self.lower)
        self.prev = np.where(cut, self.prev * (1 - self.cut), np.where(boost, self.prev * (1 + self.raise_), self.prev))
        return self.prev.copy()


def make_strategy(s: Spending):
    if s.rule == "fixed":
        return FixedReal(s.annual)
    if s.rule == "percent":
        return PercentOfPortfolio(s.rate, s.floor, s.ceiling)
    if s.rule == "guardrails":
        return Guardrails(s.annual, s.upper_band, s.lower_band, s.cut_pct, s.raise_pct)
    raise ValueError(s.rule)
