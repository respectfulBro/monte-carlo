"""Random return generators.

Every generator returns an array of REAL GROSS returns, shape (n_sims, years, n_assets),
where gross = 1 + real return. So 1.07 means +7%.

Four models, in increasing order of "how little we assume":

  lognormal        Returns are i.i.d. lognormal, correlated across assets.
                   Smooth, thin-tailed, no memory. The textbook baseline.
  student_t        Same mean / vol / correlation, but FAT TAILS, and tails that
                   arrive together across assets (multivariate t).
  bootstrap        Resample whole historical YEARS with replacement (i.i.d. draws).
                   Keeps the real shape of returns and the stock/bond correlation.
  block_bootstrap  Resample runs of consecutive years. Also keeps whatever serial
                   dependence (momentum / mean reversion) the history contains.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .config import Asset, Config


# ------------------------------------------------------------------ utilities
def cholesky(corr: np.ndarray) -> np.ndarray:
    """Lower-triangular L with L @ L.T == corr. Turns independent normals into correlated ones."""
    return np.linalg.cholesky(corr)


def _mu_sigma(assets: list[Asset]) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([a.mu for a in assets], float), np.array([a.sigma for a in assets], float))


def lognormal_params(mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(m, s) of log(gross) such that E[gross] = 1+mu and Std[gross] = sigma.

    If X = 1+R is lognormal(m, s^2):
        E[X]   = exp(m + s^2/2)          -> m = ln(1+mu) - s^2/2
        Var[X] = (exp(s^2)-1) E[X]^2     -> s^2 = ln(1 + sigma^2/(1+mu)^2)
    The "- s^2/2" is the convexity correction: it's why the median outcome
    (exp(m)) sits BELOW the mean, and why volatility eats compound growth.
    """
    s2 = np.log1p(sigma**2 / (1.0 + mu) ** 2)
    m = np.log1p(mu) - s2 / 2.0
    return m, np.sqrt(s2)


# ------------------------------------------------------------- parametric models
def gen_lognormal(assets, corr, n_sims, years, rng) -> np.ndarray:
    mu, sigma = _mu_sigma(assets)
    m, s = lognormal_params(mu, sigma)
    L = cholesky(corr)
    z = rng.standard_normal((n_sims, years, len(assets))) @ L.T       # correlated N(0,1)
    return np.exp(m + s * z)


def gen_student_t(assets, corr, df, n_sims, years, rng) -> np.ndarray:
    """Multivariate Student-t, scaled to unit variance, applied to ARITHMETIC returns.

    t = Z / sqrt(W/df) with Z ~ correlated normal and W ~ chi^2(df) SHARED across
    assets in a given year. Sharing W is what creates tail dependence: when a bad
    year hits, it hits every asset's tail at once (correlations spike in crashes).
    Var(t) = df/(df-2), so we multiply by sqrt((df-2)/df) to get unit variance.

    We do NOT exponentiate: exp() of a heavy-tailed variable has infinite mean.
    Instead gross = max(1 + r, 0), i.e. the worst case is a total loss.
    """
    mu, sigma = _mu_sigma(assets)
    L = cholesky(corr)
    z = rng.standard_normal((n_sims, years, len(assets))) @ L.T
    w = rng.chisquare(df, size=(n_sims, years, 1)) / df
    t = z / np.sqrt(w) * np.sqrt((df - 2.0) / df)
    return np.maximum(1.0 + mu + sigma * t, 0.0)


# ------------------------------------------------------------ historical models
@lru_cache(maxsize=8)
def _load_real_history(path: str, names: tuple[str, ...]) -> np.ndarray:
    """(T, k) matrix of historical REAL returns, using the file's own inflation column.

    Deflating each year with that same year's CPI keeps inflation and asset returns
    jointly sampled -- e.g. the 1970s bring high inflation AND poor real returns.
    """
    df = pd.read_csv(path)
    missing = [c for c in (*names, "inflation") if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; has {list(df.columns)}")
    nominal = df[list(names)].to_numpy(float)
    infl = df[["inflation"]].to_numpy(float)
    return (1.0 + nominal) / (1.0 + infl) - 1.0


def match_moments(hist: np.ndarray, assets: list[Asset]) -> np.ndarray:
    """Affinely rescale each historical column to the target mean / std.

    Keeps everything about SHAPE (skew, fat tails, cross-asset dependence, year
    ordering) but lets you impose forward-looking mean and volatility. This is what
    makes model comparisons fair: same first two moments, different everything else.
    """
    mu, sigma = _mu_sigma(assets)
    z = (hist - hist.mean(axis=0)) / hist.std(axis=0, ddof=1)
    return mu + sigma * z


def get_history(cfg: Config) -> np.ndarray:
    r = cfg.returns
    hist = _load_real_history(str(r.history_path), cfg.asset_names)
    return match_moments(hist, cfg.assets) if r.match_moments else hist


def gen_bootstrap(hist, n_sims, years, rng) -> np.ndarray:
    idx = rng.integers(0, len(hist), size=(n_sims, years))
    return np.maximum(1.0 + hist[idx], 0.0)


def gen_block_bootstrap(hist, n_sims, years, block_len, rng) -> np.ndarray:
    """Circular block bootstrap: paste together random runs of `block_len` consecutive years.

    Wrapping around the end of the sample (modulo T) gives every year equal
    selection probability. block_len=1 reduces exactly to the i.i.d. bootstrap.
    """
    T = len(hist)
    n_blocks = -(-years // block_len)                                   # ceil
    starts = rng.integers(0, T, size=(n_sims, n_blocks))
    offsets = np.arange(block_len)
    idx = (starts[:, :, None] + offsets[None, None, :]) % T
    idx = idx.reshape(n_sims, -1)[:, :years]
    return np.maximum(1.0 + hist[idx], 0.0)


# ---------------------------------------------------------------- dispatcher
def generate_returns(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    n, T, r = cfg.simulation.n_sims, cfg.timeline.years, cfg.returns
    if r.model == "lognormal":
        return gen_lognormal(cfg.assets, cfg.corr_matrix(), n, T, rng)
    if r.model == "student_t":
        return gen_student_t(cfg.assets, cfg.corr_matrix(), r.df, n, T, rng)
    hist = get_history(cfg)
    if r.model == "bootstrap":
        return gen_bootstrap(hist, n, T, rng)
    if r.model == "block_bootstrap":
        return gen_block_bootstrap(hist, n, T, r.block_len, rng)
    raise ValueError(r.model)
