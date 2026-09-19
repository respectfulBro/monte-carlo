"""Analysis tools built on top of `simulate`.

    summarize              headline metrics for one Result
    wilson_ci              confidence interval for a success probability
    convergence            how noisy is P(success) as n_sims grows?      (stage 2)
    tornado                which assumptions move the answer most?       (stage 6)
    max_sustainable_spend  root-find the spending that hits a target     (stage 6)
    compare_models         same moments, different return models         (stage 5)
    sequence_risk_demo     identical returns, different ORDER             (stage 6)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, override, override_many
from .returns import generate_returns
from .simulate import Result, simulate


# --------------------------------------------------------------------- basics
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Better than the naive p +/- z*sqrt(p(1-p)/n) when p is near 0 or 1 -- and
    success probabilities in retirement models live near 0 or 1 all the time.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(res: Result) -> dict:
    cfg = res.cfg
    n = res.balance.shape[0]
    ok = res.success
    k = int(ok.sum())
    lo, hi = wilson_ci(k, n)
    term = res.balance[:, -1]
    r0 = cfg.timeline.years_to_retire
    spend = res.spending[:, r0:]
    ref = spend[:, 0] if cfg.spending.rule == "percent" else np.full(n, cfg.spending.annual)
    dep = res.depletion_age
    return {
        "n_sims": n,
        "p_success": k / n,
        "p_success_lo": lo,
        "p_success_hi": hi,
        "p_spending_cut_20pct": float((spend.min(axis=1) < 0.8 * ref).mean()),
        "median_avg_spend": float(np.median(spend.mean(axis=1))),
        "p10_avg_spend": float(np.percentile(spend.mean(axis=1), 10)),
        "p10_terminal": float(np.percentile(term, 10)),
        "median_terminal": float(np.median(term)),
        "p90_terminal": float(np.percentile(term, 90)),
        "median_depletion_age": float(np.nanmedian(dep)) if (~ok).any() else float("nan"),
        "median_balance_at_retirement": float(np.median(res.balance[:, r0])),
    }


# ------------------------------------------------------------ stage 2: noise
def convergence(cfg: Config, sizes=(100, 300, 1_000, 3_000, 10_000, 30_000, 100_000),
                n_reps: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Re-estimate P(success) many times at each n_sims, with independent seeds.

    The spread of those estimates is the Monte Carlo standard error. It should track
    sqrt(p(1-p)/n): 100x more paths buys only 10x less noise.

    Estimating a standard deviation from R repetitions is itself noisy (relative error
    ~ 1/sqrt(2(R-1))), so by default small sizes get many repetitions (up to 200) and
    large sizes fewer (down to 20) to keep total work bounded.
    """
    rows = []
    for size in sizes:
        c = override(cfg, "simulation.n_sims", size)
        reps = n_reps if n_reps is not None else int(np.clip(2_000_000 // size, 20, 200))
        for rep in range(reps):
            rng = np.random.default_rng(np.random.SeedSequence([seed, size, rep]))
            rows.append({"n_sims": size, "rep": rep, "p_success": simulate(c, rng).success.mean()})
    return pd.DataFrame(rows)


def convergence_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("n_sims")["p_success"]
    out = pd.DataFrame({"mean": g.mean(), "empirical_se": g.std(ddof=1)})
    p = df["p_success"].mean()
    out["theoretical_se"] = np.sqrt(p * (1 - p) / out.index.to_numpy())
    return out


# ------------------------------------------------------- stage 6: sensitivity
def default_scenarios(cfg: Config) -> list[tuple[str, str, str, Config, Config]]:
    """(name, low_label, high_label, cfg_low, cfg_high) -- 'low'/'high' = pessimistic/optimistic side."""
    def idx(name):
        return cfg.asset_names.index(name) if name in cfg.asset_names else 0
    s, b = idx("stocks"), (idx("bonds") if len(cfg.assets) > 1 else 0)
    t, sp = cfg.timeline, cfg.spending.annual
    out = [
        ("Annual spending", f"+20% (${sp*1.2:,.0f})", f"-20% (${sp*0.8:,.0f})",
         override(cfg, "spending.annual", sp * 1.2), override(cfg, "spending.annual", sp * 0.8)),
        ("Stock real return", f"{cfg.assets[s].mu-0.01:.1%}", f"{cfg.assets[s].mu+0.01:.1%}",
         override(cfg, f"assets.{s}.mu", cfg.assets[s].mu - 0.01), override(cfg, f"assets.{s}.mu", cfg.assets[s].mu + 0.01)),
        ("Stock volatility", f"{cfg.assets[s].sigma+0.03:.0%}", f"{cfg.assets[s].sigma-0.03:.0%}",
         override(cfg, f"assets.{s}.sigma", cfg.assets[s].sigma + 0.03), override(cfg, f"assets.{s}.sigma", cfg.assets[s].sigma - 0.03)),
        ("Bond real return", f"{cfg.assets[b].mu-0.01:.1%}", f"{cfg.assets[b].mu+0.01:.1%}",
         override(cfg, f"assets.{b}.mu", cfg.assets[b].mu - 0.01), override(cfg, f"assets.{b}.mu", cfg.assets[b].mu + 0.01)),
        ("Fees", f"{cfg.portfolio.fee+0.007:.2%}", f"{max(cfg.portfolio.fee-0.003, 0):.2%}",
         override(cfg, "portfolio.fee", cfg.portfolio.fee + 0.007), override(cfg, "portfolio.fee", max(cfg.portfolio.fee - 0.003, 0))),
        ("Plan horizon (end age)", f"{t.end_age+5}", f"{t.end_age-5}",
         override(cfg, "timeline.end_age", t.end_age + 5), override(cfg, "timeline.end_age", t.end_age - 5)),
        ("Starting balance", "-20%", "+20%",
         override(cfg, "portfolio.starting_balance", cfg.portfolio.starting_balance * 0.8),
         override(cfg, "portfolio.starting_balance", cfg.portfolio.starting_balance * 1.2)),
    ]
    return out


def tornado(cfg: Config, scenarios=None) -> pd.DataFrame:
    """One-at-a-time sensitivity. Every run reuses the same seed, so the underlying
    standard-normal draws are identical -- a cheap form of common random numbers."""
    scenarios = scenarios or default_scenarios(cfg)
    base = simulate(cfg).success.mean()
    rows = []
    for name, lo_lbl, hi_lbl, c_lo, c_hi in scenarios:
        p_lo, p_hi = simulate(c_lo).success.mean(), simulate(c_hi).success.mean()
        rows.append({"driver": name, "pessimistic": lo_lbl, "optimistic": hi_lbl,
                     "p_pessimistic": p_lo, "p_optimistic": p_hi, "swing": p_hi - p_lo, "base": base})
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


def max_sustainable_spend(cfg: Config, target: float = 0.90, tol: float = 25.0) -> dict:
    """Largest fixed real spending with P(success) >= target, by bisection.

    Generate the random paths ONCE and reuse them for every trial spend. On fixed
    paths, success is a monotone step function of spending, so bisection is
    well-behaved (with fresh randomness per trial it would jitter and can misfire).
    """
    if cfg.spending.rule != "fixed":
        raise ValueError("max_sustainable_spend only makes sense for rule='fixed'")
    gross = generate_returns(cfg, np.random.default_rng(cfg.simulation.seed))

    def p(spend: float) -> float:
        return simulate(override(cfg, "spending.annual", spend), gross=gross).success.mean()

    lo, hi = 0.0, cfg.portfolio.starting_balance * 0.25
    if p(hi) >= target:
        return {"target": target, "spend": hi, "rate_of_start": hi / cfg.portfolio.starting_balance, "p_success": p(hi)}
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p(mid) >= target else (lo, mid)
    return {"target": target, "spend": lo, "rate_of_start": lo / cfg.portfolio.starting_balance, "p_success": p(lo)}


# -------------------------------------------------- stage 5: model comparison
def compare_models(cfg: Config) -> pd.DataFrame:
    """Same portfolio and same mean/vol, different assumptions about the SHAPE of returns."""
    variants = [
        ("Lognormal (thin tails, no memory)", {"returns.model": "lognormal"}),
        ("Student-t, df=5 (fat tails)", {"returns.model": "student_t", "returns.df": 5.0}),
    ]
    if cfg.returns.history_path:
        variants += [
            ("Historical bootstrap, moment-matched", {"returns.model": "bootstrap", "returns.match_moments": True}),
            (f"Block bootstrap (b={cfg.returns.block_len}), moment-matched",
             {"returns.model": "block_bootstrap", "returns.match_moments": True}),
            (f"Block bootstrap (b={cfg.returns.block_len}), RAW history",
             {"returns.model": "block_bootstrap", "returns.match_moments": False}),
        ]
    rows = []
    for label, changes in variants:
        s = summarize(simulate(override_many(cfg, changes)))
        rows.append({"model": label, "p_success": s["p_success"], "ci_lo": s["p_success_lo"], "ci_hi": s["p_success_hi"],
                     "p10_terminal": s["p10_terminal"], "median_terminal": s["median_terminal"]})
    return pd.DataFrame(rows)


# ------------------------------------------------ sequence-of-returns risk demo
def sequence_risk_demo(cfg: Config) -> pd.DataFrame:
    """Take each simulated path and re-order its years: worst-first, best-first, or as drawn.

    Each path keeps EXACTLY the same set of yearly returns, hence the same compound
    growth rate with no cash flows. Only the order changes -- and with withdrawals,
    order is everything.
    """
    rng = np.random.default_rng(cfg.simulation.seed)
    gross = generate_returns(cfg, rng)
    score = gross @ np.asarray(cfg.portfolio.weights)                 # portfolio return per year
    order = np.argsort(score, axis=1)                                 # ascending: worst first
    worst_first = np.take_along_axis(gross, order[:, :, None], axis=1)
    best_first = np.take_along_axis(gross, order[:, ::-1, None], axis=1)
    rows = []
    for label, g in [("Order as drawn (random)", gross), ("Worst years first", worst_first), ("Best years first", best_first)]:
        r = simulate(cfg, gross=g)
        rows.append({"ordering": label, "p_success": r.success.mean(), "median_terminal": float(np.median(r.balance[:, -1]))})
    return pd.DataFrame(rows)
