"""Matplotlib figures. Each function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from .simulate import Result

BLUE, RED, GREY, GREEN = "#2b6cb0", "#c53030", "#718096", "#2f855a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 130, "font.size": 10})
_usd = FuncFormatter(lambda v, _: f"${v/1e6:.1f}M" if abs(v) >= 1e6 else f"${v/1e3:.0f}k")
_pct = FuncFormatter(lambda v, _: f"{v:.0%}")


def _save(fig, path):
    """Save to `path` and return the path -- or, when path is None, return the Figure itself
    (the web app renders it with st.pyplot and closes it)."""
    if path is None:
        fig.tight_layout()
        return fig
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def fan_chart(res: Result, path) -> str:
    ages, t = res.ages, res.cfg.timeline
    q = np.percentile(res.balance, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(ages, q[0], q[4], color=BLUE, alpha=0.15, label="5th-95th pct")
    ax.fill_between(ages, q[1], q[3], color=BLUE, alpha=0.30, label="25th-75th pct")
    ax.plot(ages, q[2], color=BLUE, lw=2, label="Median")
    if t.retire_age > t.current_age:
        ax.axvline(t.retire_age, color=GREY, ls="--", lw=1)
        ax.text(t.retire_age, ax.get_ylim()[1] * 0.97, " retire", color=GREY, va="top")
    ax.set_ylim(0, q[4].max() * 1.05)
    ax.yaxis.set_major_formatter(_usd)
    ax.set_xlabel("Age"); ax.set_ylabel("Portfolio (today's dollars)")
    ax.set_title(f"Portfolio fan chart  |  P(success) = {res.success.mean():.1%}")
    ax.legend(loc="upper left", frameon=False)
    return _save(fig, path)


def paths_plot(res: Result, path, n_paths: int = 40) -> str:
    rng = np.random.default_rng(0)
    idx = rng.choice(res.balance.shape[0], size=min(n_paths, res.balance.shape[0]), replace=False)
    ok = res.success
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in idx:
        ax.plot(res.ages, res.balance[i], color=GREEN if ok[i] else RED, alpha=0.55, lw=1)
    ax.plot(res.ages, np.median(res.balance, axis=0), color="black", lw=2, label="Median of all paths")
    ax.set_ylim(0, np.percentile(res.balance, 95, axis=0).max() * 1.05)
    ax.yaxis.set_major_formatter(_usd)
    ax.set_xlabel("Age"); ax.set_ylabel("Portfolio (today's dollars)")
    ax.set_title(f"{len(idx)} individual futures  (green = funded to the end, red = ran short)")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, path)


def terminal_hist(res: Result, path) -> str:
    term = res.balance[:, -1]
    cap = np.percentile(term, 97)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(np.clip(term, 0, cap), bins=60, color=BLUE, alpha=0.8)
    ax.axvline(np.median(term), color="black", ls="--", label=f"Median {np.median(term)/1e6:.2f}M")
    ax.xaxis.set_major_formatter(_usd)
    ax.set_xlabel(f"Balance at age {res.cfg.timeline.end_age} (today's $; capped at 97th pct)")
    ax.set_ylabel("Paths")
    ax.set_title(f"Terminal wealth  |  {(1-res.success.mean()):.1%} of paths ran out")
    ax.legend(frameon=False)
    return _save(fig, path)


def depletion_curve(res: Result, path) -> str:
    failed_by = (np.cumsum(res.shortfall > 0, axis=1) > 0).mean(axis=0)
    ages = res.ages[:-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ages, failed_by, color=RED, lw=2)
    ax.fill_between(ages, failed_by, color=RED, alpha=0.15)
    ax.yaxis.set_major_formatter(_pct)
    ax.set_xlabel("Age"); ax.set_ylabel("Cumulative probability of a shortfall")
    ax.set_title("When does the money run out?")
    return _save(fig, path)


def spending_fan(res: Result, path) -> str:
    r0 = res.cfg.timeline.years_to_retire
    ages = res.ages[:-1][r0:]
    sp = res.spending[:, r0:]
    q = np.percentile(sp, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(ages, q[0], q[4], color=GREEN, alpha=0.15, label="5th-95th pct")
    ax.fill_between(ages, q[1], q[3], color=GREEN, alpha=0.30, label="25th-75th pct")
    ax.plot(ages, q[2], color=GREEN, lw=2, label="Median")
    ax.set_ylim(0, q[4].max() * 1.1)
    ax.yaxis.set_major_formatter(_usd)
    ax.set_xlabel("Age"); ax.set_ylabel("Spending (today's dollars / yr)")
    ax.set_title(f"Spending path under rule '{res.cfg.spending.rule}'")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, path)


def convergence_plot(df, table, path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df["n_sims"], df["p_success"], s=10, color=BLUE, alpha=0.35, label="Individual estimates")
    ax.set_xscale("log")
    ax.axhline(df["p_success"].mean(), color=GREY, ls="--", lw=1)
    ax.yaxis.set_major_formatter(_pct)
    ax.set_xlabel("Number of simulated paths"); ax.set_ylabel("Estimated P(success)")
    ax2 = ax.twinx()
    ax2.plot(table.index, table["empirical_se"], "o-", color=RED, label="Empirical std error")
    ax2.plot(table.index, table["theoretical_se"], "k--", label=r"Theory $\sqrt{p(1-p)/n}$")
    ax2.set_yscale("log"); ax2.set_ylabel("Standard error of P(success)"); ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    fmt = FuncFormatter(lambda v, _: f"{v:.1%}")
    ax2.yaxis.set_major_formatter(fmt); ax2.yaxis.set_minor_formatter(fmt)
    ax2.tick_params(axis="y", which="minor", labelsize=7)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=True, framealpha=0.9, loc="upper right")
    ax.set_title("Monte Carlo noise shrinks like 1/sqrt(n)")
    return _save(fig, path)


def tornado_plot(df, path) -> str:
    d = df.iloc[::-1].reset_index(drop=True)
    base = d["base"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, r in d.iterrows():
        ax.barh(i, r["p_pessimistic"] - base, left=base, color=RED, alpha=0.8)
        ax.barh(i, r["p_optimistic"] - base, left=base, color=GREEN, alpha=0.8)
        ax.text(min(r["p_pessimistic"], r["p_optimistic"]) - 0.004, i, r["pessimistic"] if r["p_pessimistic"] < r["p_optimistic"] else r["optimistic"],
                ha="right", va="center", fontsize=8, color=GREY)
        ax.text(max(r["p_pessimistic"], r["p_optimistic"]) + 0.004, i, r["optimistic"] if r["p_optimistic"] >= r["p_pessimistic"] else r["pessimistic"],
                ha="left", va="center", fontsize=8, color=GREY)
    ax.axvline(base, color="black", lw=1)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["driver"])
    ax.xaxis.set_major_formatter(_pct)
    lo = min(d["p_pessimistic"].min(), d["p_optimistic"].min()); hi = max(d["p_pessimistic"].max(), d["p_optimistic"].max())
    ax.set_xlim(lo - 0.07, min(hi + 0.07, 1.02))
    ax.set_xlabel("P(success)")
    ax.set_title(f"What moves the answer?  (base = {base:.1%})")
    ax.grid(axis="y", alpha=0)
    return _save(fig, path)


def model_comparison_plot(df, path) -> str:
    d = df.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(d["model"], d["p_success"], color=BLUE, alpha=0.85,
            xerr=[d["p_success"] - d["ci_lo"], d["ci_hi"] - d["p_success"]], capsize=3)
    for i, v in enumerate(d["p_success"]):
        ax.text(v + 0.008, i, f"{v:.1%}", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(_pct)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("P(success)  (bars = 95% Monte Carlo CI)")
    ax.set_title("Same mean and volatility, different return models")
    ax.grid(axis="y", alpha=0)
    return _save(fig, path)


def sequence_plot(df, path) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [BLUE, RED, GREEN]
    ax.bar(df["ordering"], df["p_success"], color=colors, alpha=0.85)
    for i, v in enumerate(df["p_success"]):
        ax.text(i, v + 0.015, f"{v:.1%}", ha="center", fontsize=10)
    ax.yaxis.set_major_formatter(_pct); ax.set_ylim(0, 1.1)
    ax.set_ylabel("P(success)")
    ax.set_title("Same yearly returns, different order")
    ax.grid(axis="x", alpha=0)
    return _save(fig, path)
