"""Command line interface.

    python -m retirement_mc configs/retiree.toml              # base run + figures
    python -m retirement_mc configs/retiree.toml --full       # + convergence, tornado, safe spend, model risk
    python -m retirement_mc configs/retiree.toml --n-sims 50000 --model student_t
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import analysis, plots
from .config import load_config, override
from .simulate import simulate


def _money(x: float) -> str:
    return f"${x:,.0f}"


def print_summary(cfg, s: dict) -> None:
    t, p, sp = cfg.timeline, cfg.portfolio, cfg.spending
    mix = " / ".join(f"{w:.0%} {a.name}" for w, a in zip(p.weights, cfg.assets))
    phase = "retired at start" if t.years_to_retire == 0 else f"{t.years_to_retire} yrs of saving, then retire at {t.retire_age}"
    rule = f"{sp.rule}" + (f" ({_money(sp.annual)}/yr)" if sp.rule != "percent" else f" ({sp.rate:.1%} of balance)")
    print("=" * 66)
    print(f" Retirement Monte Carlo | model: {cfg.returns.model} | {s['n_sims']:,} paths | seed {cfg.simulation.seed}")
    print("=" * 66)
    print(f" Ages {t.current_age} -> {t.end_age} ({phase})")
    print(f" Start {_money(p.starting_balance)} | {mix} | fee {p.fee:.2%} | spending: {rule}")
    print("-" * 66)
    print(f" P(success)                    {s['p_success']:.1%}   (95% CI {s['p_success_lo']:.1%} - {s['p_success_hi']:.1%})")
    if s["p_success"] < 1:
        print(f" Median age money runs out     {s['median_depletion_age']:.0f}   (among failing paths)")
    print(f" P(spending cut > 20%)         {s['p_spending_cut_20pct']:.1%}")
    print(f" Avg spending (median / p10)   {_money(s['median_avg_spend'])} / {_money(s['p10_avg_spend'])}")
    if t.years_to_retire > 0:
        print(f" Balance at retirement (median){_money(s['median_balance_at_retirement']):>12}")
    print(f" Terminal balance p10/50/90    {_money(s['p10_terminal'])} / {_money(s['median_terminal'])} / {_money(s['p90_terminal'])}")
    print("-" * 66)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="retirement_mc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to a TOML config")
    ap.add_argument("--out", default="outputs", help="folder for figures (default: outputs)")
    ap.add_argument("--full", action="store_true", help="also run convergence, sensitivity, safe-spend, model comparison, sequence demo")
    ap.add_argument("--n-sims", type=int, help="override simulation.n_sims")
    ap.add_argument("--seed", type=int, help="override simulation.seed")
    ap.add_argument("--model", choices=["lognormal", "student_t", "bootstrap", "block_bootstrap"], help="override returns.model")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.n_sims:
        cfg = override(cfg, "simulation.n_sims", args.n_sims)
    if args.seed is not None:
        cfg = override(cfg, "simulation.seed", args.seed)
    if args.model:
        cfg = override(cfg, "returns.model", args.model).validate()

    out = Path(args.out)
    res = simulate(cfg)
    s = analysis.summarize(res)
    print_summary(cfg, s)

    files = [plots.fan_chart(res, out / "01_fan_chart.png"),
             plots.paths_plot(res, out / "02_sample_paths.png"),
             plots.terminal_hist(res, out / "03_terminal_wealth.png"),
             plots.depletion_curve(res, out / "04_depletion_curve.png")]
    if cfg.spending.rule != "fixed":
        files.append(plots.spending_fan(res, out / "05_spending_fan.png"))

    if args.full:
        pd.set_option("display.width", 200, "display.max_columns", 20, "display.float_format", "{:,.4f}".format)

        print("\n[Stage 2] Convergence: Monte Carlo noise vs number of paths")
        conv = analysis.convergence(cfg)
        table = analysis.convergence_table(conv)
        print(table.to_string())
        files.append(plots.convergence_plot(conv, table, out / "06_convergence.png"))

        print("\n[Stage 6] Sequence-of-returns risk: same returns, different order")
        seq = analysis.sequence_risk_demo(cfg)
        print(seq.to_string(index=False, formatters={"p_success": "{:.1%}".format, "median_terminal": _money}))
        files.append(plots.sequence_plot(seq, out / "07_sequence_risk.png"))

        print("\n[Stage 5] Model risk: same mean & vol, different return model")
        cmp_ = analysis.compare_models(cfg)
        print(cmp_[["model", "p_success", "p10_terminal", "median_terminal"]].to_string(
            index=False, formatters={"p_success": "{:.1%}".format, "p10_terminal": _money, "median_terminal": _money}))
        files.append(plots.model_comparison_plot(cmp_, out / "08_model_comparison.png"))

        print("\n[Stage 6] Sensitivity: what moves P(success)?")
        tor = analysis.tornado(override(cfg, "simulation.n_sims", max(cfg.simulation.n_sims, 20_000)))
        print(tor[["driver", "pessimistic", "optimistic", "p_pessimistic", "p_optimistic", "swing"]].to_string(
            index=False, formatters={k: "{:.1%}".format for k in ("p_pessimistic", "p_optimistic", "swing")}))
        files.append(plots.tornado_plot(tor, out / "09_tornado.png"))

        if cfg.spending.rule == "fixed":
            print("\n[Stage 6] Max sustainable spending")
            for tgt in (0.95, 0.90, 0.80):
                r = analysis.max_sustainable_spend(cfg, target=tgt)
                print(f" P(success) >= {tgt:.0%}:  {_money(r['spend'])}/yr  ({r['rate_of_start']:.2%} of starting balance)")

    print("\nFigures written:")
    for f in files:
        print("  ", f)
