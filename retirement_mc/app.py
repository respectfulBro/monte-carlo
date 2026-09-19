"""Streamlit front end for the retirement Monte Carlo simulator.

Run locally:   streamlit run app.py

How it works (the whole pattern in five lines):
  1. Streamlit re-runs this script top to bottom whenever an input changes.
  2. The sidebar widgets are collected into a plain dict -> `config_from_dict` (validated).
  3. The config is serialized to a JSON string, which is the CACHE KEY: same inputs -> instant result.
  4. Cheap things (the main simulation) run automatically; expensive ones (sensitivity, model
     comparison, ...) sit behind buttons so dragging a slider never triggers 15 simulations.
  5. All maths lives in the `retirement_mc` package; this file only draws and wires things up.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from retirement_mc import analysis, plots
from retirement_mc.config import config_from_dict, config_to_dict
from retirement_mc.simulate import simulate

DATA = Path(__file__).resolve().parent / "data" / "us_annual_returns.csv"
SIM_CHOICES = [1_000, 5_000, 10_000, 25_000, 50_000]   # keep the cap modest: shared servers have limited RAM

PRESETS = {   # illustrative assumption sets, NOT forecasts. Real arithmetic returns, % per year.
    "Conservative": dict(s_mu=5.0, s_sig=18.0, b_mu=1.0, b_sig=7.0),
    "Moderate": dict(s_mu=6.0, s_sig=17.0, b_mu=2.0, b_sig=7.0),
    "Optimistic": dict(s_mu=7.0, s_sig=16.0, b_mu=2.5, b_sig=6.0),
}
RULE_LABELS = {"fixed": "Fixed real amount (the '4% rule' style)",
               "percent": "Percent of portfolio",
               "guardrails": "Guardrails (adaptive)"}
MODEL_LABELS = {"lognormal": "Lognormal (classic)", "student_t": "Student-t (fat tails)",
                "bootstrap": "Historical bootstrap", "block_bootstrap": "Historical block bootstrap"}

st.set_page_config(page_title="Retirement Monte Carlo", page_icon="🎲", layout="wide")


# ---------------------------------------------------------------------------- caching
@st.cache_resource(show_spinner="Simulating...", max_entries=2)
def get_result(cfg_json: str):
    """Main simulation. Keyed on the config JSON; cache_resource hands back the same object (no copy)."""
    return simulate(config_from_dict(json.loads(cfg_json)))


@st.cache_data(show_spinner=False, max_entries=8)
def get_tornado(cfg_json: str):
    return analysis.tornado(config_from_dict(json.loads(cfg_json)))


@st.cache_data(show_spinner=False, max_entries=8)
def get_models(cfg_json: str):
    return analysis.compare_models(config_from_dict(json.loads(cfg_json)))


@st.cache_data(show_spinner=False, max_entries=8)
def get_sequence(cfg_json: str):
    return analysis.sequence_risk_demo(config_from_dict(json.loads(cfg_json)))


@st.cache_data(show_spinner=False, max_entries=8)
def get_safe_spend(cfg_json: str, target: float):
    return analysis.max_sustainable_spend(config_from_dict(json.loads(cfg_json)), target=target)


def gated(key: str, label: str, compute, cfg_json: str, *args, help: str | None = None):
    """Run an expensive analysis only when its button is pressed; drop stale results when inputs change."""
    if st.button(label, key=f"btn_{key}", help=help):
        with st.spinner("Running..."):
            st.session_state[f"out_{key}"] = ((cfg_json, args), compute(cfg_json, *args))
    got = st.session_state.get(f"out_{key}")
    if got is None:
        return None
    if got[0] != (cfg_json, args):
        st.info("Your inputs changed since this was run. Press the button again to refresh.")
        return None
    return got[1]


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------------ sidebar inputs
for k, v in PRESETS["Moderate"].items():
    st.session_state.setdefault(k, v)


def apply_preset() -> None:
    p = PRESETS.get(st.session_state.preset)
    if p:                                        # "Custom" leaves the numbers alone
        for k, v in p.items():
            st.session_state[k] = v


def to_custom() -> None:
    st.session_state.preset = "Custom"


def pct_label(p: float) -> str:
    """Whole-percent display that never claims certainty a finite simulation can't support."""
    return "99%+" if p >= 0.995 else "<1%" if p <= 0.005 else f"{p:.0%}"


with st.sidebar:
    st.header("Your plan")
    st.caption("All dollar amounts are in **today's dollars**; returns are **after inflation**.")

    with st.expander("About you", expanded=True):
        current_age = st.number_input("Current age", 18, 90, 65, key="current_age")
        retire_age = st.number_input("Retirement age", 18, 100, 65, key="retire_age")
        end_age = st.number_input("Plan until age", 40, 110, 95, key="end_age")

    with st.expander("Portfolio", expanded=True):
        balance = st.number_input("Current savings ($)", 0, 50_000_000, 1_000_000, step=10_000, key="balance")
        stock_pct = st.slider("Stocks (%)  -  rest in bonds", 0, 100, 60, key="stock_pct")
        fee_pct = st.slider("Annual fees (%)", 0.0, 2.0, 0.30, 0.05, key="fee_pct")
        rebalance = st.checkbox("Rebalance every year", True, key="rebalance")

    with st.expander("Saving & other income"):
        contrib = st.number_input("Annual savings while working ($)", 0, 1_000_000, 0, step=1_000, key="contrib")
        contrib_g = st.slider("Savings growth above inflation (%/yr)", 0.0, 5.0, 0.0, 0.25, key="contrib_g")
        ss_age = st.number_input("Social Security starts at age", 50, 80, 67, key="ss_age")
        ss_amt = st.number_input("Social Security ($/yr)", 0, 200_000, 0, step=1_000, key="ss_amt")

    with st.expander("Spending", expanded=True):
        rule = st.selectbox("Spending rule", list(RULE_LABELS), format_func=RULE_LABELS.get, key="rule")
        spend, rate, floor, ceiling = 40_000, 0.04, None, None
        upper, lower, cut, raise_ = 0.20, 0.20, 0.10, 0.10
        if rule == "percent":
            rate = st.slider("Withdraw this % of the portfolio each year", 1.0, 10.0, 4.0, 0.1, key="rate") / 100
            fl = st.number_input("Minimum spending ($, 0 = none)", 0, 1_000_000, 0, step=1_000, key="floor")
            ce = st.number_input("Maximum spending ($, 0 = none)", 0, 1_000_000, 0, step=1_000, key="ceiling")
            floor, ceiling = (fl or None), (ce or None)
        else:
            spend = st.number_input("Annual spending ($)", 0, 1_000_000, 40_000, step=1_000, key="spend")
            if rule == "guardrails":
                upper = st.slider("Cut if withdrawal rate rises by (%)", 5, 50, 20, key="upper") / 100
                lower = st.slider("Raise if withdrawal rate falls by (%)", 5, 50, 20, key="lower") / 100
                cut = st.slider("Size of each cut (%)", 1, 30, 10, key="cut") / 100
                raise_ = st.slider("Size of each raise (%)", 1, 30, 10, key="raise") / 100

    with st.expander("Market assumptions", expanded=True):
        st.selectbox("Assumption set", [*PRESETS, "Custom"], index=1, key="preset", on_change=apply_preset,
                     help="Illustrative starting points, not forecasts. Edit any number to switch to Custom.")
        c1, c2 = st.columns(2)
        s_mu = c1.number_input("Stock return %", -5.0, 15.0, step=0.25, key="s_mu", on_change=to_custom)
        s_sig = c2.number_input("Stock volatility %", 1.0, 50.0, step=0.5, key="s_sig", on_change=to_custom)
        b_mu = c1.number_input("Bond return %", -5.0, 10.0, step=0.25, key="b_mu", on_change=to_custom)
        b_sig = c2.number_input("Bond volatility %", 0.5, 30.0, step=0.5, key="b_sig", on_change=to_custom)
        corr = st.slider("Stock/bond correlation", -0.5, 0.9, 0.1, 0.05, key="corr")
        model = st.selectbox("Return model", list(MODEL_LABELS), format_func=MODEL_LABELS.get, key="model")
        df_t, block_len, match = 5.0, 5, True
        if model == "student_t":
            df_t = st.slider("Degrees of freedom (lower = fatter tails)", 3.0, 15.0, 5.0, 0.5, key="df_t")
        if model in ("bootstrap", "block_bootstrap"):
            st.caption("Resamples 1928-2025 US history (Damodaran / NYU Stern).")
            match = st.checkbox("Rescale history to the return/volatility above", True, key="match",
                                help="On: keep the shape of history but use your mean and volatility. "
                                     "Off: use raw history, including its very strong stock returns.")
        if model == "block_bootstrap":
            block_len = st.slider("Block length (years)", 2, 10, 5, key="block_len")

    with st.expander("Simulation"):
        n_sims = st.select_slider("Number of simulated futures", SIM_CHOICES, value=10_000, key="n_sims")
        seed = st.number_input("Random seed", 0, 1_000_000, 42, key="seed",
                               help="Same seed + same inputs = identical results.")

# ---------------------------------------------------------------------- build the config
raw = {
    "timeline": {"current_age": int(current_age), "retire_age": int(retire_age), "end_age": int(end_age)},
    "portfolio": {"starting_balance": float(balance), "weights": [stock_pct / 100, 1 - stock_pct / 100],
                  "rebalance": rebalance, "fee": fee_pct / 100},
    "assets": [{"name": "stocks", "mu": s_mu / 100, "sigma": s_sig / 100},
               {"name": "bonds", "mu": b_mu / 100, "sigma": b_sig / 100}],
    "returns": {"model": model, "correlation": [[1.0, corr], [corr, 1.0]], "df": float(df_t),
                "block_len": int(block_len), "history_path": str(DATA), "match_moments": bool(match)},
    "cashflow": {"annual_contribution": float(contrib), "contribution_growth": contrib_g / 100,
                 "social_security_age": int(ss_age), "social_security_annual": float(ss_amt)},
    "spending": {"rule": rule, "annual": float(spend), "rate": float(rate), "floor": floor, "ceiling": ceiling,
                 "upper_band": upper, "lower_band": lower, "cut_pct": cut, "raise_pct": raise_},
    "simulation": {"n_sims": int(n_sims), "seed": int(seed)},
}
try:
    cfg = config_from_dict(raw)
except ValueError as e:
    st.title("🎲 Retirement Monte Carlo")
    st.error(f"Check your inputs: {e}")
    st.stop()
cfg_json = json.dumps(config_to_dict(cfg), sort_keys=True)

# ------------------------------------------------------------------------ main results
res = get_result(cfg_json)
summ = analysis.summarize(res)
t = cfg.timeline
n = summ["n_sims"]
p = summ["p_success"]
mc_err = 1.96 * np.sqrt(p * (1 - p) / n)

st.title("🎲 Retirement Monte Carlo")
st.caption("Simulates thousands of possible market futures for your plan and counts how often the money lasts. "
           "Educational tool - not financial advice.")

left, right = st.columns([1, 2])
with left:
    st.metric("Chance the money lasts", pct_label(p), help=f"95% Monte Carlo interval: {summ['p_success_lo']:.1%} to {summ['p_success_hi']:.1%}")
with right:
    st.markdown(
        f"In **{int(round(p * n)):,} of {n:,}** simulated futures, your plan paid for every year up to age **{t.end_age}**."
        f"  \nSimulation noise is about **±{mc_err * 100:.1f} points**; changing your *assumptions* usually moves the "
        f"answer far more. See the **Sensitivity** and **Model risk** tabs before trusting any single number."
    )

m = st.columns(4)
if t.years_to_retire > 0:
    m[0].metric("Median savings at retirement", f"${summ['median_balance_at_retirement']:,.0f}")
else:
    m[0].metric("Median average spending", f"${summ['median_avg_spend']:,.0f}/yr")
m[1].metric(f"Median balance at {t.end_age}", f"${summ['median_terminal']:,.0f}")
m[2].metric("Bad case (10th pct) balance", f"${summ['p10_terminal']:,.0f}")
m[3].metric("Chance of a >20% spending cut", f"{summ['p_spending_cut_20pct']:.0%}")
if cfg.spending.rule != "fixed":
    st.info("With an adaptive spending rule, the plan rarely 'runs out' - it cuts spending instead. "
            "Judge it by the spending cut metric and the *Spending & shortfalls* tab, not just the success rate.")

tabs = st.tabs(["Outlook", "Spending & shortfalls", "Sensitivity", "Model risk", "Sequence risk", "Safe spending"])

with tabs[0]:
    st.markdown("**The range of outcomes.** Bands show where 90% and 50% of simulated portfolios land each year.")
    show(plots.fan_chart(res, None))
    st.markdown("**Individual futures.** Green paths funded every year; red paths ran short.")
    show(plots.paths_plot(res, None))

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        show(plots.depletion_curve(res, None))
    with c2:
        show(plots.terminal_hist(res, None))
    if cfg.spending.rule != "fixed":
        show(plots.spending_fan(res, None))

with tabs[2]:
    st.markdown("**What matters most?** Each driver is nudged pessimistically and optimistically, one at a time, "
                "on the same random draws. Long bars = assumptions worth thinking hard about.")
    tor = gated("tornado", "Run sensitivity analysis", get_tornado, cfg_json)
    if tor is not None:
        show(plots.tornado_plot(tor, None))
        table = tor[["driver", "pessimistic", "optimistic", "p_pessimistic", "p_optimistic"]].copy()
        table["p_pessimistic"] *= 100
        table["p_optimistic"] *= 100
        st.dataframe(table.rename(columns={"p_pessimistic": "P(success) pessimistic", "p_optimistic": "P(success) optimistic"}),
                     hide_index=True,
                     column_config={"P(success) pessimistic": st.column_config.NumberColumn(format="%.1f%%"),
                                    "P(success) optimistic": st.column_config.NumberColumn(format="%.1f%%")})

with tabs[3]:
    st.markdown("**Does the choice of model change the answer?** Same mean and volatility, different assumptions about "
                "the *shape* of returns (fat tails, memory). The last row uses raw history, which includes an unusually "
                "strong century for US stocks.")
    cmp_ = gated("models", "Compare return models", get_models, cfg_json)
    if cmp_ is not None:
        show(plots.model_comparison_plot(cmp_, None))

with tabs[4]:
    st.markdown("**Same returns, different order.** Each simulated path keeps exactly the same yearly returns but is "
                "re-ordered. These are extreme bounds, not forecasts: they show why *when* bad years arrive matters "
                "as much as how bad they are.")
    seq = gated("sequence", "Run sequence-risk demo", get_sequence, cfg_json)
    if seq is not None:
        show(plots.sequence_plot(seq, None))

with tabs[5]:
    if cfg.spending.rule != "fixed":
        st.info("This solver applies to the *fixed real amount* spending rule. Switch the rule in the sidebar.")
    else:
        target = st.select_slider("Target chance the money lasts", [0.70, 0.80, 0.90, 0.95, 0.99], value=0.90,
                                  format_func=lambda x: f"{x:.0%}", key="target")
        safe = gated("safe", "Find max sustainable spending", get_safe_spend, cfg_json, target)
        if safe is not None:
            a, b = st.columns(2)
            a.metric(f"Max spending for {safe['target']:.0%} success", f"${safe['spend']:,.0f}/yr")
            b.metric("As % of current savings", f"{safe['rate_of_start']:.2%}")

st.divider()
d1, d2, _ = st.columns([1, 1, 3])
d1.download_button("Download plan (JSON)", json.dumps(config_to_dict(cfg), indent=2), "plan.json", "application/json",
                   help="Re-load or share the exact inputs behind this result.")
pct = np.percentile(res.balance, [5, 25, 50, 75, 95], axis=0).T
d2.download_button("Download balances (CSV)",
                   pd.DataFrame(pct, index=res.ages, columns=["p5", "p25", "median", "p75", "p95"]).rename_axis("age").round(0).to_csv(),
                   "balance_percentiles.csv", "text/csv")

with st.expander("How this works & what it leaves out"):
    st.markdown("""
* Each simulated future draws a random return for every year, applies your saving/spending plan, and checks whether
  the money ever runs short. The percentage shown is the share of futures that never do.
* Everything is in **today's dollars**; inflation is folded into the real returns.
* **Not modelled:** taxes, how long you'll actually live, changing asset mix over time, inflation shocks beyond what is
  in the return assumptions, and anything about your personal circumstances.
* Historical data: US stocks (S&P 500 incl. dividends), 10-year Treasuries, and CPI, 1928-2025 (Damodaran, NYU Stern).
  The US 20th century was an unusually good sample, so treat raw-history results as optimistic.
* This is an educational simulation, not financial advice. Results depend entirely on the assumptions you enter.
""")
