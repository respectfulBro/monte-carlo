# retirement_mc — a Monte Carlo retirement simulator

**Question it answers:** given a portfolio, a savings/spending plan, and uncertain markets,
what is the probability the money lasts — and how does that answer change when you change
the assumptions?

Everything is in **real (inflation-adjusted) dollars and real returns.**

## Quick start

```bash
pip install -r requirements-dev.txt
python -m pytest -q                                            # 46 tests (engine, models, app), ~40 s
streamlit run app.py                                           # the web app
python -m retirement_mc configs/retiree.toml                   # base run + 4 figures
python -m retirement_mc configs/retiree.toml --full            # + every analysis below
python -m retirement_mc configs/accumulator.toml               # 30 yrs saving, then retire
python -m retirement_mc configs/guardrails.toml                # adaptive spending
python -m retirement_mc configs/retiree.toml --model student_t --n-sims 100000
```

## Project map (the seven stages)

| Stage | Concept | Where |
|---|---|---|
| 1 | Vectorized simulation, success probability, fan charts | `simulate.py`, `plots.py` |
| 2 | Monte Carlo error, seeds, validation against closed forms | `analysis.convergence`, `tests/test_engine.py` |
| 3 | Real cash flows: contributions, Social Security, fees, spending rules | `simulate.py`, `strategies.py` |
| 4 | Multi-asset portfolios: correlation via Cholesky, rebalancing | `returns.py`, `simulate.py` |
| 5 | Beyond the normal: fat tails, historical & block bootstrap | `returns.py`, `analysis.compare_models` |
| 6 | Sensitivity, root-finding, sequence risk, common random numbers | `analysis.py` |
| 7 | Packaging: TOML configs, CLI, tests | `config.py`, `cli.py`, `tests/` |

```
retirement_mc/
  config.py       typed dataclasses, TOML loading, validation, `override()` for sweeps
  returns.py      return generators (lognormal, student-t, bootstrap, block bootstrap)
  strategies.py   spending rules (fixed real, % of portfolio, guardrails)
  simulate.py     the engine: vectorized over paths, loop over years
  analysis.py     metrics, convergence, tornado, safe-spend solver, model comparison, sequence demo
  plots.py        figures
  cli.py          command line
app.py            Streamlit web app (sidebar form -> validated config -> package -> charts)
configs/          retiree.toml, accumulator.toml, guardrails.toml
data/             us_annual_returns.csv  (1928-2025, see "Data" below)
tests/            46 tests (engine, return models, analysis, and headless app tests)
docs/             screenshots of the web app
```

## Suggested reading order

1. `returns.lognormal_params` — why the `- s²/2` term exists.
2. `simulate.simulate` — the whole engine is ~50 lines. Read the docstring's timing convention first.
3. `tests/test_engine.py::test_matches_annuity_closed_form` — how we know the engine is right.
4. `strategies.py` — three rules, each ~10 lines.
5. `analysis.py` — `sequence_risk_demo`, `max_sustainable_spend`, `compare_models`.

## Baseline results (`retiree.toml`, seed 42, 20,000 paths)

$1M at 65, $40k/yr real (4%), 60/40 stocks/bonds, 0.30% fee, plan to 95; stocks 6% ± 17%, bonds 2% ± 7% real.

| Question | Answer |
|---|---|
| P(success), lognormal | 82.1% (95% CI 81.6–82.7%) |
| Same returns, worst years first / best years first | 0.7% / 100% |
| Student-t (df=5) / bootstrap / block bootstrap, all moment-matched | 81.8% / 81.1% / 83.5% |
| Block bootstrap on RAW history (stocks ≈ 8.6% real) | 91.6% |
| Spending with 90% success | ≈ $36.1k/yr (3.6% of start) |
| Biggest drivers (±) | starting balance, spending, horizon, then returns/fees/vol |
| Guardrails, same $40k start | 100% success, but 55% of paths see a >20% spending cut |

These depend on the return assumptions in the config; change them and re-run.

## Design decisions worth understanding

* **Vectorize across paths, loop across years.** Years are sequential; paths are independent.
* **Real terms everywhere.** Removes nominal/real mixing bugs. Bootstrap deflates each historical year with that year's own inflation.
* **Timing convention:** cash flow at the *start* of the year, then returns. (Slightly optimistic for withdrawals.)
* **Shortfall, not just ruin.** A path fails if it ever cannot fund the year's net spending; the unfunded amount is recorded.
* **Common random numbers.** Sensitivity and root-finding reuse the same random draws so differences are signal, not noise.
* **Moment matching for fair model comparison.** Bootstrap history is rescaled to the configured mean/vol, so models differ only in *shape* and *dependence*.
* **Stateful strategies, one instance per run.** Guardrails remembers each path's last spending.

## Known simplifications (candidates for extension)

* Annual time step; withdrawals at start of year (add end-of-year timing as an exercise).
* Fixed horizon: no mortality/longevity uncertainty (a real planner would sample death age).
* No taxes, no sequence-dependent asset location, costless rebalancing.
* Inflation is folded into real returns; there is no separate inflation-risk model.
* Lognormal correlation is applied to the underlying normals; the correlation of the resulting arithmetic returns differs slightly.
* Guardrails ignore remaining horizon (a 90-year-old gets cut like a 65-year-old) and have no spending floor.
* Historical data is US-only, 98 annual observations: bootstrap results carry their own sampling uncertainty, and the US 20th century is a famously good sample.
* Student-t is applied in arithmetic space and floored at a total loss — deliberately not exponentiated (exp of a heavy-tailed variable has an infinite mean).

## Data

`data/us_annual_returns.csv` is derived from Aswath Damodaran's *Historical Returns on Stocks, Bonds and Bills* (NYU Stern, `histretSP.xls`, workbook dated 2026-01-01):
`stocks` = S&P 500 including dividends, `bonds` = 10-year US Treasury total return, `bills` = 3-month T-bill, `inflation` = CPI-U annual % change (FRED CPIAUCNS, as included in that workbook). Inflation is an annual average while returns are calendar-year, so the real-return alignment is approximate. Check the source for terms of use before redistributing.


## Web app (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py            # opens http://localhost:8501
```

**How `app.py` works.** Streamlit re-runs the script top to bottom whenever an input changes.
The sidebar widgets are collected into a plain dict, validated by `config_from_dict`, and serialized to a
JSON string that serves as the *cache key*: identical inputs return instantly. The main simulation runs
automatically; expensive analyses (sensitivity, model comparison, sequence demo, safe-spend solver) sit behind
buttons, and their results are hidden as soon as the inputs change so a stale chart is never shown.
All maths stays in the `retirement_mc` package; `app.py` only draws and wires.

### Deploy on Streamlit Community Cloud (free)

1. Put the project in a GitHub repo with `app.py`, `requirements.txt`, `retirement_mc/` and `data/` at the repo root.
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **New app**.
3. Choose the repo, branch (`main`) and main file path `app.py`.
4. Under *Advanced settings* pick Python 3.11 or newer, then **Deploy**.
5. Every `git push` redeploys automatically.

Other hosts (Render, Railway, Fly.io, Cloud Run, a VPS) work too: install `requirements.txt` and run
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

### Running it as a public service: things to know

* **Compute is on your server.** Every visitor's simulation uses your CPU and RAM. The simulation-count slider is capped at 50,000 and only 2 full result sets are cached (`max_entries=2`) to protect memory; lower `SIM_CHOICES` in `app.py` if you host on a small instance.
* **Inputs are sent to your server.** Add a privacy note if you publish it. Nothing is stored by this app.
* **Disclaimer.** The app labels itself educational, not financial advice. Get proper legal/compliance wording before publishing a retirement tool to the public.
* **Data licence.** Check Damodaran's terms before redistributing `data/us_annual_returns.csv`; the parametric models work without it.
* **Not implemented (good next steps):** a "load plan (JSON)" uploader, shareable URLs via `st.query_params`, a glide path (changing stock % with age), stochastic lifespan.
