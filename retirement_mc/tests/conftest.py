from pathlib import Path

import pytest

from retirement_mc.config import Asset, Config, Portfolio, Returns, Simulation, Spending, Timeline

DATA = Path(__file__).resolve().parent.parent / "data" / "us_annual_returns.csv"


@pytest.fixture
def base_cfg() -> Config:
    """A small, fast retiree config. Tests tweak copies of it."""
    return Config(
        timeline=Timeline(65, 65, 95),
        portfolio=Portfolio(1_000_000, [0.6, 0.4], True, 0.003),
        assets=[Asset("stocks", 0.06, 0.17), Asset("bonds", 0.02, 0.07)],
        returns=Returns(model="lognormal", correlation=[[1, 0.1], [0.1, 1]], history_path=str(DATA)),
        spending=Spending(rule="fixed", annual=40_000),
        simulation=Simulation(n_sims=5_000, seed=1),
    ).validate()


@pytest.fixture
def data_path() -> str:
    return str(DATA)
