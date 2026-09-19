"""Configuration: typed dataclasses + TOML loading + validation.

Everything in the model is expressed in REAL (inflation-adjusted, today's) dollars
and REAL returns. That removes an entire class of bugs (nominal/real mixing) and
matches how people actually think about retirement spending.

All the knobs of a run live in a `Config`, so a simulation is fully described by
one object -- which makes runs reproducible, diffable, and easy to sweep.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

RETURN_MODELS = ("lognormal", "student_t", "bootstrap", "block_bootstrap")
SPENDING_RULES = ("fixed", "percent", "guardrails")


@dataclass
class Asset:
    name: str
    mu: float      # expected REAL arithmetic annual return, e.g. 0.05
    sigma: float   # standard deviation of the annual real return, e.g. 0.17


@dataclass
class Timeline:
    current_age: int = 65
    retire_age: int = 65
    end_age: int = 95          # plan until this age (simulation covers ages current..end-1)

    @property
    def years(self) -> int:
        return self.end_age - self.current_age

    @property
    def years_to_retire(self) -> int:
        return self.retire_age - self.current_age


@dataclass
class Portfolio:
    starting_balance: float = 1_000_000.0
    weights: list[float] = field(default_factory=lambda: [0.6, 0.4])
    rebalance: bool = True     # rebalance to target weights every year
    fee: float = 0.003         # annual expense ratio, taken as a drag on gross return


@dataclass
class Returns:
    model: str = "lognormal"
    correlation: list[list[float]] | None = None   # None -> independent assets
    df: float = 5.0                                # Student-t degrees of freedom (> 2)
    block_len: int = 5                             # block-bootstrap block length (years)
    history_path: str | None = None                # CSV with year, <asset cols>, inflation
    match_moments: bool = True                     # rescale history to the assets' mu/sigma


@dataclass
class Cashflow:
    annual_contribution: float = 0.0     # real $/yr while working
    contribution_growth: float = 0.0     # real growth of contributions (raises above inflation)
    social_security_age: int = 67
    social_security_annual: float = 0.0  # real $/yr, offsets spending once it starts


@dataclass
class Spending:
    rule: str = "fixed"
    annual: float = 40_000.0             # fixed / guardrails: real $/yr at retirement start
    rate: float = 0.04                   # percent rule: fraction of portfolio per year
    floor: float | None = None           # percent rule: min spend
    ceiling: float | None = None         # percent rule: max spend
    upper_band: float = 0.20             # guardrails: cut if rate > initial*(1+upper_band)
    lower_band: float = 0.20             # guardrails: raise if rate < initial*(1-lower_band)
    cut_pct: float = 0.10                # guardrails: spending cut size
    raise_pct: float = 0.10              # guardrails: spending raise size


@dataclass
class Simulation:
    n_sims: int = 10_000
    seed: int = 42


def _default_assets() -> list[Asset]:
    return [Asset("stocks", 0.06, 0.17), Asset("bonds", 0.02, 0.07)]


@dataclass
class Config:
    timeline: Timeline = field(default_factory=Timeline)
    portfolio: Portfolio = field(default_factory=Portfolio)
    assets: list[Asset] = field(default_factory=_default_assets)
    returns: Returns = field(default_factory=lambda: Returns(correlation=[[1.0, 0.1], [0.1, 1.0]]))
    cashflow: Cashflow = field(default_factory=Cashflow)
    spending: Spending = field(default_factory=Spending)
    simulation: Simulation = field(default_factory=Simulation)

    # ------------------------------------------------------------------ helpers
    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def asset_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.assets)

    def corr_matrix(self) -> np.ndarray:
        if self.returns.correlation is None:
            return np.eye(self.n_assets)
        return np.asarray(self.returns.correlation, dtype=float)

    # --------------------------------------------------------------- validation
    def validate(self) -> "Config":
        t, p, r, s, sim = self.timeline, self.portfolio, self.returns, self.spending, self.simulation
        k = self.n_assets
        if not (t.current_age <= t.retire_age < t.end_age):
            raise ValueError("Need current_age <= retire_age < end_age")
        if len(p.weights) != k:
            raise ValueError(f"{len(p.weights)} weights for {k} assets")
        if abs(sum(p.weights)) < 1e-12 or abs(sum(p.weights) - 1.0) > 1e-9 or min(p.weights) < 0:
            raise ValueError("weights must be non-negative and sum to 1")
        if not (0 <= p.fee < 1):
            raise ValueError("fee must be in [0, 1)")
        if p.starting_balance < 0:
            raise ValueError("starting_balance must be >= 0")
        for a in self.assets:
            if a.sigma < 0 or a.mu <= -1:
                raise ValueError(f"bad asset parameters: {a}")
        if r.model not in RETURN_MODELS:
            raise ValueError(f"returns.model must be one of {RETURN_MODELS}")
        if r.model == "student_t" and r.df <= 2:
            raise ValueError("student_t needs df > 2 (finite variance)")
        if r.model in ("bootstrap", "block_bootstrap") and not r.history_path:
            raise ValueError("bootstrap models need returns.history_path")
        if r.block_len < 1:
            raise ValueError("block_len must be >= 1")
        c = self.corr_matrix()
        if c.shape != (k, k) or not np.allclose(c, c.T) or not np.allclose(np.diag(c), 1.0):
            raise ValueError("correlation must be a symmetric k x k matrix with unit diagonal")
        if np.linalg.eigvalsh(c).min() <= 1e-10:
            raise ValueError("correlation matrix must be positive definite")
        if s.rule not in SPENDING_RULES:
            raise ValueError(f"spending.rule must be one of {SPENDING_RULES}")
        if sim.n_sims < 1:
            raise ValueError("n_sims must be >= 1")
        return self


# ------------------------------------------------------------------- loading
def _build(cls, d: dict, where: str):
    allowed = {f.name for f in fields(cls)}
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"Unknown key(s) {sorted(unknown)} in [{where}]; allowed: {sorted(allowed)}")
    return cls(**d)


def config_from_dict(raw: dict) -> Config:
    """Build and validate a Config from a plain nested dict (same shape as the TOML files).

    This is the single entry point for every front end: the TOML loader, the web app's
    form, and JSON round-trips all funnel through here, so validation happens in one place.
    """
    unknown_sections = set(raw) - {"timeline", "portfolio", "assets", "returns", "cashflow", "spending", "simulation"}
    if unknown_sections:
        raise ValueError(f"Unknown section(s): {sorted(unknown_sections)}")
    cfg = Config(
        timeline=_build(Timeline, raw.get("timeline", {}), "timeline"),
        portfolio=_build(Portfolio, raw.get("portfolio", {}), "portfolio"),
        assets=[_build(Asset, a, "assets") for a in raw["assets"]] if "assets" in raw else _default_assets(),
        returns=_build(Returns, raw.get("returns", {}), "returns"),
        cashflow=_build(Cashflow, raw.get("cashflow", {}), "cashflow"),
        spending=_build(Spending, raw.get("spending", {}), "spending"),
        simulation=_build(Simulation, raw.get("simulation", {}), "simulation"),
    )
    return cfg.validate()


def config_to_dict(cfg: Config) -> dict:
    """Plain-dict (JSON-serializable) form of a Config; inverse of `config_from_dict`."""
    return asdict(cfg)


def load_config(path: str | Path) -> Config:
    """Load a TOML file. A relative `returns.history_path` is resolved against the file's folder."""
    import tomllib   # stdlib in Python >= 3.11; imported lazily so the web app also runs on older versions

    path = Path(path)
    cfg = config_from_dict(tomllib.loads(path.read_text()))
    if cfg.returns.history_path and not Path(cfg.returns.history_path).is_absolute():
        cfg.returns.history_path = str((path.parent / cfg.returns.history_path).resolve())
    return cfg


# ----------------------------------------------------------------- overrides
def override(cfg: Config, path: str, value) -> Config:
    """Return a deep copy of `cfg` with one dotted-path field changed.

    Examples:  override(cfg, "spending.annual", 45_000)
               override(cfg, "assets.0.mu", 0.04)         # list items by index
    Used for sensitivity sweeps, root-finding, and model comparisons.
    """
    new = copy.deepcopy(cfg)
    parts = path.split(".")
    obj = new
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        if not hasattr(obj, last):
            raise AttributeError(f"{type(obj).__name__} has no field '{last}' (path '{path}')")
        setattr(obj, last, value)
    return new


def override_many(cfg: Config, changes: dict) -> Config:
    for path, value in changes.items():
        cfg = override(cfg, path, value)
    return cfg
