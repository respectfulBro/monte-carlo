import numpy as np
import pytest
from scipy import stats

from retirement_mc.config import Asset
from retirement_mc.returns import (gen_block_bootstrap, gen_bootstrap, gen_lognormal, gen_student_t,
                                   get_history, lognormal_params, match_moments)

ASSETS = [Asset("s", 0.06, 0.17), Asset("b", 0.02, 0.07)]
CORR = np.array([[1.0, 0.3], [0.3, 1.0]])


def test_lognormal_hits_target_moments():
    rng = np.random.default_rng(0)
    g = gen_lognormal(ASSETS, CORR, 200_000, 5, rng)
    r = (g - 1).reshape(-1, 2)
    assert r.mean(0) == pytest.approx([0.06, 0.02], abs=0.002)
    assert r.std(0) == pytest.approx([0.17, 0.07], abs=0.003)
    assert np.corrcoef(r.T)[0, 1] == pytest.approx(0.3, abs=0.02)
    assert (g > 0).all()


def test_lognormal_median_below_mean():
    """The convexity correction: median gross = exp(m) < E[gross] = 1 + mu."""
    m, s = lognormal_params(np.array([0.06]), np.array([0.17]))
    assert np.exp(m[0]) < 1.06
    assert m[0] == pytest.approx(np.log(1.06) - s[0] ** 2 / 2)


def test_student_t_matches_variance_but_has_fat_tails():
    rng = np.random.default_rng(1)
    g = gen_student_t(ASSETS, CORR, 5.0, 300_000, 3, rng)
    r = (g - 1).reshape(-1, 2)
    assert r.mean(0) == pytest.approx([0.06, 0.02], abs=0.003)
    assert r.std(0) == pytest.approx([0.17, 0.07], rel=0.05)
    assert stats.kurtosis(r[:, 0]) > 2.0                        # normal would be ~0
    gn = (gen_lognormal(ASSETS, CORR, 300_000, 3, rng) - 1).reshape(-1, 2)
    # extreme-loss frequency: t should beat lognormal by a wide margin
    thr = 0.06 - 4 * 0.17
    assert (r[:, 0] < thr).mean() > 3 * max((gn[:, 0] < thr).mean(), 1e-6)


def test_student_t_tail_dependence():
    """Shared chi-square mixing: joint extreme losses are more common than under independent-tail (normal) models."""
    rng = np.random.default_rng(2)
    n = 400_000
    t = (gen_student_t(ASSETS, np.eye(2), 4.0, n, 1, rng) - 1).reshape(-1, 2)
    ln = (gen_lognormal(ASSETS, np.eye(2), n, 1, rng) - 1).reshape(-1, 2)
    def joint(x):
        z = (x - x.mean(0)) / x.std(0)
        return ((z[:, 0] < -2.5) & (z[:, 1] < -2.5)).mean()
    assert joint(t) > 3 * joint(ln)


def test_bootstrap_samples_only_historical_rows():
    hist = np.arange(10, dtype=float).reshape(-1, 1) / 100.0     # 10 years x 1 asset
    g = gen_bootstrap(hist, 1000, 20, np.random.default_rng(0))
    assert g.shape == (1000, 20, 1)
    assert set(np.round((g - 1).ravel(), 2)) <= set(np.round(hist.ravel(), 2))


def test_block_bootstrap_preserves_runs():
    T, b = 12, 4
    hist = (np.arange(T, dtype=float) / 1000.0).reshape(-1, 1)   # row i encodes its own index
    g = gen_block_bootstrap(hist, 500, 8, b, np.random.default_rng(0))
    idx = np.round((g - 1).squeeze(-1) * 1000).astype(int)       # recover the sampled indices
    assert idx.shape == (500, 8)
    step = (idx[:, 1:] - idx[:, :-1]) % T
    within = np.ones(7, bool); within[b - 1] = False             # boundary between block 1 and 2 is free
    assert (step[:, within] == 1).all()                          # inside a block: consecutive (circular) years


def test_block_len_one_is_iid_bootstrap_in_distribution():
    hist = np.random.default_rng(0).normal(0.05, 0.15, (60, 1))
    a = gen_block_bootstrap(hist, 20_000, 10, 1, np.random.default_rng(1)) - 1
    ac = np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1]
    assert abs(ac) < 0.02


def test_block_bootstrap_keeps_serial_correlation():
    rng = np.random.default_rng(0)
    x = np.zeros(400); x[0] = 0.0
    for i in range(1, 400):                                      # strongly autocorrelated AR(1) history
        x[i] = 0.8 * x[i - 1] + rng.normal(0, 0.05)
    hist = x.reshape(-1, 1)
    def lag1(g):
        r = (g - 1).squeeze(-1)
        return np.corrcoef(r[:, :-1].ravel(), r[:, 1:].ravel())[0, 1]
    iid = lag1(gen_bootstrap(hist, 5000, 20, np.random.default_rng(1)))
    blk = lag1(gen_block_bootstrap(hist, 5000, 20, 10, np.random.default_rng(1)))
    assert abs(iid) < 0.05 and blk > 0.4


def test_match_moments_is_exact():
    hist = np.random.default_rng(0).normal(0.03, 0.2, (90, 2))
    m = match_moments(hist, ASSETS)
    np.testing.assert_allclose(m.mean(0), [0.06, 0.02], atol=1e-12)
    np.testing.assert_allclose(m.std(0, ddof=1), [0.17, 0.07], atol=1e-12)


def test_history_file_loads_and_is_real(base_cfg):
    import pandas as pd
    hist = get_history(base_cfg)                                # matched to config mu/sigma
    assert hist.shape == (98, 2)
    np.testing.assert_allclose(hist.mean(0), [0.06, 0.02], atol=1e-12)
    raw = pd.read_csv(base_cfg.returns.history_path)
    assert raw["year"].min() == 1928 and raw["year"].max() == 2025 and raw.notna().all().all()
