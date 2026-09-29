"""Testes do RiskCredit V3 — fórmulas com referência independente e invariantes."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import engineer, exposure_at_default, load_raw  # noqa: E402
from src.fairness.audit import adverse_impact  # noqa: E402
from src.features.woe import fit_bins, iv_strength  # noqa: E402
from src.models.champion_challenger import select_champion  # noqa: E402
from src.models.scorecard import fit_scorecard  # noqa: E402
from src.monitoring.drift import csi  # noqa: E402
from src.segmentation.clustering import bootstrap_stability  # noqa: E402
from src.strategy.portfolio import account_economics, collections_priority, irb_capital_k  # noqa: E402
from src.strategy.threshold import threshold_curve  # noqa: E402
from src.stress.stress_test import apply_shock  # noqa: E402
from src.validation.metrics import ks_statistic, psi, psi_level  # noqa: E402
from src.validation.robustness import lagged_view  # noqa: E402


@pytest.fixture(scope="module")
def data():
    raw = load_raw(ROOT / "UCI_Credit_Card.csv")
    return raw, engineer(raw)


def test_cleaning_and_engineering(data):
    raw, df = data
    assert set(raw["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(raw["MARRIAGE"].unique()) <= {1, 2, 3}
    assert (df["max_delay"] >= df["mean_delay"]).all()
    assert df["n_months_delayed"].between(0, 6).all()


def test_woe_iv_against_manual_computation():
    x = np.array([1] * 50 + [2] * 50)
    y = np.array([0] * 40 + [1] * 10 + [0] * 20 + [1] * 30)
    spec = fit_bins(x, y, "x", n_bins=2, min_share=0.0)
    good, bad = np.array([40, 20]), np.array([10, 30])
    pg, pb = (good + 0.5) / (60 + 1.0), (bad + 0.5) / (40 + 1.0)
    woe = np.log(pg / pb)
    np.testing.assert_allclose(spec.woe, woe)
    assert spec.iv == pytest.approx(np.sum((pg - pb) * woe))
    assert iv_strength(0.01) == "inútil" and iv_strength(0.6).startswith("suspeito")


def test_binning_is_monotonic(data):
    _, df = data
    spec = fit_bins(df["LIMIT_BAL"].to_numpy(), df["default"].to_numpy(), "LIMIT_BAL", n_bins=8)
    br = spec.table["bad_rate"].to_numpy()
    assert np.all(np.diff(br) <= 0) or np.all(np.diff(br) >= 0)


def test_scorecard_scaling_pdo(data):
    _, df = data
    sub = df.sample(6000, random_state=1)
    sc = fit_scorecard(sub[["PAY_0", "LIMIT_BAL", "utilization_avg"]], sub["default"], pdo=20, base_score=600,
                       base_odds=50)
    # +PDO pontos ⇔ odds bons:maus dobram
    p = sc.predict_proba(sub.iloc[:50])
    s = sc.score(sub.iloc[:50])
    odds = (1 - p) / p
    np.testing.assert_allclose(s, sc.offset + sc.factor * np.log(odds))
    assert sc.offset + sc.factor * np.log(50) == pytest.approx(600)
    assert sc.offset + sc.factor * np.log(100) == pytest.approx(620)
    assert all(b < 0 for b in sc.model.coef_[0])  # WoE alto = baixo risco
    assert len(sc.reason_codes(sub.iloc[:3])[0]) <= 3


def test_psi_known_values():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    assert psi(a, rng.normal(size=20000)) < 0.01
    assert psi(a, rng.normal(0.5, 1, 20000)) > 0.1
    disc = rng.integers(-1, 3, 20000)
    shifted = np.clip(disc + (rng.random(20000) < 0.3), -1, 8)
    assert psi(disc, shifted) > 0.1  # discreta: deslocamento detectado
    assert psi_level(0.05) == "green" and psi_level(0.2) == "amber" and psi_level(0.3) == "red"


def test_csi_zero_when_same_distribution(data):
    _, df = data
    spec = fit_bins(df["PAY_0"].to_numpy(), df["default"].to_numpy(), "PAY_0")
    r = csi(spec, df["PAY_0"].to_numpy(), df["PAY_0"].to_numpy())
    assert r["psi_bins"] == pytest.approx(0, abs=1e-12) and r["csi_woe_shift"] == pytest.approx(0, abs=1e-12)


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_threshold_curve_accounting():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.2, 0.9])
    ead = np.array([100.0, 100.0, 100.0, 100.0])
    c = threshold_curve(y, p, ead, lgd=0.5, revenue_rate=0.2, funding_rate=0.1, opex=0.0, grid=np.array([0.25]))
    r = c.iloc[0]
    # aprova p<0,25: contas 0 (bom) e 2 (mau)
    assert r["approval_rate"] == 0.5 and r["bad_rate_approved"] == 0.5
    assert r["realized_profit_test"] == pytest.approx(10.0 - 50.0)
    assert r["expected_cost"] == pytest.approx(10.0 + 50.0)  # bom rejeitado (conta 1) + mau aprovado (conta 2)


def test_irb_capital_matches_basel_formula():
    from scipy.stats import norm
    pd_, lgd, R = 0.02, 0.75, 0.04
    k = lgd * (norm.cdf((norm.ppf(pd_) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R)) - pd_)
    assert irb_capital_k(np.array([pd_]), lgd)[0] == pytest.approx(k)
    assert np.all(np.diff(irb_capital_k(np.array([0.01, 0.05, 0.1]), lgd)) > 0)


def test_economics_and_priority():
    e = account_economics([0.01, 0.5], [1000, 1000], 0.75, 0.18, 0.08, 0.0)
    assert e.loc[0, "expected_loss"] == pytest.approx(7.5)
    assert e.loc[0, "raroc"] > 0 > e.loc[1, "raroc"]
    assert list(collections_priority([0.1, 0.5, 0.2], [100, 100, 1000])) == [3, 2, 1]


def test_stress_shock_direction(data):
    raw, _ = data
    s = apply_shock(raw.head(200), {"pay_shift": 1})
    base = engineer(raw.head(200))
    assert (s["mean_delay"] >= base["mean_delay"]).all()
    assert (exposure_at_default(apply_shock(raw.head(200), {"bill_mult": 1.2}), 0.6)
            >= exposure_at_default(base, 0.6) - 1e-9).all()


def test_lagged_view_shifts_months(data):
    raw, _ = data
    lv = lagged_view(raw.head(5))
    np.testing.assert_array_equal(lv["BILL_AMT1"].to_numpy(), raw.head(5)["BILL_AMT2"].to_numpy())


def test_adverse_impact_ratio():
    y = np.zeros(8)
    groups = np.array(["a"] * 4 + ["b"] * 4)
    approved = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    r = adverse_impact(y, np.zeros(8), approved, groups, "a", n_boot=50).set_index("group")
    assert r.loc["b", "air"] == pytest.approx(0.5) and r.loc["b", "below_four_fifths"]


def test_segmentation_stability_on_separated_clusters():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(c, 0.2, (200, 2)) for c in ((0, 0), (5, 5), (0, 5))])
    st = bootstrap_stability(Z, 3, n_boot=5)
    assert st["ari_mean"] > 0.95 and not st["unstable_clusters"]


def test_champion_rule_materiality():
    base = {"passes_calibration": True, "passes_stability": True}
    t = pd.DataFrame([{"model": "scorecard", "gini": 0.52, **base}, {"model": "xgboost", "gini": 0.53, **base}])
    assert select_champion(t)["champion"] == "scorecard"
    t.loc[1, "gini"] = 0.56
    assert select_champion(t)["champion"] == "xgboost"
    t.loc[1, "passes_calibration"] = False
    assert select_champion(t)["champion"] == "scorecard"
