"""Stress testing de carteira: choques em utilização, atraso, limite e pagamento.

O UCI não tem renda; o choque de "renda" do plano não é aplicável e foi
substituído por redução de pagamento (proxy de capacidade de pagamento).
Mede: variação de PD média, EL (PD × LGD × EAD), concentração (HHI por
segmento e participação do decil de maior risco na EL).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import BILL, PAY, PAYAMT, engineer, exposure_at_default


def apply_shock(raw: pd.DataFrame, shock: dict) -> pd.DataFrame:
    s = raw.copy()
    if "bill_mult" in shock:
        s[BILL] = s[BILL] * shock["bill_mult"]
    if "pay_shift" in shock:
        s[PAY] = (s[PAY] + shock["pay_shift"]).clip(upper=8)
    if "limit_mult" in shock:
        s["LIMIT_BAL"] = s["LIMIT_BAL"] * shock["limit_mult"]
    if "payamt_mult" in shock:
        s[PAYAMT] = s[PAYAMT] * shock["payamt_mult"]
    return engineer(s)


def _concentration(el: np.ndarray, seg: np.ndarray) -> dict:
    shares = pd.Series(el).groupby(seg).sum() / el.sum()
    top = np.sort(el)[::-1][: max(1, len(el) // 10)].sum() / el.sum()
    return {"hhi_segments": float((shares**2).sum()), "top_decile_el_share": float(top)}


def run_stress(raw_test: pd.DataFrame, predict, scenarios: dict, lgd: float, ccf: float, segments) -> pd.DataFrame:
    base = engineer(raw_test)
    p0 = predict(base)
    ead0 = exposure_at_default(base, ccf)
    el0 = p0 * lgd * ead0
    rows = [{"scenario": "base", "mean_pd": float(p0.mean()), "expected_loss": float(el0.sum()),
             "delta_pd_pp": 0.0, "delta_el_pct": 0.0, **_concentration(el0, segments)}]
    for name, shock in scenarios.items():
        s = apply_shock(raw_test, shock)
        p = predict(s)
        ead = exposure_at_default(s, ccf)
        el = p * lgd * ead
        rows.append({"scenario": name, "mean_pd": float(p.mean()), "expected_loss": float(el.sum()),
                     "delta_pd_pp": float((p.mean() - p0.mean()) * 100),
                     "delta_el_pct": float((el.sum() / el0.sum() - 1) * 100), **_concentration(el, segments)})
    return pd.DataFrame(rows)
