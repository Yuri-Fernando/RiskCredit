"""Otimização de ponto de corte por custo esperado e lucro esperado.

Não usa 0,5. Para cada limiar t (aprova se PD < t):
  Expected Cost  = FP_cost · FP + FN_cost · FN
    FP = bom rejeitado (margem perdida), FN = mau aprovado (perda LGD·EAD)
  Expected Profit = Σ_aprovados [ (receita − funding)·EAD·(1 − PD) − PD·LGD·EAD − opex ]
Premissas econômicas em configs/v3.yaml (ilustrativas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def threshold_curve(y, pd_, ead, lgd: float, revenue_rate: float, funding_rate: float, opex: float,
                    grid: np.ndarray | None = None) -> pd.DataFrame:
    y, p, ead = np.asarray(y), np.asarray(pd_), np.asarray(ead, dtype=float)
    grid = grid if grid is not None else np.round(np.arange(0.02, 0.81, 0.01), 2)
    margin = (revenue_rate - funding_rate) * ead
    loss = lgd * ead
    rows = []
    for t in grid:
        ap = p < t
        fp = (~ap) & (y == 0)
        fn = ap & (y == 1)
        exp_profit = np.sum((margin * (1 - p) - p * loss - opex)[ap])          # ex-ante (usa PD)
        realized = np.sum(np.where(y == 1, -loss, margin)[ap]) - opex * ap.sum()  # ex-post (usa y)
        rows.append({"threshold": float(t), "approval_rate": float(ap.mean()),
                     "bad_rate_approved": float(y[ap].mean()) if ap.any() else np.nan,
                     "expected_loss": float(np.sum((p * loss)[ap])),
                     "expected_cost": float(np.sum(margin[fp]) + np.sum(loss[fn])),
                     "expected_profit": float(exp_profit), "realized_profit_test": float(realized),
                     "precision_bad_rejected": float(y[~ap].mean()) if (~ap).any() else np.nan,
                     "recall_bad_rejected": float(((~ap) & (y == 1)).sum() / max(1, (y == 1).sum()))})
    return pd.DataFrame(rows)


def optimal_thresholds(curve: pd.DataFrame) -> dict:
    c = curve.loc[curve["expected_cost"].idxmin()]
    p = curve.loc[curve["expected_profit"].idxmax()]
    return {"min_expected_cost": {"threshold": float(c["threshold"]), "approval_rate": float(c["approval_rate"]),
                                  "bad_rate_approved": float(c["bad_rate_approved"])},
            "max_expected_profit": {"threshold": float(p["threshold"]), "approval_rate": float(p["approval_rate"]),
                                    "bad_rate_approved": float(p["bad_rate_approved"]),
                                    "realized_profit_test": float(p["realized_profit_test"])},
            "naive_0_5": curve.iloc[(curve["threshold"] - 0.5).abs().argmin()][
                ["approval_rate", "bad_rate_approved", "expected_profit", "realized_profit_test"]].to_dict()}
