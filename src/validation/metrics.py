"""Métricas de discriminação, calibração e estabilidade."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    g = df.groupby("bin").agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
    return float((g["n"] / g["n"].sum() * (g["obs"] - g["pred"]).abs()).sum())


def discrimination(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": ks_statistic(y, p),
            "pr_auc": float(average_precision_score(y, p))}


def calibration(y, p) -> dict:
    return {"brier": float(brier_score_loss(y, p)), "ece": expected_calibration_error(y, p),
            "mean_pred": float(np.mean(p)), "observed_rate": float(np.mean(y))}


def psi(expected, actual, n_bins: int = 10, eps: float = 1e-4, max_discrete: int = 20) -> float:
    """Population Stability Index. Contínuas: bins por quantis da referência.
    Discretas (≤ ``max_discrete`` valores distintos, ex.: PAY_x): um bin por
    valor — quantis colapsam em empates e escondem deslocamentos."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3 or len(np.unique(expected)) <= max_discrete:
        vals = np.unique(np.concatenate([expected, actual]))
        e = np.array([(expected == v).mean() for v in vals])
        a = np.array([(actual == v).mean() for v in vals])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected, edges)[0] / len(expected)
        a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_level(value: float, green: float = 0.10, amber: float = 0.25) -> str:
    return "green" if value < green else "amber" if value < amber else "red"
