"""Binning monotônico, Weight of Evidence (WoE) e Information Value (IV).

Convenção: WoE_i = ln(%bons_i / %maus_i) → WoE positivo = menor risco.
IV = Σ (%bons_i − %maus_i) · WoE_i. Suavização de 0,5 para bins sem eventos.
Faixas de IV (Siddiqi): <0,02 inútil · 0,02–0,1 fraco · 0,1–0,3 médio · >0,3 forte
(IV > 0,5 é suspeito de vazamento e deve ser investigado).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class BinSpec:
    feature: str
    edges: np.ndarray            # limites internos; bins = (-inf, e1], (e1, e2], ..., (ek, inf)
    woe: np.ndarray
    table: pd.DataFrame
    iv: float

    def bin_index(self, x: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, np.asarray(x, dtype=float), side="left")

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.woe[self.bin_index(x)]


def _woe_table(idx: np.ndarray, y: np.ndarray, n_bins: int) -> pd.DataFrame:
    df = pd.DataFrame({"bin": idx, "y": y})
    g = df.groupby("bin")["y"].agg(n="size", bad="sum").reindex(range(n_bins), fill_value=0)
    g["good"] = g["n"] - g["bad"]
    tot_good, tot_bad = g["good"].sum(), g["bad"].sum()
    g["pct_good"] = (g["good"] + 0.5) / (tot_good + 0.5 * n_bins)
    g["pct_bad"] = (g["bad"] + 0.5) / (tot_bad + 0.5 * n_bins)
    g["woe"] = np.log(g["pct_good"] / g["pct_bad"])
    g["iv_contrib"] = (g["pct_good"] - g["pct_bad"]) * g["woe"]
    g["bad_rate"] = g["bad"] / g["n"].replace(0, np.nan)
    return g.reset_index()


def fit_bins(x: np.ndarray, y: np.ndarray, feature: str, n_bins: int = 6, min_share: float = 0.05) -> BinSpec:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1))[1:-1])
    direction = np.sign(spearmanr(x, y).statistic) or 1.0

    def table_for(e):
        return _woe_table(np.searchsorted(e, x, side="left"), y, len(e) + 1)

    # funde bins pequenos e bins que violam monotonicidade do bad rate
    changed = True
    while changed and len(edges) > 0:
        changed = False
        t = table_for(edges)
        share = t["n"] / len(x)
        small = np.where(share < min_share)[0]
        if small.size:
            i = int(small[0])
            edges = np.delete(edges, min(i, len(edges) - 1))
            changed = True
            continue
        br = t["bad_rate"].to_numpy()
        diffs = np.diff(br) * direction
        bad = np.where(diffs < 0)[0]
        if bad.size:
            edges = np.delete(edges, int(bad[0]))
            changed = True
    t = table_for(edges)
    labels = []
    full = np.concatenate([[-np.inf], edges, [np.inf]])
    for lo, hi in zip(full[:-1], full[1:]):
        labels.append(f"({lo:g}, {hi:g}]")
    t.insert(1, "range", labels)
    return BinSpec(feature, edges, t["woe"].to_numpy(), t, float(t["iv_contrib"].sum()))


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "inútil"
    if iv < 0.1:
        return "fraco"
    if iv < 0.3:
        return "médio"
    if iv <= 0.5:
        return "forte"
    return "suspeito (>0,5) — investigar vazamento"


class WoETransformer:
    def __init__(self, n_bins: int = 6, min_share: float = 0.05):
        self.n_bins, self.min_share = n_bins, min_share
        self.specs: dict[str, BinSpec] = {}

    def fit(self, X: pd.DataFrame, y) -> "WoETransformer":
        for c in X.columns:
            self.specs[c] = fit_bins(X[c].to_numpy(), np.asarray(y), c, self.n_bins, self.min_share)
        return self

    def transform(self, X: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
        feats = features or list(self.specs)
        return pd.DataFrame({c: self.specs[c].transform(X[c].to_numpy()) for c in feats}, index=X.index)

    def iv_table(self) -> pd.DataFrame:
        rows = [{"feature": f, "iv": s.iv, "n_bins": len(s.woe), "strength": iv_strength(s.iv)}
                for f, s in self.specs.items()]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
