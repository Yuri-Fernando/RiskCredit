"""Scorecard logístico tradicional (WoE → logística → pontos).

Escala: Score = offset + factor · ln(odds bons:maus), factor = PDO / ln 2,
offset = base_score − factor · ln(base_odds). Com PDO=20 e 600 pontos em
50:1, cada +20 pontos dobram as odds.
Seleção: IV ≥ iv_min, filtro de correlação |r| < 0,7 (mantém maior IV) e
remoção iterativa de coeficientes com sinal contraintuitivo.
Reason codes: atributos com maior distância até a pontuação máxima possível
da característica (padrão de adverse action).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.woe import WoETransformer


@dataclass
class Scorecard:
    woe: WoETransformer
    features: list[str]
    model: LogisticRegression
    base_score: float = 600
    base_odds: float = 50
    pdo: float = 20
    dropped: dict = field(default_factory=dict)

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.woe.transform(X, self.features))[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.predict_proba(X), 1e-9, 1 - 1e-9)
        return self.offset + self.factor * np.log((1 - p) / p)

    def points_table(self) -> pd.DataFrame:
        b0 = self.model.intercept_[0]
        n = len(self.features)
        rows = []
        for f, beta in zip(self.features, self.model.coef_[0]):
            spec = self.woe.specs[f]
            for _, r in spec.table.iterrows():
                pts = -(beta * r["woe"] + b0 / n) * self.factor + self.offset / n
                rows.append({"feature": f, "range": r["range"], "woe": r["woe"], "n": int(r["n"]),
                             "bad_rate": r["bad_rate"], "points": round(float(pts), 1)})
        return pd.DataFrame(rows)

    def reason_codes(self, X: pd.DataFrame, top: int = 3) -> list[list[str]]:
        beta = dict(zip(self.features, self.model.coef_[0]))
        W = self.woe.transform(X, self.features)
        gaps = pd.DataFrame(index=X.index)
        for f in self.features:
            pts = -beta[f] * W[f] * self.factor
            best = (-beta[f] * self.woe.specs[f].woe * self.factor).max()
            gaps[f] = best - pts
        return [list(r.nlargest(top).index) for _, r in gaps.iterrows()]


def fit_scorecard(X: pd.DataFrame, y, n_bins: int = 6, min_share: float = 0.05, iv_min: float = 0.02,
                  corr_max: float = 0.7, base_score: float = 600, base_odds: float = 50, pdo: float = 20) -> Scorecard:
    woe = WoETransformer(n_bins, min_share).fit(X, y)
    iv = woe.iv_table()
    dropped = {r.feature: f"IV={r.iv:.3f} < {iv_min}" for r in iv.itertuples() if r.iv < iv_min}
    cands = [r.feature for r in iv.itertuples() if r.iv >= iv_min]
    W = woe.transform(X, cands)
    selected: list[str] = []
    for f in cands:  # ordem decrescente de IV
        if all(abs(np.corrcoef(W[f], W[s])[0, 1]) < corr_max for s in selected):
            selected.append(f)
        else:
            dropped[f] = f"correlação WoE ≥ {corr_max} com feature de maior IV"
    y = np.asarray(y)
    while True:
        m = LogisticRegression(C=1.0, max_iter=2000).fit(W[selected], y)
        wrong = [f for f, b in zip(selected, m.coef_[0]) if b > 0]  # WoE alto = baixo risco → β < 0
        if not wrong:
            break
        worst = max(wrong, key=lambda f: m.coef_[0][selected.index(f)])
        selected.remove(worst)
        dropped[worst] = "sinal contraintuitivo na presença das demais"
    return Scorecard(woe, selected, m, base_score, base_odds, pdo, dropped)
