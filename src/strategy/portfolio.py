"""Portfolio strategy: do PD à decisão (simulação — não é sistema de concessão).

- EL = PD × LGD × EAD.
- Capital econômico: fórmula IRB de Basileia II para varejo rotativo
  qualificado (R = 0,04): K = LGD · [Φ((Φ⁻¹(PD) + √R Φ⁻¹(0,999)) / √(1−R)) − PD],
  capital = K · EAD. Usado como proxy de capital, sem ajuste de maturidade.
- RAROC simplificado = (receita − funding − EL − opex) / capital.
- Pricing bands por faixa de PD; sugestão de limite pelo RAROC mínimo;
  prioridade de cobrança por PD × EAD; matriz de estratégia segmento × faixa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

PD_BANDS = [0, 0.05, 0.10, 0.20, 0.35, 0.60, 1.0]
BAND_LABELS = ["A (<5%)", "B (5–10%)", "C (10–20%)", "D (20–35%)", "E (35–60%)", "F (≥60%)"]


def irb_capital_k(pd_, lgd: float, R: float = 0.04, conf: float = 0.999) -> np.ndarray:
    p = np.clip(np.asarray(pd_, dtype=float), 3e-4, 0.9999)  # piso de PD de Basileia (0,03%)
    return lgd * (norm.cdf((norm.ppf(p) + np.sqrt(R) * norm.ppf(conf)) / np.sqrt(1 - R)) - p)


def account_economics(pd_, ead, lgd, revenue_rate, funding_rate, opex, R=0.04, conf=0.999) -> pd.DataFrame:
    p, ead = np.asarray(pd_, dtype=float), np.asarray(ead, dtype=float)
    el = p * lgd * ead
    capital = irb_capital_k(p, lgd, R, conf) * ead
    revenue = revenue_rate * ead * (1 - p)
    funding = funding_rate * ead
    net = revenue - funding - el - opex
    raroc = np.divide(net, capital, out=np.full_like(net, np.nan), where=capital > 0)
    return pd.DataFrame({"pd": p, "ead": ead, "expected_loss": el, "capital": capital, "revenue": revenue,
                         "net_income": net, "raroc": raroc,
                         "pd_band": pd.cut(p, PD_BANDS, labels=BAND_LABELS, include_lowest=True).astype(str)})


def pricing_bands(econ: pd.DataFrame, lgd, funding_rate, opex_rate_proxy: float, hurdle: float, R=0.04) -> pd.DataFrame:
    """Taxa mínima por faixa que cobre funding + EL + opex + hurdle × capital."""
    g = econ.groupby("pd_band").agg(n=("pd", "size"), mean_pd=("pd", "mean"), ead=("ead", "sum"),
                                    el=("expected_loss", "sum"), capital=("capital", "sum"),
                                    raroc_median=("raroc", "median"))
    g["min_rate_to_hurdle"] = funding_rate + (g["el"] + hurdle * g["capital"]) / g["ead"] + opex_rate_proxy
    return g.reset_index()


def limit_suggestion(pd_, current_limit, lgd, revenue_rate, funding_rate, hurdle, R=0.04) -> np.ndarray:
    """Limite = limite atual se RAROC marginal por R$ ≥ hurdle; reduzido
    proporcionalmente caso contrário (piso 50%). Heurística de simulação."""
    p = np.asarray(pd_, dtype=float)
    k = irb_capital_k(p, lgd, R)
    unit_raroc = ((revenue_rate * (1 - p)) - funding_rate - p * lgd) / np.where(k > 0, k, np.nan)
    factor = np.clip(np.nan_to_num(unit_raroc / hurdle, nan=0.5), 0.5, 1.0)
    return np.asarray(current_limit, dtype=float) * factor


def collections_priority(pd_, ead) -> np.ndarray:
    """Ranking por perda esperada evitável (PD × EAD); 1 = maior prioridade."""
    v = np.asarray(pd_) * np.asarray(ead)
    return (-v).argsort().argsort() + 1


def strategy_matrix(econ: pd.DataFrame, segments: pd.Series, hurdle: float) -> pd.DataFrame:
    d = econ.assign(segment=np.asarray(segments))
    g = d.groupby(["segment", "pd_band"]).agg(n=("pd", "size"), raroc=("raroc", "median"),
                                             el=("expected_loss", "sum")).reset_index()
    g["action"] = np.select(
        [g["raroc"] >= 2 * hurdle, g["raroc"] >= hurdle, g["raroc"] >= 0],
        ["aprovar / aumentar limite", "aprovar", "aprovar com reprecificação"], "recusar ou reduzir exposição")
    return g
