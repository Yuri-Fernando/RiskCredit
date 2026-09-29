"""Validação temporal possível com o UCI — e o que NÃO é possível.

Não há OOT verdadeiro: todas as contas têm o mesmo alvo (out/2005) e não
existe data de originação. Em vez de fingir OOT, dois testes honestos:

1. **Robustez de defasagem (horizon-shift):** o modelo treinado com a janela
   comportamental completa é aplicado a uma visão DEFASADA de 1 mês (mês mais
   recente descartado; demais meses deslocados). Mede quanto o desempenho cai
   quando a informação mais recente não está disponível — proxy de sensibilidade
   a atraso de dados, NÃO desempenho out-of-time.
2. **Estabilidade entre folds** (já na v2, K-Fold estratificado).

OOT real fica no ROADMAP: exige dataset com data de originação/performance
(ex.: base pública de empréstimos com safras) ou o motor de ECL do IFRS17_Risk,
que tem painel sintético por safra.
"""

from __future__ import annotations

import pandas as pd

from src.data.load import BILL, PAY, PAYAMT, engineer
from src.validation.metrics import discrimination


def lagged_view(raw: pd.DataFrame) -> pd.DataFrame:
    """Desloca a janela 1 mês para trás: mês 2 vira o mais recente; o mais
    antigo é repetido (não há mês 7 disponível)."""
    v = raw.copy()
    for cols in (PAY, BILL, PAYAMT):
        v[cols] = v[cols].to_numpy()[:, [1, 2, 3, 4, 5, 5]]
    return engineer(v)


def horizon_shift_test(predict, raw_test: pd.DataFrame, y) -> dict:
    base = discrimination(y, predict(engineer(raw_test)))
    lag = discrimination(y, predict(lagged_view(raw_test)))
    return {"full_window": base, "lagged_1_month": lag,
            "gini_drop": base["gini"] - lag["gini"],
            "interpretation": "queda de Gini ao perder o mês mais recente (sensibilidade à defasagem de dados)",
            "is_out_of_time": False}
