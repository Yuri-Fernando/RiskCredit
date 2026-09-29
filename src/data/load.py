"""Carga, limpeza e engenharia de atributos do UCI Default of Credit Card Clients.

Limpeza documentada: EDUCATION {0,5,6} → 4 ("outros"); MARRIAGE 0 → 3 ("outros").
Atributos demográficos (SEX, MARRIAGE, AGE, EDUCATION) ficam FORA das features
do modelo V3 e são usados apenas na auditoria de equidade — decisão de desenho
registrada no README (diferente da v2, que os usava como preditores).

Limitação temporal: o dataset tem 6 meses de histórico comportamental
(abr–set/2005) e um único alvo (out/2005). Não há data de originação nem safra;
portanto não existe OOT verdadeiro com estes dados (ver src/validation/robustness.py).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT = [f"PAY_AMT{i}" for i in range(1, 7)]
TARGET = "default"

BEHAVIOR_FEATURES = [
    "LIMIT_BAL", *PAY, *BILL, *PAYAMT,
    "mean_delay", "max_delay", "n_months_delayed", "utilization", "utilization_avg",
    "pay_ratio", "pay_ratio_avg", "bill_trend", "pay_trend",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"default.payment.next.month": TARGET})
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pay = out[PAY].clip(lower=0)
    out["mean_delay"] = pay.mean(axis=1)
    out["max_delay"] = pay.max(axis=1)
    out["n_months_delayed"] = (pay > 0).sum(axis=1)
    lim = out["LIMIT_BAL"].clip(lower=1)
    out["utilization"] = (out["BILL_AMT1"] / lim).clip(-1, 5)
    out["utilization_avg"] = (out[BILL].mean(axis=1) / lim).clip(-1, 5)
    bills = out[BILL].clip(lower=0)
    out["pay_ratio"] = (out["PAY_AMT1"] / bills["BILL_AMT2"].replace(0, np.nan)).fillna(1.0).clip(0, 5)
    out["pay_ratio_avg"] = (out[PAYAMT].sum(axis=1) / bills.sum(axis=1).replace(0, np.nan)).fillna(1.0).clip(0, 5)
    months = np.arange(6)[::-1]  # BILL_AMT1 é o mês mais recente
    out["bill_trend"] = _slope(out[BILL].to_numpy(float), months) / lim
    out["pay_trend"] = _slope(out[PAYAMT].to_numpy(float), months) / lim
    return out


def _slope(Y: np.ndarray, x: np.ndarray) -> np.ndarray:
    xc = x - x.mean()
    return ((Y - Y.mean(axis=1, keepdims=True)) * xc).sum(axis=1) / (xc**2).sum()


def exposure_at_default(df: pd.DataFrame, ccf: float) -> np.ndarray:
    """EAD = saldo utilizado + CCF × limite não utilizado (produto rotativo)."""
    drawn = df["BILL_AMT1"].clip(lower=0).to_numpy(float)
    undrawn = (df["LIMIT_BAL"].to_numpy(float) - drawn).clip(min=0)
    return drawn + ccf * undrawn


def data_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
