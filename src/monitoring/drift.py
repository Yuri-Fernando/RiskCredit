"""Monitoramento de modelo: PSI por feature, CSI, drift de score/alvo/calibração
e performance por lote, com níveis de alerta (green < 0,10 ≤ amber < 0,25 ≤ red).

O UCI não tem série temporal de produção. Os "lotes de produção" são
construídos a partir do conjunto de teste com choques DOCUMENTADOS
(``simulate_batches``) — servem para demonstrar que o monitor detecta o que
deve detectar, não para descrever uma carteira real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load import BILL, PAY, PAYAMT, engineer
from src.validation.metrics import calibration, discrimination, psi, psi_level


def csi(spec, ref_x: np.ndarray, cur_x: np.ndarray) -> dict:
    """Characteristic Stability Index: PSI por atributo (bin de WoE) ponderado
    pelo WoE — mostra quanto o deslocamento de uma característica move o score."""
    n = len(spec.woe)
    e = np.bincount(spec.bin_index(ref_x), minlength=n) / len(ref_x)
    a = np.bincount(spec.bin_index(cur_x), minlength=n) / len(cur_x)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return {"psi_bins": float(np.sum((a - e) * np.log(a / e))), "csi_woe_shift": float(np.sum((a - e) * spec.woe))}


def simulate_batches(test_raw: pd.DataFrame, seed: int = 3) -> dict[str, pd.DataFrame]:
    """Lotes a partir do teste (sem reposição): 1–2 sem choque; 3 utilização +25%;
    4 atraso +1 mês em 30% das contas. Choques são premissas explícitas."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(test_raw))
    parts = np.array_split(idx, 4)
    batches = {}
    for i, p in enumerate(parts, start=1):
        b = test_raw.iloc[p].copy()
        if i == 3:
            b[BILL] = b[BILL] * 1.25
        if i == 4:
            hit = rng.random(len(b)) < 0.30
            b.loc[hit, PAY] = (b.loc[hit, PAY] + 1).clip(upper=8)
            b.loc[hit, PAYAMT] = b.loc[hit, PAYAMT] * 0.5
        batches[f"lote_{i}" + ("" if i < 3 else "_choque_utilizacao" if i == 3 else "_choque_atraso")] = engineer(b)
    return batches


def monitor(model_predict, woe_specs: dict, features: list[str], ref: pd.DataFrame, y_ref,
            batches: dict[str, pd.DataFrame], target: str, green: float = 0.10, amber: float = 0.25) -> dict:
    p_ref = model_predict(ref)
    out_batches, feat_rows = [], []
    for name, b in batches.items():
        p = model_predict(b)
        y = b[target].to_numpy()
        s_psi = psi(p_ref, p)
        cal = calibration(y, p)
        row = {"batch": name, "n": len(b), "score_psi": s_psi, "score_alert": psi_level(s_psi, green, amber),
               "target_rate": float(y.mean()), "target_drift_pp": float((y.mean() - np.mean(y_ref)) * 100),
               **{f"perf_{k}": v for k, v in discrimination(y, p).items()},
               "calibration_gap_pp": float((cal["mean_pred"] - cal["observed_rate"]) * 100), "ece": cal["ece"]}
        out_batches.append(row)
        for f in features:
            v = psi(ref[f].to_numpy(), b[f].to_numpy())
            c = csi(woe_specs[f], ref[f].to_numpy(), b[f].to_numpy()) if f in woe_specs else {}
            feat_rows.append({"batch": name, "feature": f, "psi": v, "alert": psi_level(v, green, amber), **c})
    fb = pd.DataFrame(feat_rows)
    return {"batches": pd.DataFrame(out_batches), "features": fb,
            "red_features": fb[fb["alert"] == "red"][["batch", "feature", "psi"]].to_dict(orient="records"),
            "note": "lotes 3 e 4 carregam choques simulados; target só é observável após maturação real"}
