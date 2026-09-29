"""Champion/Challenger formalizado.

Critérios (definidos antes de ver resultados):
1. Discriminação — Gini e KS no teste;
2. Calibração — Brier e ECE (limite ECE ≤ 0,02);
3. Estabilidade — PSI do score treino → teste (limite < 0,10);
4. Explicabilidade — scorecard/logística = alta; GBM + SHAP = média;
5. Custo operacional — nº de features e latência de inferência por 1.000 contas.

Regra: entre os modelos que passam nos limites de calibração e estabilidade,
o challenger só substitui o scorecard se o ganho de Gini for ≥ 0,02
(materialidade); caso contrário o scorecard é mantido por explicabilidade.
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.scorecard import Scorecard
from src.validation.metrics import calibration, discrimination, psi

ECE_MAX, PSI_MAX, MATERIAL_GINI = 0.02, 0.10, 0.02
EXPLAINABILITY = {"scorecard": "alta", "logistic": "alta", "lightgbm": "média (SHAP)",
                  "xgboost": "média (SHAP)", "lightgbm_calibrated": "média (SHAP)"}


def train_challengers(Xtr: pd.DataFrame, ytr, seed: int = 42) -> dict:
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)),
        "lightgbm": lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                                       verbose=-1),
        "xgboost": xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                                     colsample_bytree=0.8, eval_metric="logloss", random_state=seed),
    }
    for m in models.values():
        m.fit(Xtr, ytr)
    cal = CalibratedClassifierCV(lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                                                    min_child_samples=50, subsample=0.8, subsample_freq=1,
                                                    colsample_bytree=0.8, random_state=seed, verbose=-1),
                                 method="isotonic", cv=3)
    cal.fit(Xtr, ytr)
    models["lightgbm_calibrated"] = cal
    return models


def evaluate(models: dict, scorecard: Scorecard, Xtr, ytr, Xte, yte) -> pd.DataFrame:
    rows = []
    allm = {"scorecard": scorecard, **models}
    for name, m in allm.items():
        p_tr = m.predict_proba(Xtr)[:, 1] if name != "scorecard" else m.predict_proba(Xtr)
        t0 = time.perf_counter()
        p_te = m.predict_proba(Xte)[:, 1] if name != "scorecard" else m.predict_proba(Xte)
        lat = (time.perf_counter() - t0) / len(Xte) * 1000 * 1000  # ms por 1.000 contas
        n_feat = len(scorecard.features) if name == "scorecard" else Xtr.shape[1]
        rows.append({"model": name, **discrimination(yte, p_te), **calibration(yte, p_te),
                     "psi_train_test": psi(p_tr, p_te), "explainability": EXPLAINABILITY[name],
                     "n_features": n_feat, "latency_ms_per_1k": round(lat, 2)})
    df = pd.DataFrame(rows)
    df["passes_calibration"] = df["ece"] <= ECE_MAX
    df["passes_stability"] = df["psi_train_test"] < PSI_MAX
    return df


def select_champion(table: pd.DataFrame) -> dict:
    ok = table[table["passes_calibration"] & table["passes_stability"]]
    sc = table.set_index("model").loc["scorecard"]
    if ok.empty:
        return {"champion": "scorecard", "reason": "nenhum modelo passou calibração+estabilidade; mantém scorecard"}
    best = ok.sort_values("gini", ascending=False).iloc[0]
    gain = best["gini"] - sc["gini"]
    if best["model"] != "scorecard" and gain >= MATERIAL_GINI:
        return {"champion": best["model"], "gini_gain_vs_scorecard": float(gain),
                "reason": f"ganho de Gini {gain:.3f} ≥ {MATERIAL_GINI} com calibração e estabilidade dentro do limite"}
    return {"champion": "scorecard", "gini_gain_best_challenger": float(gain), "best_challenger": best["model"],
            "reason": f"ganho do melhor challenger ({gain:.3f}) abaixo da materialidade {MATERIAL_GINI}; "
                      "scorecard mantido por explicabilidade"}
