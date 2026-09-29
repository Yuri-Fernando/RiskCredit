"""Exporta o champion como contrato consumível pelo repositório AWS
(Credit-Score-Predictor---AWS-Streamlit): model.joblib + model.tar.gz,
model_card.json, feature_schema.json e metrics.json.

O artefato XGBoost (``xgboost-model.json``, formato JSON nativo) acompanha o pacote
porque o endpoint SageMaker existente usa o container XGBoost.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAW_INPUT_FEATURES = ["LIMIT_BAL", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                      "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                      "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
RANGES = {"LIMIT_BAL": (1, 5_000_000), "PAY_AMT": (0, 5_000_000), "BILL_AMT": (-1_000_000, 5_000_000),
          "PAY_": (-2, 9)}


def _range(col: str) -> tuple[float, float]:
    # prefixo mais longo primeiro: "PAY_AMT1" também começa com "PAY_"
    for prefix in sorted(RANGES, key=len, reverse=True):
        if col.startswith(prefix):
            return RANGES[prefix]
    return (-np.inf, np.inf)


def feature_schema(model_features: list[str]) -> dict:
    return {
        "schema_version": "1.0.0",
        "raw_input_features": [{"name": c, "dtype": "number", "min": _range(c)[0], "max": _range(c)[1],
                                "required": True} for c in RAW_INPUT_FEATURES],
        "derived_features": [f for f in model_features if f not in RAW_INPUT_FEATURES],
        "model_features_ordered": model_features,
        "excluded_attributes": ["SEX", "MARRIAGE", "AGE", "EDUCATION"],
        "feature_engineering": "src/data/load.py::engineer (RiskCredit V3)",
    }


def reference_profile(train_df: pd.DataFrame, features: list[str], train_scores: np.ndarray, n_bins: int = 10) -> dict:
    """Perfil de referência para monitoramento de drift no serviço de inferência:
    bordas de bins (quantis; valores únicos para discretas) e proporções do treino."""
    prof = {}
    for f in [*features, "__score__"]:
        x = np.asarray(train_scores if f == "__score__" else train_df[f], dtype=float)
        uniq = np.unique(x)
        if len(uniq) <= 20:
            prof[f] = {"type": "discrete", "values": uniq.tolist(),
                       "proportions": [float((x == v).mean()) for v in uniq]}
        else:
            edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1))[1:-1])
            idx = np.searchsorted(edges, x, side="right")
            prof[f] = {"type": "continuous", "edges": edges.tolist(),
                       "proportions": (np.bincount(idx, minlength=len(edges) + 1) / len(x)).tolist()}
    return {"schema_version": "1.0.0", "n_reference": int(len(train_df)), "features": prof}


def golden_samples(raw_rows: pd.DataFrame, pd_values: np.ndarray) -> list[dict]:
    """Entradas brutas + PD esperada do champion — teste de contrato/paridade no repositório AWS."""
    out = []
    for (_, r), p in zip(raw_rows.iterrows(), pd_values):
        out.append({"input": {c: float(r[c]) for c in RAW_INPUT_FEATURES}, "expected_pd": float(p)})
    return out


def export_champion(out_dir: Path, champion_name: str, model, model_features: list[str], metrics: dict,
                    data_hash_: str, xgb_model=None, extra_card: dict | None = None,
                    reference: dict | None = None, golden: list[dict] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": model_features, "name": champion_name}, out_dir / "model.joblib")
    files = ["model.joblib"]
    if xgb_model is not None:
        xgb_model.get_booster().save_model(str(out_dir / "xgboost-model.json"))
        files.append("xgboost-model.json")
    schema = feature_schema(model_features)
    (out_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")
    if reference is not None:
        (out_dir / "reference_profile.json").write_text(json.dumps(reference, indent=1), encoding="utf-8")
    if golden is not None:
        (out_dir / "golden_samples.json").write_text(json.dumps(golden, indent=1), encoding="utf-8")
    with tarfile.open(out_dir / "model.tar.gz", "w:gz") as tar:
        for f in files:
            tar.add(out_dir / f, arcname=f)
    sha = hashlib.sha256((out_dir / "model.tar.gz").read_bytes()).hexdigest()
    card = {
        "model_name": "riskcredit-pd-12m", "champion": champion_name,
        "version": f"v3-{datetime.now(timezone.utc):%Y%m%d}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "intended_use": "PoC de portfólio — score de PD para simulação de estratégia; NÃO é sistema de concessão real",
        "training_data": {"source": "UCI Default of Credit Card Clients (Taiwan, 2005)", "sha256": data_hash_},
        "target": "default no mês seguinte (out/2005)",
        "artifact_sha256": sha,
        "approval_status": "PendingManualApproval",
        "limitations": ["sem OOT real (dataset sem originação/safra)", "premissas econômicas ilustrativas",
                        "dados de 2005, Taiwan — sem validade para carteira brasileira atual"],
        **(extra_card or {}),
    }
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False, default=float),
                                             encoding="utf-8")
    return card
