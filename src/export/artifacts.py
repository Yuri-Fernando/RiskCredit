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
RANGES = {"LIMIT_BAL": (1, 5_000_000), "PAY_": (-2, 9), "BILL_AMT": (-1_000_000, 5_000_000),
          "PAY_AMT": (0, 5_000_000)}


def _range(col: str) -> tuple[float, float]:
    for prefix, r in RANGES.items():
        if col.startswith(prefix):
            return r
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


def export_champion(out_dir: Path, champion_name: str, model, model_features: list[str], metrics: dict,
                    data_hash_: str, xgb_model=None, extra_card: dict | None = None) -> dict:
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
