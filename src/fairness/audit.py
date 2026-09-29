"""Auditoria de equidade (atributos fora do modelo; nenhum ajuste de limiar).

Por grupo: taxa de aprovação, TPR (recall de maus), FPR, calibração
(PD média − taxa observada), Adverse Impact Ratio (aprovação do grupo /
aprovação do grupo de referência; regra dos 4/5 como sinal de investigação),
com IC 95% por bootstrap. Igualdade de uma métrica não garante equidade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUPINGS = {
    "SEX": {1: "masculino", 2: "feminino"},
    "EDUCATION": {1: "pós-graduação", 2: "universidade", 3: "ensino médio", 4: "outros"},
    "AGE_BAND": None,
}


def _age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [0, 25, 35, 45, 60, 200], labels=["≤25", "26–35", "36–45", "46–60", ">60"]).astype(str)


def group_metrics(y, pd_, approved, groups) -> pd.DataFrame:
    y, p, a, g = map(np.asarray, (y, pd_, approved, groups))
    rows = []
    for name in sorted(np.unique(g)):
        m = g == name
        yy, aa, pp = y[m], a[m], p[m]
        bad, good = yy == 1, yy == 0
        rows.append({"group": name, "n": int(m.sum()), "approval_rate": float(aa.mean()),
                     "tpr_bad_rejected": float((~aa[bad]).mean()) if bad.any() else np.nan,
                     "fpr_good_rejected": float((~aa[good]).mean()) if good.any() else np.nan,
                     "mean_pd": float(pp.mean()), "observed_bad_rate": float(yy.mean()),
                     "calibration_gap_pp": float((pp.mean() - yy.mean()) * 100)})
    return pd.DataFrame(rows)


def adverse_impact(y, pd_, approved, groups, reference, n_boot: int = 300, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a, g = np.asarray(approved), np.asarray(groups)
    base = group_metrics(y, pd_, approved, groups).set_index("group")
    ref_rate = base.loc[reference, "approval_rate"]
    out = []
    for name in base.index:
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(a), len(a))
            gi, ai = g[i], a[i]
            r_ref, r_g = ai[gi == reference].mean(), ai[gi == name].mean()
            if r_ref > 0:
                boots.append(r_g / r_ref)
        air = base.loc[name, "approval_rate"] / ref_rate
        lo, hi = np.quantile(boots, [0.025, 0.975])
        out.append({"group": name, "air": float(air), "air_ci_low": float(lo), "air_ci_high": float(hi),
                    "below_four_fifths": bool(air < 0.8)})
    return base.reset_index().merge(pd.DataFrame(out), on="group")


def run_audit(df_test: pd.DataFrame, y, pd_, threshold: float) -> dict:
    approved = np.asarray(pd_) < threshold
    res = {}
    for col, mapping in GROUPINGS.items():
        if col == "AGE_BAND":
            groups = _age_band(df_test["AGE"])
            ref = "26–35"
        else:
            groups = df_test[col].map(mapping).astype(str)
            ref = pd.Series(groups).value_counts().idxmax()
        res[col] = adverse_impact(y, pd_, approved, groups, ref)
        res[col]["reference_group"] = ref
    return res
