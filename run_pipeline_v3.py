#!/usr/bin/env python
"""RiskCredit V3 — Credit Risk Analytics & Portfolio Strategy.

    python run_pipeline_v3.py            # executa tudo → reports/v3/ e artifacts/champion/

Fluxo: dados → features → scorecard WoE/IV → champion/challenger → robustez de
defasagem → threshold por custo/lucro → segmentação → estratégia (EL, capital,
RAROC, pricing, limites, cobrança) → stress → monitoramento → equidade →
export do champion para o repositório AWS.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.load import BEHAVIOR_FEATURES, TARGET, data_hash, engineer, exposure_at_default, load_raw  # noqa: E402
from src.export.artifacts import export_champion, golden_samples, reference_profile  # noqa: E402
from src.fairness.audit import run_audit  # noqa: E402
from src.models.champion_challenger import evaluate, select_champion, train_challengers  # noqa: E402
from src.models.scorecard import fit_scorecard  # noqa: E402
from src.monitoring.drift import monitor, simulate_batches  # noqa: E402
from src.segmentation.clustering import run_segmentation  # noqa: E402
from src.strategy.portfolio import (account_economics, collections_priority, limit_suggestion,  # noqa: E402
                                    pricing_bands, strategy_matrix)
from src.strategy.threshold import optimal_thresholds, threshold_curve  # noqa: E402
from src.stress.stress_test import run_stress  # noqa: E402
from src.validation.robustness import horizon_shift_test  # noqa: E402

OUT = ROOT / "reports" / "v3"
ART = ROOT / "artifacts" / "champion"


def _save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False)


def _fig(name: str):
    plt.tight_layout()
    plt.savefig(OUT / "figures" / f"{name}.png", dpi=120)
    plt.close()


def _book(econ: pd.DataFrame, y: np.ndarray, thr: float) -> dict:
    return {"threshold": thr, "n": int(len(econ)), "ead": float(econ["ead"].sum()),
            "expected_loss": float(econ["expected_loss"].sum()),
            "el_rate": float(econ["expected_loss"].sum() / econ["ead"].sum()),
            "raroc": float(econ["net_income"].sum() / econ["capital"].sum()),
            "observed_bad_rate": float(y.mean())}


def main() -> dict:
    t0 = time.perf_counter()
    cfg = yaml.safe_load((ROOT / "configs" / "v3.yaml").read_text(encoding="utf-8"))
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    eco = cfg["economics"]

    raw = load_raw(ROOT / cfg["data"]["file"])
    raw_tr, raw_te = train_test_split(raw, test_size=cfg["data"]["test_size"], stratify=raw[TARGET],
                                      random_state=cfg["data"]["random_state"])
    tr, te = engineer(raw_tr), engineer(raw_te)
    ytr, yte = tr[TARGET].to_numpy(), te[TARGET].to_numpy()
    Xtr, Xte = tr[BEHAVIOR_FEATURES], te[BEHAVIOR_FEATURES]
    print(f"[1/10] dados: treino={len(tr)} teste={len(te)} default={raw[TARGET].mean():.3f}")

    # 2. scorecard
    sc_cfg = cfg["scorecard"]
    sc = fit_scorecard(Xtr, ytr, sc_cfg["n_bins"], sc_cfg["min_bin_share"], sc_cfg["iv_min"],
                       base_score=sc_cfg["base_score"], base_odds=sc_cfg["base_odds"], pdo=sc_cfg["pdo"])
    iv = sc.woe.iv_table()
    _save(iv, "iv_table")
    _save(sc.points_table(), "scorecard_points")
    rc = sc.reason_codes(Xte.iloc[:5])
    print(f"[2/10] scorecard: {len(sc.features)} features {sc.features}")

    # 3. champion/challenger
    models = train_challengers(Xtr, ytr)
    table = evaluate(models, sc, Xtr, ytr, Xte, yte)
    _save(table, "champion_challenger")
    decision = select_champion(table)
    champ = decision["champion"]
    champ_model = sc if champ == "scorecard" else models[champ]

    def predict(df: pd.DataFrame) -> np.ndarray:
        X = df[BEHAVIOR_FEATURES]
        return champ_model.predict_proba(X) if champ == "scorecard" else champ_model.predict_proba(X)[:, 1]

    p_te = predict(te)
    print(f"[3/10] champion={champ} — {decision['reason']}")

    # 4. robustez temporal honesta
    robust = horizon_shift_test(predict, raw_te, yte)

    # 5. thresholds
    ead_te = exposure_at_default(te, eco["ccf"])
    curve = threshold_curve(yte, p_te, ead_te, eco["lgd"], eco["revenue_rate"], eco["funding_cost_rate"],
                            eco["opex_per_account"])
    _save(curve, "threshold_curve")
    opt = optimal_thresholds(curve)
    thr = opt["max_expected_profit"]["threshold"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(curve["threshold"], curve["expected_profit"] / 1e6, label="lucro esperado (ex-ante)")
    ax[0].plot(curve["threshold"], curve["realized_profit_test"] / 1e6, label="lucro realizado (teste)")
    ax[0].axvline(thr, ls="--", c="grey")
    ax[0].set(xlabel="limiar de PD", ylabel="NT$ milhões", title="Lucro × limiar")
    ax[0].legend()
    ax[1].plot(curve["approval_rate"], curve["bad_rate_approved"])
    ax[1].set(xlabel="taxa de aprovação", ylabel="bad rate dos aprovados", title="Trade-off aprovação × risco")
    _fig("threshold_curves")
    print(f"[4-5/10] robustez: queda de Gini com defasagem={robust['gini_drop']:.3f}; limiar ótimo={thr}")

    # 6. segmentação (base completa, sem alvo nas features)
    full = pd.concat([tr, te])
    ead_full = exposure_at_default(full, eco["ccf"])
    seg = run_segmentation(full, full[TARGET].to_numpy(), ead_full, tuple(cfg["segmentation"]["k_range"]),
                           cfg["segmentation"]["bootstrap"], cfg["segmentation"]["random_state"])
    _save(seg["k_selection"], "segmentation_k_selection")
    _save(seg["profile_kmeans"], "segmentation_profile_kmeans")
    _save(seg["profile_gmm"], "segmentation_profile_gmm")
    _save(seg["dbscan_profile"], "segmentation_dbscan")
    seg_te = pd.Series(seg["kmeans_labels"][len(tr):], index=te.index)
    fig, ax = plt.subplots(figsize=(7, 5))
    pc = seg["pca_coords"]
    idx = np.random.default_rng(0).choice(len(pc), 6000, replace=False)
    sca = ax.scatter(pc[idx, 0], pc[idx, 1], c=seg["kmeans_labels"][idx], s=4, cmap="tab10")
    ax.legend(*sca.legend_elements(), title="segmento", fontsize=7)
    ax.set(title=f"Segmentos K-Means (k={seg['k_kmeans']}) — PCA só para visualização",
           xlabel=f"PC1 ({seg['pca_explained'][0]:.0%})", ylabel=f"PC2 ({seg['pca_explained'][1]:.0%})")
    _fig("segmentation_pca")
    print(f"[6/10] segmentação: k={seg['k_kmeans']} ARI bootstrap={seg['stability']['ari_mean']:.3f}")

    # 7. estratégia
    econ = account_economics(p_te, ead_te, eco["lgd"], eco["revenue_rate"], eco["funding_cost_rate"],
                             eco["opex_per_account"], eco["asset_correlation_qrre"], eco["capital_confidence"])
    opex_rate = eco["opex_per_account"] / max(float(np.mean(ead_te)), 1.0)
    bands = pricing_bands(econ, eco["lgd"], eco["funding_cost_rate"], opex_rate, eco["hurdle_raroc"])
    _save(bands, "pricing_bands")
    labels = seg["profile_kmeans"].set_index("segment")["label"]
    matrix = strategy_matrix(econ, seg_te.map(lambda s: f"{s}: {labels[s]}").to_numpy(), eco["hurdle_raroc"])
    _save(matrix, "strategy_matrix")
    lim = limit_suggestion(p_te, te["LIMIT_BAL"], eco["lgd"], eco["revenue_rate"], eco["funding_cost_rate"],
                           eco["hurdle_raroc"])
    prio = collections_priority(p_te, ead_te)
    top100 = prio <= 100
    portfolio = {"total_ead": float(ead_te.sum()), "expected_loss": float(econ["expected_loss"].sum()),
                 "el_rate": float(econ["expected_loss"].sum() / ead_te.sum()),
                 "capital": float(econ["capital"].sum()),
                 "raroc_portfolio": float(econ["net_income"].sum() / econ["capital"].sum()),
                 "limit_reduction_share": float((lim < te["LIMIT_BAL"].to_numpy()).mean()),
                 "collections_top100_observed_bad_rate": float(yte[top100].mean()),
                 "approved_book_at_threshold": _book(econ[p_te < thr], yte[p_te < thr], thr),
                 "note": ("carteira inteira inclui contas que a política recusaria; UCI tem 22% de default "
                          "(crise de cartões de Taiwan, 2005) — premissas econômicas são ilustrativas"),
                 "currency": "NT$ (dólar taiwanês, moeda do dataset)"}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(bands["pd_band"], bands["raroc_median"])
    ax.axhline(eco["hurdle_raroc"], c="red", ls="--", label="hurdle")
    ax.set(title="RAROC mediano por faixa de PD", ylabel="RAROC")
    ax.legend()
    _fig("raroc_by_band")
    print(f"[7/10] estratégia: EL={portfolio['expected_loss']:,.0f} RAROC carteira={portfolio['raroc_portfolio']:.2f}")

    # 8. stress
    stress = run_stress(raw_te, predict, cfg["stress"]["scenarios"], eco["lgd"], eco["ccf"], seg_te.to_numpy())
    _save(stress, "stress_test")

    # 9. monitoramento
    batches = simulate_batches(raw_te)
    mon_feats = sc.features if champ == "scorecard" else list(iv["feature"].head(8))
    mon = monitor(predict, sc.woe.specs, mon_feats, tr, ytr, batches, TARGET,
                  cfg["monitoring"]["thresholds"]["green"], cfg["monitoring"]["thresholds"]["amber"])
    _save(mon["batches"], "monitoring_batches")
    _save(mon["features"], "monitoring_features")
    piv = mon["features"].pivot(index="feature", columns="batch", values="psi")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn_r", vmin=0, vmax=0.3, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=20, fontsize=7)
    ax.set_yticks(range(len(piv.index)), piv.index, fontsize=7)
    plt.colorbar(im, label="PSI")
    ax.set_title("PSI por feature e lote (lotes 3–4 com choque simulado)")
    _fig("monitoring_psi")
    print(f"[8-9/10] stress e monitoramento: {len(mon['red_features'])} alertas vermelhos")

    # 10. equidade + export
    audit = run_audit(te, yte, p_te, thr)
    for k, v in audit.items():
        _save(v, f"fairness_{k.lower()}")
    metrics = {"champion": champ, "decision": decision,
               "test": table.set_index("model").loc[champ].drop(["explainability"]).to_dict(),
               "threshold_policy": opt, "robustness": robust}
    xgb_model = models["xgboost"]
    extra = {"feature_list": sc.features if champ == "scorecard" else BEHAVIOR_FEATURES,
             "threshold_max_expected_profit": thr, "scorecard_scale": {"base_score": sc.base_score,
                                                                       "base_odds": sc.base_odds, "pdo": sc.pdo}}
    feats_out = sc.features if champ == "scorecard" else BEHAVIOR_FEATURES
    ref = reference_profile(tr, feats_out, predict(tr))
    gold = golden_samples(raw_te.head(25), p_te[:25])
    card = export_champion(ART, champ, champ_model, feats_out, metrics, data_hash(raw), xgb_model=xgb_model,
                           extra_card=extra, reference=ref, golden=gold)
    if champ == "scorecard":
        (ART / "scorecard.json").write_text(json.dumps({
            "features": sc.features, "intercept": float(sc.model.intercept_[0]),
            "coefficients": dict(zip(sc.features, map(float, sc.model.coef_[0]))),
            "bins": {f: {"edges": sc.woe.specs[f].edges.tolist(), "woe": sc.woe.specs[f].woe.tolist()}
                     for f in sc.features},
            "base_score": sc.base_score, "base_odds": sc.base_odds, "pdo": sc.pdo,
            "bin_rule": "índice = searchsorted(edges, x, side='left'); bins (-inf,e1], (e1,e2], ..."},
            indent=2), encoding="utf-8")

    summary = {
        "version": "3.0.0", "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "champion": decision, "scorecard_features": sc.features, "scorecard_dropped": sc.dropped,
        "reason_codes_example": rc,
        "champion_challenger": table.round(4).to_dict(orient="records"),
        "robustness": robust, "threshold": opt,
        "segmentation": {"k_kmeans": seg["k_kmeans"], "k_gmm": seg["k_gmm"], "stability": seg["stability"],
                         "ari_kmeans_vs_gmm": seg["ari_kmeans_vs_gmm"], "dbscan": seg["dbscan"],
                         "profiles": seg["profile_kmeans"].round(4).to_dict(orient="records")},
        "portfolio": portfolio, "stress": stress.round(4).to_dict(orient="records"),
        "monitoring": {"batches": mon["batches"].round(4).to_dict(orient="records"),
                       "red_features": mon["red_features"]},
        "fairness": {k: v.round(4).to_dict(orient="records") for k, v in audit.items()},
        "model_card": card,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    print(f"[10/10] concluído em {summary['elapsed_seconds']}s → {OUT}")
    return summary


if __name__ == "__main__":
    main()
