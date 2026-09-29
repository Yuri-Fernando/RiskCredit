"""Segmentação comportamental da carteira: K-Means, GMM e DBSCAN.

- Features comportamentais padronizadas (sem demografia, sem o alvo).
- k escolhido por silhouette (K-Means) e BIC (GMM); Davies-Bouldin reportado.
- Estabilidade: bootstrap refit → ARI contra a solução de referência e
  Jaccard máximo por cluster (Hennig, 2007: < 0,6 = instável).
- PCA só para visualização.
- Clusters NÃO são verdade econômica: o rótulo é descritivo, gerado a
  partir do perfil, e acompanhado da taxa de default e da estabilidade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEG_FEATURES = ["utilization_avg", "pay_ratio_avg", "mean_delay", "max_delay", "n_months_delayed",
                "bill_trend", "LIMIT_BAL"]


def _prep(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    X = df[SEG_FEATURES].copy()
    X["LIMIT_BAL"] = np.log1p(X["LIMIT_BAL"])
    sc = StandardScaler().fit(X)
    return sc.transform(X), sc


def select_k(Z: np.ndarray, k_range=(3, 8), seed: int = 7, sample: int = 8000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=min(sample, len(Z)), replace=False)
    rows = []
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(k, n_init=10, random_state=seed).fit(Z)
        gm = GaussianMixture(k, covariance_type="full", random_state=seed, n_init=2).fit(Z)
        rows.append({"k": k,
                     "kmeans_silhouette": float(silhouette_score(Z[idx], km.labels_[idx])),
                     "kmeans_davies_bouldin": float(davies_bouldin_score(Z, km.labels_)),
                     "gmm_bic": float(gm.bic(Z))})
    return pd.DataFrame(rows)


def bootstrap_stability(Z: np.ndarray, k: int, n_boot: int = 30, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    ref = KMeans(k, n_init=10, random_state=seed).fit(Z)
    aris, jacc = [], np.zeros((n_boot, k))
    for b in range(n_boot):
        idx = rng.choice(len(Z), size=len(Z), replace=True)
        km = KMeans(k, n_init=5, random_state=seed + b + 1).fit(Z[idx])
        lab_b = km.predict(Z)  # rótulos do refit aplicados à base completa
        aris.append(adjusted_rand_score(ref.labels_, lab_b))
        for c in range(k):
            a = ref.labels_ == c
            jacc[b, c] = max((a & (lab_b == d)).sum() / (a | (lab_b == d)).sum() for d in range(k))
    j = jacc.mean(axis=0)
    return {"ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris)),
            "jaccard_by_cluster": [float(x) for x in j],
            "unstable_clusters": [int(c) for c in np.where(j < 0.6)[0]], "n_boot": n_boot}


def dbscan_segments(Z: np.ndarray, min_samples: int = 25, quantile: float = 0.90, sample: int = 12000,
                    seed: int = 7) -> dict:
    """DBSCAN em amostra (custo O(n²) no pior caso). eps pelo quantil da
    distância ao k-ésimo vizinho. Ruído (-1) é informação, não erro."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=min(sample, len(Z)), replace=False)
    Zs = Z[idx]
    d = NearestNeighbors(n_neighbors=min_samples).fit(Zs).kneighbors(Zs)[0][:, -1]
    eps = float(np.quantile(d, quantile))
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Zs)
    return {"eps": eps, "min_samples": min_samples, "sample_idx": idx, "labels": lab,
            "n_clusters": int(len(set(lab)) - (1 if -1 in lab else 0)), "noise_share": float((lab == -1).mean())}


def profile(df: pd.DataFrame, labels: np.ndarray, y: np.ndarray, ead: np.ndarray) -> pd.DataFrame:
    d = df[SEG_FEATURES].copy()
    d["segment"], d["default"], d["ead"] = labels, y, ead
    g = d.groupby("segment").agg(n=("default", "size"), default_rate=("default", "mean"),
                                 utilization_avg=("utilization_avg", "mean"), pay_ratio_avg=("pay_ratio_avg", "median"),
                                 mean_delay=("mean_delay", "mean"), n_months_delayed=("n_months_delayed", "mean"),
                                 bill_trend=("bill_trend", "mean"), limit_median=("LIMIT_BAL", "median"),
                                 exposure=("ead", "sum"))
    g["share"] = g["n"] / g["n"].sum()
    g["label"] = [_label(r, g) for _, r in g.iterrows()]
    return g.reset_index().sort_values("default_rate")


def _label(r, g) -> str:
    """Rótulo descritivo derivado do perfil (não é verdade econômica)."""
    hi_delay = r["mean_delay"] >= g["mean_delay"].quantile(0.75)
    lo_delay = r["mean_delay"] <= g["mean_delay"].quantile(0.25)
    hi_util = r["utilization_avg"] >= g["utilization_avg"].median()
    hi_limit = r["limit_median"] >= g["limit_median"].median()
    if hi_delay and r["n_months_delayed"] >= 3:
        return "atraso persistente"
    if hi_delay:
        return "deterioração recente / atraso"
    if lo_delay and hi_limit and not hi_util:
        return "baixo risco, alto limite, bom pagamento"
    if hi_util and r["pay_ratio_avg"] < 0.2:
        return "rotativo recorrente (utilização alta, pagamento mínimo)"
    if hi_util:
        return "utilização alta com pagamento consistente"
    return "uso moderado"


def run_segmentation(df: pd.DataFrame, y, ead, k_range=(3, 8), n_boot: int = 30, seed: int = 7) -> dict:
    Z, _ = _prep(df)
    ksel = select_k(Z, k_range, seed)
    k = int(ksel.sort_values("kmeans_silhouette", ascending=False).iloc[0]["k"])
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    k_gmm = int(ksel.sort_values("gmm_bic").iloc[0]["k"])
    gm = GaussianMixture(k_gmm, covariance_type="full", random_state=seed, n_init=2).fit(Z)
    db = dbscan_segments(Z, seed=seed)
    stab = bootstrap_stability(Z, k, n_boot, seed)
    y = np.asarray(y)
    prof = profile(df, km.labels_, y, ead)
    prof_gmm = profile(df, gm.predict(Z), y, ead)
    pca = PCA(2, random_state=seed).fit(Z)
    db_lab = db["labels"]
    db_dr = pd.Series(y[db["sample_idx"]]).groupby(db_lab).agg(["size", "mean"]).rename(
        columns={"size": "n", "mean": "default_rate"})
    return {
        "k_selection": ksel, "k_kmeans": k, "k_gmm": k_gmm,
        "kmeans_labels": km.labels_, "profile_kmeans": prof, "profile_gmm": prof_gmm,
        "ari_kmeans_vs_gmm": float(adjusted_rand_score(km.labels_, gm.predict(Z))),
        "stability": stab,
        "dbscan": {k_: v for k_, v in db.items() if k_ not in ("labels", "sample_idx")},
        "dbscan_profile": db_dr.reset_index().rename(columns={"index": "cluster"}),
        "pca_coords": pca.transform(Z), "pca_explained": pca.explained_variance_ratio_.tolist(),
    }
