"""Automatic choice of the number of clusters and of the algorithm (KMeans vs HDBSCAN)."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from sklearn import metrics
from sklearn.cluster import HDBSCAN, KMeans

log = logging.getLogger(__name__)


def k_range(n: int, k_max_cap: int = 20) -> range:
    """k from 2 to min(cap, max(3, sqrt(n/2))), and never more than n - 1."""
    k_max = min(k_max_cap, max(3, int(math.sqrt(n / 2))), n - 1)
    return range(2, max(k_max, 2) + 1)


def hdbscan_min_cluster_size(n: int) -> int:
    """About 2% of the corpus, at least 3 documents."""
    return max(3, int(round(0.02 * n)))


def internal_scores(X, labels) -> dict:
    mask = labels >= 0
    Xm, lm = X[mask], labels[mask]
    k = len(set(lm))
    if k < 2 or len(lm) <= k:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    return {"silhouette": float(metrics.silhouette_score(Xm, lm, metric="cosine")),
            "calinski_harabasz": float(metrics.calinski_harabasz_score(Xm, lm)),
            "davies_bouldin": float(metrics.davies_bouldin_score(Xm, lm))}


def external_scores(true_labels, labels) -> dict:
    return {"ari": float(metrics.adjusted_rand_score(true_labels, labels)),
            "nmi": float(metrics.normalized_mutual_info_score(true_labels, labels)),
            "homogeneity": float(metrics.homogeneity_score(true_labels, labels)),
            "completeness": float(metrics.completeness_score(true_labels, labels))}


@dataclass
class ModelChoice:
    algorithm: str
    labels: np.ndarray
    n_clusters: int
    noise_ratio: float
    scores: dict
    k_curve: list = field(default_factory=list)
    candidates: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)


def select_model(X, random_state=42, k_max_cap=20, max_noise=0.3, algorithms=("kmeans", "hdbscan"),
                 fixed_k=None) -> ModelChoice:
    n = X.shape[0]
    candidates = {}
    curve = []
    if "kmeans" in algorithms:
        ks = [fixed_k] if fixed_k else list(k_range(n, k_max_cap))
        best = None
        for k in ks:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
            sc = internal_scores(X, km.labels_)
            curve.append({"k": k, "inertia": float(km.inertia_), **sc})
            if sc["silhouette"] is not None and (best is None or sc["silhouette"] > best[1]["silhouette"]):
                best = (km, sc, k)
        if best:
            km, sc, k = best
            candidates["kmeans"] = {"labels": km.labels_, "n_clusters": k, "noise_ratio": 0.0, "scores": sc,
                                    "params": {"k": k, "k_range": [ks[0], ks[-1]], "n_init": 10}}
    if "hdbscan" in algorithms and n >= 10:
        mcs = hdbscan_min_cluster_size(n)
        hd = HDBSCAN(min_cluster_size=mcs, copy=True).fit(X)
        lab = hd.labels_
        k = len(set(lab[lab >= 0]))
        noise = float(np.mean(lab < 0))
        candidates["hdbscan"] = {"labels": lab, "n_clusters": k, "noise_ratio": noise,
                                 "scores": internal_scores(X, lab), "params": {"min_cluster_size": mcs}}

    # Selection: highest silhouette (on non-noise points), among candidates with >= 2 clusters
    # and a noise ratio <= max_noise (otherwise HDBSCAN could "win" by discarding hard points).
    eligible = {a: c for a, c in candidates.items()
                if c["scores"]["silhouette"] is not None and c["n_clusters"] >= 2 and c["noise_ratio"] <= max_noise}
    if not eligible:
        raise RuntimeError("no clustering candidate produced at least two clusters")
    algo = max(eligible, key=lambda a: eligible[a]["scores"]["silhouette"])
    c = eligible[algo]
    summary = {a: {"n_clusters": v["n_clusters"], "noise_ratio": round(v["noise_ratio"], 4),
                   "eligible": a in eligible, **v["scores"], **v["params"]} for a, v in candidates.items()}
    log.info("model selection: %s", {a: (v["n_clusters"], v["silhouette"], v["noise_ratio"]) for a, v in summary.items()})
    log.info("chosen: %s with %d clusters, silhouette=%.3f, noise=%.1f%%",
             algo, c["n_clusters"], c["scores"]["silhouette"], 100 * c["noise_ratio"])
    return ModelChoice(algo, c["labels"], c["n_clusters"], c["noise_ratio"], c["scores"], curve, summary, c["params"])
