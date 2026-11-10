"""Cluster description (top terms, sizes, examples), 2D projection, plots and metrics.json."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

log = logging.getLogger(__name__)


def top_terms(tfidf, labels, vectorizer, n_terms=8) -> dict:
    """Highest mean TF-IDF weights (the cluster centroid in term space)."""
    vocab = np.asarray(vectorizer.get_feature_names_out())
    out = {}
    for c in sorted(set(labels)):
        if c < 0:
            continue
        centroid = np.asarray(tfidf[labels == c].mean(axis=0)).ravel()
        out[int(c)] = vocab[np.argsort(centroid)[::-1][:n_terms]].tolist()
    return out


def representatives(embedding, labels, texts, n_examples=3) -> dict:
    """Documents closest (cosine) to each cluster centroid in the reduced space."""
    texts = list(texts)
    out = {}
    for c in sorted(set(labels)):
        if c < 0:
            continue
        idx = np.where(labels == c)[0]
        centroid = embedding[idx].mean(axis=0)
        sims = embedding[idx] @ centroid
        out[int(c)] = [texts[i] for i in idx[np.argsort(sims)[::-1][:n_examples]]]
    return out


def cluster_table(labels, terms, examples) -> pd.DataFrame:
    sizes = pd.Series(labels).value_counts()
    rows = [{"cluster": int(c), "size": int(sizes[c]), "top_terms": ", ".join(terms.get(int(c), [])),
             "example": (examples.get(int(c)) or ["(noise)"])[0]} for c in sorted(sizes.index)]
    return pd.DataFrame(rows)


def project_2d(embedding, random_state=42):
    """t-SNE (perplexity < n) for n >= 30, PCA otherwise."""
    n = embedding.shape[0]
    if n >= 30:
        perplexity = min(30.0, (n - 1) / 3)
        return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state).fit_transform(embedding), \
            {"method": "t-SNE", "perplexity": round(perplexity, 2)}
    comps = min(2, embedding.shape[1])
    xy = PCA(n_components=comps, random_state=random_state).fit_transform(embedding)
    if comps == 1:
        xy = np.column_stack([xy, np.zeros(n)])
    return xy, {"method": "PCA"}


def plot_k_curve(curve, path):
    if len(curve) < 2:
        return None
    ks = [c["k"] for c in curve]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, [c["inertia"] for c in curve], "o-")
    ax[0].set(title="K-Means inertia (elbow)", xlabel="k", ylabel="inertia")
    ax[1].plot(ks, [c["silhouette"] for c in curve], "o-", color="tab:orange")
    ax[1].set(title="Silhouette (cosine)", xlabel="k", ylabel="silhouette")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return fig


def plot_projection(xy, labels, info, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    noise = labels < 0
    if noise.any():
        ax.scatter(xy[noise, 0], xy[noise, 1], s=6, c="lightgrey", label="noise")
    sc = ax.scatter(xy[~noise, 0], xy[~noise, 1], s=8, c=labels[~noise], cmap="tab20")
    fig.colorbar(sc, ax=ax, label="cluster")
    ax.set_title(f"{info['method']} projection coloured by cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return fig


def plot_sizes(table, path):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(range(len(table)), table["size"])
    ax.set_xticks(range(len(table)), [str(c) for c in table["cluster"]])
    ax.set(title="Cluster sizes", xlabel="cluster (-1 = noise)", ylabel="documents")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return fig


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    log.info("wrote %s", path)
