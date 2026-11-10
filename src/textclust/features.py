"""TF-IDF + TruncatedSVD (LSA) + L2 normalization, with parameters derived from corpus size."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

log = logging.getLogger(__name__)


def tfidf_bounds(n_docs: int) -> tuple[int, float]:
    """min_df grows with corpus size (1, 2, 3, 5); max_df is looser for small corpora."""
    if n_docs < 50:
        min_df = 1
    elif n_docs < 2000:
        min_df = 2
    elif n_docs < 10000:
        min_df = 3
    else:
        min_df = 5
    max_df = 0.95 if n_docs < 100 else 0.8
    return min_df, max_df


@dataclass
class Features:
    vectorizer: TfidfVectorizer
    tfidf: object  # sparse matrix
    svd: TruncatedSVD | None
    embedding: np.ndarray  # L2-normalized reduced vectors
    params: dict


def build_features(texts, stop_words=None, ngram_range=(1, 1), max_components=100,
                   target_variance=None, random_state=42) -> Features:
    texts = list(texts)
    n = len(texts)
    min_df, max_df = tfidf_bounds(n)
    try:
        vec = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words, ngram_range=ngram_range)
        X = vec.fit_transform(texts)
    except ValueError:  # vocabulary pruned to nothing: relax the bounds
        min_df, max_df = 1, 1.0
        vec = TfidfVectorizer(min_df=1, max_df=1.0, stop_words=stop_words, ngram_range=ngram_range)
        X = vec.fit_transform(texts)
    n_features = X.shape[1]
    n_comp = min(max_components, n_features - 1, n - 1)
    svd, explained = None, None
    if n_comp >= 2:
        svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        Z = svd.fit_transform(X)
        cum = np.cumsum(svd.explained_variance_ratio_)
        if target_variance is not None and cum[-1] >= target_variance:
            keep = int(np.searchsorted(cum, target_variance) + 1)
            keep = max(keep, 2)
            Z = Z[:, :keep]
            n_comp = keep
        explained = float(cum[n_comp - 1])
    else:
        Z = X.toarray()
    emb = normalize(Z, norm="l2")
    params = {"min_df": min_df, "max_df": max_df, "ngram_range": list(ngram_range),
              "n_features": int(n_features), "svd_components": int(n_comp) if svd else 0,
              "svd_explained_variance": round(explained, 4) if explained is not None else None,
              "stop_words": "none" if stop_words is None else f"{len(stop_words)} words"}
    log.info("features: %d docs x %d terms (min_df=%s, max_df=%s), SVD %s comps, variance %s",
             n, n_features, min_df, max_df, params["svd_components"], params["svd_explained_variance"])
    return Features(vec, X, svd, emb, params)
