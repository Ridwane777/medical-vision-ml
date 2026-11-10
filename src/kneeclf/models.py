"""Candidate pipelines and their cross-validated hyperparameter search.

Feature extraction (raw / HOG / LBP) is stateless, so it is computed once per feature kind and the
searched pipelines start after it; the saved model re-attaches the extractor in front (see train.py).
"""
from __future__ import annotations

import logging
import time

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

log = logging.getLogger(__name__)

# name -> (feature kind, ordered steps). Every searched parameter is prefixed by its step name.
CANDIDATES = {
    "raw_pca_logreg": ("raw", ("scale", "pca", "logreg")),
    "raw_pca_svc": ("raw", ("scale", "pca", "svc")),
    "hog_logreg": ("hog", ("scale", "logreg")),
    "hog_svc": ("hog", ("scale", "svc")),
    "hog_rf": ("hog", ("rf",)),
    "hog_pca_hgb": ("hog", ("scale", "pca", "hgb")),
    "lbp_logreg": ("lbp", ("scale", "logreg")),          # optional texture baseline
    "hog+lbp_svc": ("hog+lbp", ("scale", "svc")),        # optional combination
}

SCORING = {"roc_auc": "roc_auc", "recall": "recall", "f1": "f1", "balanced_accuracy": "balanced_accuracy",
           "average_precision": "average_precision"}


def _step(name, seed):
    if name == "scale":
        return StandardScaler()
    if name == "pca":
        return PCA(n_components=50, svd_solver="randomized", random_state=seed)
    if name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced")
    if name == "svc":
        return SVC(kernel="rbf", class_weight="balanced", cache_size=1000)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=1)
    if name == "hgb":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    raise ValueError(name)


def build_pipeline(name, seed=0):
    if name not in CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}; choose from {sorted(CANDIDATES)}")
    _, steps = CANDIDATES[name]
    return Pipeline([(s, _step(s, seed)) for s in steps])


def feature_kind(name):
    return CANDIDATES[name][0]


def search(name, X, y, grid, cv_folds=3, refit="roc_auc", n_jobs=4, seed=0, verbose=1):
    """GridSearchCV with StratifiedKFold; records every metric of SCORING, refits on `refit`."""
    pipe = build_pipeline(name, seed)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, grid or {}, scoring=SCORING, refit=refit, cv=cv, n_jobs=n_jobs,
                      verbose=verbose, error_score="raise")
    t0 = time.perf_counter()
    gs.fit(X, y)
    elapsed = time.perf_counter() - t0
    i = gs.best_index_
    cv_scores = {m: round(float(gs.cv_results_[f"mean_test_{m}"][i]), 4) for m in SCORING}
    log.info("%s: best %s=%.4f params=%s (%.1fs, %d fits)", name, refit, gs.best_score_, gs.best_params_,
             elapsed, len(gs.cv_results_["params"]) * cv_folds)
    return gs, {"best_params": gs.best_params_, "cv": cv_scores, "search_seconds": round(elapsed, 2),
                "n_settings": len(gs.cv_results_["params"])}
