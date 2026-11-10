"""End-to-end training: load -> features -> per-candidate grid search -> threshold on val -> test once."""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import pickle
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from . import data, evaluate, features, models

log = logging.getLogger("kneeclf")

DEFAULT_GRIDS = {
    "raw_pca_logreg": {"pca__n_components": [100, 200], "logreg__C": [0.001, 0.01, 0.1]},
    "raw_pca_svc": {"pca__n_components": [100, 200], "svc__C": [1, 10], "svc__gamma": ["scale"]},
    "hog_logreg": {"logreg__C": [0.0003, 0.001, 0.01]},
    "hog_svc": {"svc__C": [1, 10], "svc__gamma": ["scale", 0.0001]},
    "hog_rf": {"rf__max_depth": [None, 20], "rf__min_samples_leaf": [1, 5]},
    "hog_pca_hgb": {"pca__n_components": [100], "hgb__learning_rate": [0.05, 0.1], "hgb__max_leaf_nodes": [31]},
    "lbp_logreg": {"logreg__C": [0.01, 0.1, 1]},
    "hog+lbp_svc": {"svc__C": [1, 10], "svc__gamma": ["scale"]},
}


@dataclass
class Config:
    label_scheme: str = "kl01_vs_2to4"
    image_size: int | None = 112            # None keeps the native 224 px
    balance_train: bool = True              # undersample the majority class, TRAIN only
    max_per_class: int | None = None        # cap per binary class on train (quick runs)
    max_per_class_eval: int | None = None   # cap per class on val/test (quick runs only; never balances)
    extra_test_splits: list = field(default_factory=list)   # e.g. ["auto_test"]
    candidates: list = field(default_factory=lambda: ["raw_pca_logreg", "raw_pca_svc", "hog_logreg",
                                                      "hog_svc", "hog_rf", "hog_pca_hgb"])
    param_grids: dict = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_GRIDS)))
    hog: dict = field(default_factory=lambda: {"cell": 8, "block": 2, "bins": 9})
    cv_folds: int = 3
    selection_metric: str = "roc_auc"       # CV refit metric and validation ranking metric
    threshold_mode: str = "target_recall"   # "target_recall" or "f2"
    target_recall: float = 0.85
    n_jobs: int = 4
    seed: int = 42
    verbose: int = 1

    @classmethod
    def from_json(cls, path, **overrides):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        if "param_grids" in raw:
            raw["param_grids"] = {**DEFAULT_GRIDS, **raw["param_grids"]}
        return cls(**raw)


@dataclass
class Result:
    run_dir: Path
    metrics: dict
    model: Pipeline
    threshold: float


def environment():
    import matplotlib
    import pandas
    import PIL
    import sklearn
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "pandas": pandas.__version__,
            "matplotlib": matplotlib.__version__, "pillow": PIL.__version__}


def _attach_log(run_dir):
    logger = logging.getLogger("kneeclf")
    handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return handler


def save_json(obj, path):
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=2, default=default), encoding="utf-8")


def load_model(path):
    """Return the bundle saved by `run`: {"model", "threshold", "label_scheme", "side", ...}.

    pickle can execute code: only load model files produced by your own runs.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(bundle, X):
    """Binary predictions for flattened images already resized to bundle['side']."""
    return (evaluate.decision_scores(bundle["model"], X) >= bundle["threshold"]).astype(int)


def run(data_root, config: Config | None = None, output_dir="runs") -> Result:
    import matplotlib.pyplot as plt

    cfg = config or Config()
    t_start = time.perf_counter()
    run_dir = Path(output_dir) / dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    handler = _attach_log(run_dir)
    timings = {}
    try:
        log.info("run dir: %s", run_dir)
        log.info("config: %s", json.dumps(dataclasses.asdict(cfg)))
        save_json(dataclasses.asdict(cfg), run_dir / "config.json")

        t = time.perf_counter()
        load = dict(label_scheme=cfg.label_scheme, image_size=cfg.image_size, seed=cfg.seed)
        train = data.load_split(data_root, "train", balance=cfg.balance_train, max_per_class=cfg.max_per_class,
                                **load)
        val = data.load_split(data_root, "val", max_per_class=cfg.max_per_class_eval, **load)
        test = data.load_split(data_root, "test", max_per_class=cfg.max_per_class_eval, **load)
        extra = [data.load_split(data_root, s, max_per_class=cfg.max_per_class_eval, **load)
                 for s in cfg.extra_test_splits]
        timings["load_seconds"] = round(time.perf_counter() - t, 2)
        profile = {s.name: s.profile for s in [train, val, test, *extra]}

        # Stateless features, computed once per kind and split.
        kinds = sorted({models.feature_kind(c) for c in cfg.candidates})
        feats, extractors = {}, {}
        for kind in kinds:
            t = time.perf_counter()
            ext = features.make_extractor(kind, cfg.hog)
            extractors[kind] = ext
            feats[kind] = {s.name: ext.transform(s.X) for s in [train, val, test, *extra]}
            sec = round(time.perf_counter() - t, 2)
            timings[f"features_{kind}_seconds"] = sec
            log.info("features %s: %d dims, %.1fs", kind, feats[kind]["train"].shape[1], sec)

        rows, searches = [], {}
        for i, name in enumerate(cfg.candidates, 1):
            kind = models.feature_kind(name)
            log.info("[%d/%d] grid search %s on %s features", i, len(cfg.candidates), name, kind)
            gs, info = models.search(name, feats[kind]["train"], train.y, cfg.param_grids.get(name, {}),
                                     cv_folds=cfg.cv_folds, refit=cfg.selection_metric, n_jobs=cfg.n_jobs,
                                     seed=cfg.seed, verbose=cfg.verbose)
            s_val = evaluate.decision_scores(gs.best_estimator_, feats[kind]["val"])
            thr = evaluate.choose_threshold(val.y, s_val, cfg.threshold_mode, cfg.target_recall)
            vm = evaluate.binary_metrics(val.y, s_val, thr)
            searches[name] = {**info, "val": vm, "estimator": gs.best_estimator_, "threshold": thr}
            rows.append({"candidate": name, "features": kind,
                         f"cv_{cfg.selection_metric}": info["cv"][cfg.selection_metric],
                         "val_roc_auc": vm["roc_auc"], "val_pr_auc": vm["pr_auc"], "val_recall": vm["recall"],
                         "val_specificity": vm["specificity"], "val_f2": vm["f2"],
                         "val_balanced_accuracy": vm["balanced_accuracy"],
                         "search_seconds": info["search_seconds"], "best_params": json.dumps(info["best_params"])})
            log.info("%s on val: roc_auc=%.4f recall=%.4f specificity=%.4f f2=%.4f", name, vm["roc_auc"],
                     vm["recall"], vm["specificity"], vm["f2"])

        table = evaluate.comparison_table(rows)
        rank_col = "val_" + cfg.selection_metric
        if rank_col in table:
            table = table.sort_values(rank_col, ascending=False, kind="stable").reset_index(drop=True)
        table.to_csv(run_dir / "comparison.csv", index=False)
        (run_dir / "comparison.md").write_text(evaluate.to_markdown(table), encoding="utf-8")
        best = table.loc[0, "candidate"]
        kind = models.feature_kind(best)
        chosen = searches[best]
        log.info("selected %s (by %s on validation); threshold %.4f", best, rank_col, chosen["threshold"])

        # The test set is touched exactly once, with the model and threshold frozen on train/val.
        t = time.perf_counter()
        s_test = evaluate.decision_scores(chosen["estimator"], feats[kind]["test"])
        test_m = evaluate.binary_metrics(test.y, s_test, chosen["threshold"])
        extra_m = {s.name: evaluate.binary_metrics(
            s.y, evaluate.decision_scores(chosen["estimator"], feats[kind][s.name]), chosen["threshold"])
            for s in extra}
        timings["test_predict_seconds"] = round(time.perf_counter() - t, 4)
        log.info("TEST %s: %s", best, {k: v for k, v in test_m.items() if k != "confusion_matrix"})
        log.info("TEST confusion matrix: %s", test_m["confusion_matrix"])

        names = data.CLASS_NAMES[cfg.label_scheme]
        s_val = evaluate.decision_scores(chosen["estimator"], feats[kind]["val"])
        fig_dir = run_dir / "figures"
        for fig in [evaluate.plot_curves(val.y, s_val, chosen["threshold"], f"{best} - validation",
                                         fig_dir / "val_curves.png"),
                    evaluate.plot_curves(test.y, s_test, chosen["threshold"], f"{best} - test",
                                         fig_dir / "test_curves.png"),
                    evaluate.plot_confusion(chosen["val"]["confusion_matrix"], names, "Validation",
                                            fig_dir / "val_confusion.png"),
                    evaluate.plot_confusion(test_m["confusion_matrix"], names, "Test",
                                            fig_dir / "test_confusion.png")]:
            plt.close(fig)

        model = Pipeline([("features", extractors[kind]), ("model", chosen["estimator"])])
        bundle = {"model": model, "threshold": chosen["threshold"], "label_scheme": cfg.label_scheme,
                  "class_names": names, "image_size": cfg.image_size, "side": train.side, "candidate": best}
        with open(run_dir / "model.pkl", "wb") as f:
            pickle.dump(bundle, f)
        timings["total_seconds"] = round(time.perf_counter() - t_start, 2)

        metrics = {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "data_root": str(data_root),
            "config": dataclasses.asdict(cfg),
            "environment": environment(),
            "data_profile": profile,
            "feature_dims": {k: int(v["train"].shape[1]) for k, v in feats.items()},
            "selected": {"candidate": best, "features": kind, "best_params": chosen["best_params"],
                         "threshold": chosen["threshold"], "threshold_mode": cfg.threshold_mode,
                         "target_recall": cfg.target_recall, "cv": chosen["cv"]},
            "validation": chosen["val"],
            "test": test_m,
            "extra_tests": extra_m,
            "candidates": {n: {k: v for k, v in s.items() if k != "estimator"} for n, s in searches.items()},
            "timings": timings,
        }
        save_json(metrics, run_dir / "metrics.json")
        save_json(chosen["best_params"], run_dir / "params.json")
        log.info("done in %.1fs; outputs in %s", timings["total_seconds"], run_dir)
        return Result(run_dir, metrics, model, chosen["threshold"])
    finally:
        logging.getLogger("kneeclf").removeHandler(handler)
        handler.close()
