"""Beyond the specification: frozen ImageNet CNNs as feature extractors for the knee task.

Uses torch/torchvision, which the project specification forbids, so this lives outside src/.
Same protocol as src/kneeclf: label scheme kl01_vs_2to4, balanced train, real-ratio val/test,
GridSearchCV on train, threshold for recall 0.85 on validation, test touched once for the
candidate selected on validation ROC-AUC.

Run from the repository root:
    PYTHONPATH=src:. python beyond_scope/knee_cnn_embeddings.py --data data/knee
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kneeclf import data, evaluate, models  # noqa: E402

log = logging.getLogger("beyond.knee")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
BACKBONES = {
    "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1),
}
GRIDS = {
    "logreg": {"logreg__C": [0.001, 0.01, 0.1, 1]},
    "svc": {"svc__C": [1, 10], "svc__gamma": ["scale", 0.0001]},
}


def load_backbone(name):
    ctor, weights = BACKBONES[name]
    net = ctor(weights=weights)
    if name == "resnet50":
        net.fc = torch.nn.Identity()          # 2048-d global-average-pooled features
    else:
        net.classifier = torch.nn.Identity()  # 1280-d
    n_params = sum(p.numel() for p in net.parameters())
    return net.eval(), n_params


def to_tensor(X, side):
    """(n, side*side) floats in [0, 1] -> (n, 3, side, side) ImageNet-normalized."""
    t = torch.from_numpy(X.reshape(-1, 1, side, side)).repeat(1, 3, 1, 1)
    return (t - MEAN) / STD


@torch.no_grad()
def embed(net, split, batch=64):
    out = []
    for i in range(0, len(split.X), batch):
        out.append(net(to_tensor(split.X[i:i + batch], split.side)).numpy())
    return np.concatenate(out).astype(np.float32)


def head(name, seed):
    clf = (LogisticRegression(max_iter=5000, class_weight="balanced") if name == "logreg"
           else SVC(kernel="rbf", class_weight="balanced", cache_size=1000))
    return Pipeline([("scale", StandardScaler()), (name, clf)])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/knee")
    p.add_argument("--out", default="results/beyond_scope/knee")
    p.add_argument("--cache", default="runs/beyond_cache")
    p.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(a.threads)
    out, cache = Path(a.out), Path(a.cache)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    timings, sizes = {}, {}

    load = dict(label_scheme="kl01_vs_2to4", image_size=None, seed=a.seed)   # native 224 px
    t = time.perf_counter()
    splits = {"train": data.load_split(a.data, "train", balance=True, **load),
              "val": data.load_split(a.data, "val", **load),
              "test": data.load_split(a.data, "test", **load)}
    timings["load_seconds"] = round(time.perf_counter() - t, 1)

    feats = {}
    for bb in a.backbones:
        net, n_params = load_backbone(bb)
        sizes[bb] = {"parameters": n_params, "weights_mb_fp32": round(n_params * 4 / 2**20, 1)}
        feats[bb] = {}
        t = time.perf_counter()
        for name, s in splits.items():
            f = cache / f"{bb}_{name}.npy"
            if f.exists():
                feats[bb][name] = np.load(f)
            else:
                ts = time.perf_counter()
                feats[bb][name] = embed(net, s)
                np.save(f, feats[bb][name])
                log.info("%s %s: %s in %.1fs", bb, name, feats[bb][name].shape, time.perf_counter() - ts)
        timings[f"embed_{bb}_seconds"] = round(time.perf_counter() - t, 1)
        sizes[bb]["embedding_dims"] = int(feats[bb]["train"].shape[1])

    y = {k: s.y for k, s in splits.items()}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=a.seed)
    rows, chosen = [], {}
    for bb in a.backbones:
        for h in GRIDS:
            name = f"{bb}_{h}"
            gs = GridSearchCV(head(h, a.seed), GRIDS[h], scoring=models.SCORING, refit="roc_auc", cv=cv,
                              n_jobs=a.threads, error_score="raise")
            t = time.perf_counter()
            gs.fit(feats[bb]["train"], y["train"])
            sec = round(time.perf_counter() - t, 1)
            s_val = evaluate.decision_scores(gs.best_estimator_, feats[bb]["val"])
            thr = evaluate.choose_threshold(y["val"], s_val, "target_recall", 0.85)
            vm = evaluate.binary_metrics(y["val"], s_val, thr)
            chosen[name] = (gs.best_estimator_, thr, vm, bb, gs.best_params_, s_val)
            rows.append({"candidate": name, "features": bb,
                         "cv_roc_auc": round(float(gs.best_score_), 4), "val_roc_auc": vm["roc_auc"],
                         "val_pr_auc": vm["pr_auc"], "val_recall": vm["recall"],
                         "val_specificity": vm["specificity"], "val_f2": vm["f2"],
                         "val_balanced_accuracy": vm["balanced_accuracy"], "search_seconds": sec,
                         "best_params": json.dumps(gs.best_params_)})
            log.info("%s val: roc_auc=%.4f specificity=%.4f (%ss)", name, vm["roc_auc"], vm["specificity"], sec)

    table = evaluate.comparison_table(rows)
    table.to_csv(out / "comparison.csv", index=False)
    (out / "comparison.md").write_text(evaluate.to_markdown(table), encoding="utf-8")
    best = table.loc[0, "candidate"]
    est, thr, vm, bb, params, s_val = chosen[best]
    # Test set touched once, for the candidate selected on validation.
    s_test = evaluate.decision_scores(est, feats[bb]["test"])
    tm = evaluate.binary_metrics(y["test"], s_test, thr)
    log.info("TEST %s: %s", best, tm)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = data.CLASS_NAMES["kl01_vs_2to4"]
    for fig in [evaluate.plot_curves(y["test"], s_test, thr, f"{best} - test", out / "figures/test_curves.png"),
                evaluate.plot_confusion(tm["confusion_matrix"], names, "Test", out / "figures/test_confusion.png"),
                evaluate.plot_curves(y["val"], s_val, thr, f"{best} - validation", out / "figures/val_curves.png")]:
        plt.close(fig)
    timings["total_seconds"] = round(time.perf_counter() - t0, 1)
    metrics = {"protocol": "kl01_vs_2to4, balanced train, threshold for recall 0.85 on val, test once",
               "environment": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                               "threads": a.threads, "device": "cpu"},
               "data_profile": {k: s.profile for k, s in splits.items()},
               "backbones": sizes, "selected": {"candidate": best, "best_params": params, "threshold": thr},
               "validation": vm, "test": tm, "timings": timings}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    log.info("done in %.1fs", timings["total_seconds"])


if __name__ == "__main__":
    main()
