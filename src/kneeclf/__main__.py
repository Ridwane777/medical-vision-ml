"""CLI: python -m kneeclf --data ROOT [--config cfg.json] [--quick] [--n-jobs 4] [--output runs]."""
import argparse
import dataclasses
import json
import logging
import sys

import matplotlib

matplotlib.use("Agg")  # the CLI only writes PNG files

from .train import Config, run  # noqa: E402

# Built-in preset for --quick: small sample, 64 px, tiny grids (same spirit as configs/quick.json).
QUICK = {"image_size": 64, "max_per_class": 150, "max_per_class_eval": 150, "cv_folds": 3,
         "candidates": ["raw_pca_logreg", "hog_logreg", "hog_svc"]}
QUICK_GRIDS = {"raw_pca_logreg": {"pca__n_components": [30], "logreg__C": [0.01]},
               "hog_logreg": {"logreg__C": [0.001, 0.01]},
               "hog_svc": {"svc__C": [1], "svc__gamma": ["scale"]}}


def main(argv=None):
    p = argparse.ArgumentParser(prog="kneeclf", description="Binary knee X-ray classification (KL grades).")
    p.add_argument("--data", required=True, help="dataset root containing train/ val/ test/ with 0..4 subfolders")
    p.add_argument("--config", help="JSON file with Config fields (see configs/)")
    p.add_argument("--quick", action="store_true", help="small built-in preset (overrides sizes and grids)")
    p.add_argument("--n-jobs", type=int, help="parallel jobs for GridSearchCV (default from config: 4)")
    p.add_argument("--label-scheme", choices=["kl01_vs_2to4", "kl0_vs_1to4"])
    p.add_argument("--output", default="runs", help="parent folder of the run directory")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = Config.from_json(a.config) if a.config else Config()
    if a.quick:
        cfg = dataclasses.replace(cfg, **QUICK, param_grids={**cfg.param_grids, **QUICK_GRIDS})
    if a.n_jobs is not None:
        cfg.n_jobs = a.n_jobs
    if a.label_scheme:
        cfg.label_scheme = a.label_scheme
    res = run(a.data, cfg, a.output)
    t = res.metrics["test"]
    print(json.dumps({"run_dir": str(res.run_dir), "selected": res.metrics["selected"]["candidate"],
                      "threshold": round(res.threshold, 4),
                      "test": {k: t[k] for k in ("recall", "specificity", "precision", "f2", "roc_auc", "pr_auc",
                                                 "balanced_accuracy")},
                      "total_seconds": res.metrics["timings"]["total_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
