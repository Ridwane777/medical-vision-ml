"""CLI for kneeclf.

    python -m kneeclf train --data ROOT [--config cfg.json] [--quick] [--n-jobs 4] [--output runs]
    python -m kneeclf --data ROOT ...            (old form, same as `train`)
    python -m kneeclf export RUN_DIR [--out dist/kneeclf-hog-logreg.pkl]
    python -m kneeclf download-model [--dest DIR]
    python -m kneeclf predict IMAGE [IMAGE ...] [--model PATH] [--json]
"""
import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the CLI only writes PNG files

from . import inference  # noqa: E402
from .train import Config, run  # noqa: E402

# Built-in preset for --quick: small sample, 64 px, tiny grids (same spirit as configs/quick.json).
QUICK = {"image_size": 64, "max_per_class": 150, "max_per_class_eval": 150, "cv_folds": 3,
         "candidates": ["raw_pca_logreg", "hog_logreg", "hog_svc"]}
QUICK_GRIDS = {"raw_pca_logreg": {"pca__n_components": [30], "logreg__C": [0.01]},
               "hog_logreg": {"logreg__C": [0.001, 0.01]},
               "hog_svc": {"svc__C": [1], "svc__gamma": ["scale"]}}
COMMANDS = ("train", "predict", "download-model", "export")


def _train_args(p):
    p.add_argument("--data", required=True, help="dataset root containing train/ val/ test/ with 0..4 subfolders")
    p.add_argument("--config", help="JSON file with Config fields (see configs/)")
    p.add_argument("--quick", action="store_true", help="small built-in preset (overrides sizes and grids)")
    p.add_argument("--n-jobs", type=int, help="parallel jobs for GridSearchCV (default from config: 4)")
    p.add_argument("--label-scheme", choices=["kl01_vs_2to4", "kl0_vs_1to4"])
    p.add_argument("--output", default="runs", help="parent folder of the run directory")
    p.add_argument("--quiet", action="store_true")


def cmd_train(a):
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
    return 0


def cmd_export(a):
    run_dir = Path(a.run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    with open(run_dir / "model.pkl", "rb") as f:   # your own run output (trusted)
        import pickle
        run_bundle = pickle.load(f)
    t, v = metrics["test"], metrics["validation"]
    keep = ("n", "recall", "specificity", "precision", "f2", "roc_auc", "pr_auc", "balanced_accuracy")
    summary = {"test": {k: t[k] for k in keep}, "validation": {k: v[k] for k in keep},
               "created": metrics.get("created"), "best_params": metrics["selected"]["best_params"]}
    bundle = inference.bundle_from_run(run_bundle, metrics["config"]["hog"], summary)
    path, digest, size = inference.export_bundle(bundle, a.out)
    print(json.dumps({"model": str(path), "sha256": digest, "bytes": size}, indent=2))
    return 0


def cmd_download(a):
    path = inference.download_model(a.dest)
    print(f"model saved and verified: {path}")
    return 0


def cmd_predict(a):
    try:
        bundle = inference.load_bundle(a.model)
    except (FileNotFoundError, inference.ChecksumError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    results = inference.predict_images(a.images, bundle)
    if a.json:
        print(json.dumps({"model": str(a.model), "threshold": round(inference.threshold_probability(bundle), 4),
                          "results": results, "disclaimer": inference.DISCLAIMER}, indent=2))
    else:
        for r in results:
            if "error" in r:
                print(f"{r['image']}: error: {r['error']}")
                continue
            print(f"{r['image']}: score {r['score']:.3f} (threshold {r['threshold']:.3f}) -> "
                  f"signs of osteoarthritis (KL>=2): {r['oa_signs']}")
        print(inference.DISCLAIMER)
    return 1 if any("error" in r for r in results) else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backward compatibility: `python -m kneeclf --data ROOT ...` (no subcommand) trains.
    if not argv or argv[0] not in COMMANDS + ("-h", "--help"):
        argv = ["train", *argv]
    p = argparse.ArgumentParser(prog="kneeclf", description="Binary knee X-ray classification (KL grades).")
    sub = p.add_subparsers(dest="command", required=True)
    _train_args(sub.add_parser("train", help="train and evaluate candidates (default command)"))
    pe = sub.add_parser("export", help="export a run's model to a small self-contained file + .sha256")
    pe.add_argument("run_dir")
    pe.add_argument("--out", default=f"dist/{inference.MODEL_FILENAME}")
    pd_ = sub.add_parser("download-model", help="download the released demo model and verify its SHA-256")
    pd_.add_argument("--dest", help=f"folder (default: {inference.default_cache_dir()})")
    pp = sub.add_parser("predict", help="score knee X-ray images (PNG/JPG)")
    pp.add_argument("images", nargs="+")
    pp.add_argument("--model", default=str(inference.default_model_path()),
                    help="exported model file (default: the download-model cache)")
    pp.add_argument("--json", action="store_true", help="machine-readable output")
    a = p.parse_args(argv)
    return {"train": cmd_train, "export": cmd_export, "download-model": cmd_download,
            "predict": cmd_predict}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
