"""Inference on single images: self-contained model export, checksum-verified loading, prediction, HOG display.

Security note: model files are Python pickles, and unpickling can execute arbitrary code. Only load model files
from a source you trust (your own runs or the project's GitHub releases), and keep the SHA-256 check enabled: when a
`<model>.sha256` file sits next to the model, `load_bundle` refuses a file whose hash does not match.
"""
from __future__ import annotations

import hashlib
import math
import os
import pickle
import platform
import sys
import urllib.request
from pathlib import Path

import numpy as np

from . import data, evaluate, portable

FORMAT = "kneeclf-model/1"
MODEL_FILENAME = "kneeclf-hog-logreg.pkl"
RELEASE_URL = "https://github.com/Ridwane777/medical-vision-ml/releases/latest/download/"
DISCLAIMER = ("Research/education demo, not a medical device: the output is not a diagnosis. "
              "Trained on one public dataset; ask a clinician to read real X-rays.")
NATIVE_SIDE = 224  # side of the training images before the pipeline's own downscale


class ChecksumError(RuntimeError):
    """The model file does not match its SHA-256 checksum file."""


def default_cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "medical-vision-ml"


def default_model_path():
    return default_cache_dir() / MODEL_FILENAME


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksum(path):
    """Write `<path>.sha256` in `sha256sum` format and return the hex digest."""
    path = Path(path)
    digest = sha256_file(path)
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def read_checksum(sha_path):
    text = Path(sha_path).read_text(encoding="utf-8").strip()
    digest = text.split()[0].lower() if text else ""
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ChecksumError(f"{sha_path}: not a SHA-256 checksum file")
    return digest


def verify_checksum(path, sha_path=None):
    """Raise ChecksumError on mismatch. Returns True if verified, False if no checksum file exists."""
    sha_path = Path(sha_path or f"{path}.sha256")
    if not sha_path.is_file():
        return False
    expected, actual = read_checksum(sha_path), sha256_file(path)
    if expected != actual:
        raise ChecksumError(f"{path}: SHA-256 {actual} does not match {sha_path} ({expected}); "
                            "the file is corrupted or was tampered with, do not load it")
    return True


def _versions():
    import sklearn
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "scikit-learn": sklearn.__version__}


def make_bundle(pipeline, threshold, label_scheme, side, hog_params, candidate, metrics=None):
    """Small self-contained dict holding everything `predict` needs (pipeline starts at the feature step)."""
    return {"format": FORMAT, "pipeline": pipeline, "threshold": float(threshold), "label_scheme": label_scheme,
            "class_names": list(data.CLASS_NAMES[label_scheme]), "image_size": int(side),
            "input_side": NATIVE_SIDE, "hog": dict(hog_params or {}), "candidate": candidate,
            "metrics": metrics or {}, "versions": _versions()}


def bundle_from_run(run_bundle, hog_params, metrics=None):
    """Convert the `model.pkl` dict saved by `train.run` into the export format."""
    return make_bundle(run_bundle["model"], run_bundle["threshold"], run_bundle["label_scheme"],
                       run_bundle["side"], hog_params, run_bundle.get("candidate"), metrics)


def export_bundle(bundle, path):
    """Pickle the bundle and write its `.sha256` next to it. Returns (path, digest, size in bytes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path, write_checksum(path), path.stat().st_size


def load_bundle(path, verify=True):
    """Load an exported model. Pickle executes code: only load files from a trusted source.

    The SHA-256 file next to the model, when present, is checked first (`verify=False` skips it).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path} (run `python -m kneeclf download-model`)")
    if verify:
        verify_checksum(path)
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict) or bundle.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} model file")
    return bundle


def preprocess(image, bundle):
    """Any PIL image / path / array -> flattened float32 row at the model's size, values in [0, 1].

    Same steps as training: grayscale, square 224 x 224 (resized with Pillow's area filter), then the
    pipeline's own block-mean downscale (224 -> 112). Shared with the portable JSON predictor.
    """
    return portable.preprocess(image, bundle)


def _probability(pipeline, X, scores):
    if hasattr(pipeline, "predict_proba"):
        try:
            return np.asarray(pipeline.predict_proba(X)[:, 1], dtype=float)
        except AttributeError:
            pass
    return 1.0 / (1.0 + np.exp(-scores))


def threshold_probability(bundle):
    """Decision threshold expressed on the probability scale (exact for logistic regression)."""
    return 1.0 / (1.0 + math.exp(-bundle["threshold"]))


def predict_one(image, bundle):
    X = preprocess(image, bundle)
    score = float(evaluate.decision_scores(bundle["pipeline"], X)[0])
    prob = float(_probability(bundle["pipeline"], X, np.array([score]))[0])
    positive = score >= bundle["threshold"]
    return {"score": round(prob, 4), "decision_score": round(score, 4),
            "threshold": round(threshold_probability(bundle), 4), "positive": bool(positive),
            "label": bundle["class_names"][int(positive)],
            "oa_signs": "yes" if positive else "no"}


def predict_images(paths, bundle):
    out = []
    for p in paths:
        try:
            out.append({"image": str(p), **predict_one(p, bundle)})
        except (OSError, ValueError) as e:  # unreadable or non-image file: report it, keep going
            out.append({"image": str(p), "error": str(e)})
    return out


def download_model(dest=None, base_url=RELEASE_URL, filename=MODEL_FILENAME, timeout=60):
    """Download the model and its .sha256 from the GitHub release, verify, then move into place."""
    dest = Path(dest) if dest else default_cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    target, sha_target = dest / filename, dest / f"{filename}.sha256"
    tmp, sha_tmp = dest / f"{filename}.part", dest / f"{filename}.sha256.part"
    try:
        for url, out in [(base_url + filename + ".sha256", sha_tmp), (base_url + filename, tmp)]:
            if not url.startswith("https://"):
                raise ValueError(f"refusing non-HTTPS URL {url}")
            with urllib.request.urlopen(url, timeout=timeout) as r, open(out, "wb") as f:  # noqa: S310
                while chunk := r.read(1 << 16):
                    f.write(chunk)
        verify_checksum(tmp, sha_tmp)
        os.replace(sha_tmp, sha_target)
        os.replace(tmp, target)
    finally:
        for p in (tmp, sha_tmp):
            if p.exists():
                p.unlink()
    return target


# HOG visualization lives in viz.py (no pickle, browser-friendly); re-exported here for compatibility.
from .viz import cell_histograms, draw_hog, hog_figure  # noqa: E402,F401
