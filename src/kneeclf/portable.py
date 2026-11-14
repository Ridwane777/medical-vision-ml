"""Portable demo model: JSON export and a pure-numpy predictor (no pickle, no scikit-learn at prediction time).

The exported pipeline is HOG -> StandardScaler -> LogisticRegression. This module reproduces its
`decision_function` from the plain numbers stored in `kneeclf-hog-logreg.json`, so the model can run anywhere
numpy and Pillow run, including the static web demo (demo/, JavaScript port in demo/knee.js).

Only numpy, Pillow and the standard library are imported here, on purpose.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

JSON_FORMAT = "kneeclf-portable/1"
JSON_FILENAME = "kneeclf-hog-logreg.json"
NATIVE_SIDE = 224


# ---- HOG (single source of truth: features.HOG calls this function) ----

def hog_batch(imgs, cell=8, block=2, bins=9, clip=0.2):
    """HOG of a batch of square images, shape (n, h, w) float32 -> (n, n_features) float32.

    Central-difference gradients, unsigned orientations in [0, 180) split into `bins` bins with linear
    interpolation between neighbouring bins, per-cell histograms weighted by gradient magnitude, then
    overlapping blocks of block x block cells (stride one cell) normalized with L2-Hys (clip at `clip`).
    """
    imgs = np.asarray(imgs, dtype=np.float32)
    n, h, w = imgs.shape
    gx = np.zeros_like(imgs)
    gy = np.zeros_like(imgs)
    gx[:, :, 1:-1] = imgs[:, :, 2:] - imgs[:, :, :-2]
    gy[:, 1:-1, :] = imgs[:, 2:, :] - imgs[:, :-2, :]
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    pos = ang / (180.0 / bins) - 0.5
    b0 = np.floor(pos)
    w1 = (pos - b0).astype(np.float32)
    b0 = b0.astype(np.int64) % bins
    b1 = (b0 + 1) % bins

    c = cell
    ch, cw = h // c, w // c
    if ch < block or cw < block:
        raise ValueError(f"image {h}x{w} too small for cell={c}, block={block}")
    sl = (slice(None), slice(0, ch * c), slice(0, cw * c))
    mag, w1, b0, b1 = mag[sl], w1[sl], b0[sl], b1[sl]
    m0, m1 = mag * (1 - w1), mag * w1
    hist = np.empty((n, ch, cw, bins), dtype=np.float32)
    for b in range(bins):
        contrib = np.where(b0 == b, m0, 0) + np.where(b1 == b, m1, 0)
        hist[..., b] = contrib.reshape(n, ch, c, cw, c).sum(axis=(2, 4))

    k = block
    bh, bw = ch - k + 1, cw - k + 1
    blocks = np.concatenate([hist[:, i:i + bh, j:j + bw] for i in range(k) for j in range(k)], axis=-1)
    eps = 1e-6
    blocks = blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + eps ** 2)
    blocks = np.minimum(blocks, clip)
    blocks = blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + eps ** 2)
    return blocks.reshape(n, -1).astype(np.float32, copy=False)


# ---- preprocessing (same steps as training) ----

def downscale(img, size):
    """Area-average downscale of a square float array (exact block mean when size divides the side)."""
    h, w = img.shape
    if size is None or size == h:
        return img
    if h != w or size > h:
        raise ValueError(f"cannot downscale {h}x{w} to {size}x{size}")
    if h % size == 0:
        f = h // size
        return img.reshape(size, f, size, f).mean(axis=(1, 3), dtype=np.float32)
    resized = Image.fromarray(img.astype(np.float32), mode="F").resize((size, size), Image.Resampling.BOX)
    return np.asarray(resized, dtype=np.float32)


def preprocess(image, model):
    """Any PIL image / path / array -> flattened float32 row at the model's size, values in [0, 1].

    Grayscale, square input_side x input_side (Pillow area filter), then block-mean downscale to image_size.
    `model` is the JSON model dict (or a pickle bundle: both carry `input_side` and `image_size`).
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            im.load()
            image = im.copy()
    if isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
        if arr.max() > 1.0:
            arr = arr / 255.0
        image = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
    side_in = model.get("input_side", NATIVE_SIDE)
    gray = image.convert("L")
    if gray.size != (side_in, side_in):
        gray = gray.resize((side_in, side_in), Image.Resampling.BOX)
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    return downscale(arr, model["image_size"]).reshape(1, -1)


# ---- JSON model ----

def load_model(path_or_dict):
    """Read and validate a JSON model (a path, a JSON string's parsed dict, or a dict)."""
    m = path_or_dict
    if not isinstance(m, dict):
        m = json.loads(Path(m).read_text(encoding="utf-8"))
    if m.get("format") != JSON_FORMAT:
        raise ValueError(f"not a {JSON_FORMAT} model")
    out = dict(m)
    out["_coef"] = np.asarray(m["logreg"]["coef"], dtype=np.float64)
    out["_intercept"] = float(m["logreg"]["intercept"])
    sc = m.get("scaler")
    out["_mean"] = None if sc is None or sc.get("mean") is None else np.asarray(sc["mean"], dtype=np.float64)
    out["_scale"] = None if sc is None or sc.get("scale") is None else np.asarray(sc["scale"], dtype=np.float64)
    if out["_coef"].shape[0] != m["n_features"]:
        raise ValueError("coef length does not match n_features")
    return out


def features(X, model):
    """Flattened images (n, side * side) -> HOG features (n, n_features) float32."""
    X = np.asarray(X, dtype=np.float32)
    side = math.isqrt(X.shape[1])
    h = model["hog"]
    return hog_batch(X.reshape(len(X), side, side), h["cell"], h["block"], h["bins"], h.get("clip", 0.2))


def decision_function(X, model):
    """Same value as the scikit-learn pipeline's decision_function on flattened images X."""
    F = features(X, model)  # float32, like the pipeline
    # StandardScaler keeps float32 input in float32 (its statistics are cast to the input dtype).
    if model["_mean"] is not None:
        F -= model["_mean"].astype(np.float32)
    if model["_scale"] is not None:
        F /= model["_scale"].astype(np.float32)
    # Same arithmetic as scikit-learn: (n, d) @ (d, 1) in the dtype the model was fitted in (float32 here,
    # because the HOG features are float32), then the intercept in that dtype.
    dt = np.dtype(model["logreg"].get("dtype", "float64"))
    z = (F @ model["_coef"].astype(dt)[:, None]).ravel() + np.asarray(model["_intercept"], dtype=dt)
    return z.astype(np.float64)


def threshold_probability(model):
    return 1.0 / (1.0 + math.exp(-model["threshold"]))


def predict_one(image, model):
    """Same output dict as `inference.predict_one` (score = logistic probability)."""
    X = preprocess(image, model)
    score = float(decision_function(X, model)[0])
    prob = 1.0 / (1.0 + math.exp(-score))
    positive = score >= model["threshold"]
    return {"score": round(prob, 4), "decision_score": round(score, 4),
            "threshold": round(threshold_probability(model), 4), "positive": bool(positive),
            "label": model["class_names"][int(positive)], "oa_signs": "yes" if positive else "no"}


def export_dict(bundle):
    """Pickle bundle (HOG -> [StandardScaler] -> LogisticRegression pipeline) -> JSON-serializable dict.

    Uses only attributes of the fitted objects (no scikit-learn import), and refuses any other pipeline shape.
    """
    pipe = bundle["pipeline"]
    steps = list(pipe.named_steps.items())
    hog = steps[0][1]
    if type(hog).__name__ != "HOG":
        raise ValueError(f"first step must be HOG, got {type(hog).__name__}")
    rest = steps[1][1] if len(steps) == 2 and hasattr(steps[1][1], "named_steps") else None
    inner = list(rest.named_steps.values()) if rest is not None else [s for _, s in steps[1:]]
    scaler, logreg = (inner[0], inner[1]) if len(inner) == 2 else (None, inner[0])
    if type(logreg).__name__ != "LogisticRegression" or (scaler is not None and
                                                         type(scaler).__name__ != "StandardScaler"):
        raise ValueError(f"unsupported pipeline: {pipe}")
    if logreg.coef_.shape[0] != 1:
        raise ValueError("binary logistic regression expected")
    sc = None
    if scaler is not None:
        sc = {"mean": None if scaler.mean_ is None else [float(v) for v in scaler.mean_],
              "scale": None if scaler.scale_ is None else [float(v) for v in scaler.scale_]}
    coef = [float(v) for v in logreg.coef_[0]]
    return {"format": JSON_FORMAT, "candidate": bundle.get("candidate"), "label_scheme": bundle["label_scheme"],
            "class_names": list(bundle["class_names"]), "image_size": int(bundle["image_size"]),
            "input_side": int(bundle.get("input_side", NATIVE_SIDE)),
            "hog": {"cell": int(hog.cell), "block": int(hog.block), "bins": int(hog.bins), "clip": float(hog.clip)},
            "n_features": len(coef), "scaler": sc,
            "logreg": {"coef": coef, "intercept": float(logreg.intercept_[0]), "C": float(logreg.C),
                       "class_weight": logreg.class_weight,
                       "dtype": str(logreg.coef_.dtype)},
            "threshold": float(bundle["threshold"]),
            "threshold_probability": threshold_probability(bundle),
            "metrics": bundle.get("metrics", {}), "trained_with": bundle.get("versions", {})}


def export_json(bundle, path):
    """Write the JSON model. Returns (path, size in bytes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(export_dict(bundle), separators=(",", ":")), encoding="utf-8")
    return path, path.stat().st_size
