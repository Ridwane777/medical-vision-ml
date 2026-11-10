"""Image loading: Kellgren-Lawrence grade folders -> float32 arrays in [0, 1] with a binary label."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

GRADES = (0, 1, 2, 3, 4)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Binary label (0 = negative, 1 = positive) for each KL grade.
LABEL_SCHEMES = {
    # Previous notebook: any sign of OA (KL >= 1) is "sick".
    "kl0_vs_1to4": {0: 0, 1: 1, 2: 1, 3: 1, 4: 1},
    # Clinical / radiographic definition of OA: KL >= 2 (definite osteophyte).
    "kl01_vs_2to4": {0: 0, 1: 0, 2: 1, 3: 1, 4: 1},
}
CLASS_NAMES = {
    "kl0_vs_1to4": ("healthy (KL0)", "OA (KL1-4)"),
    "kl01_vs_2to4": ("no OA (KL0-1)", "OA (KL2-4)"),
}


@dataclass
class Split:
    name: str
    X: np.ndarray            # (n, side * side) float32 in [0, 1]
    y: np.ndarray            # (n,) int8 binary label
    grades: np.ndarray       # (n,) int8 KL grade
    paths: list
    side: int
    profile: dict = field(default_factory=dict)


def list_split(root, split):
    """Return [(path, grade)] for <root>/<split>/<grade>/*.<image>, sorted for determinism."""
    base = Path(root) / split
    if not base.is_dir():
        raise FileNotFoundError(f"missing split folder: {base}")
    items = []
    for g in GRADES:
        d = base / str(g)
        if d.is_dir():
            items += [(p, g) for p in sorted(d.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not items:
        raise FileNotFoundError(f"no images under {base}/{{0..4}}/")
    return items


def downscale(img, size):
    """Area-average downscale of a square (h, w) float array to (size, size).

    Exact numpy block mean when size divides h (224 -> 112, 56, 28); otherwise Pillow's BOX filter,
    which is the same area average with fractional box edges (224 -> 64).
    """
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


def load_image(path, size=None):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    return downscale(arr, size)


def _sample(idx, n, rng):
    return idx if len(idx) <= n else np.sort(rng.choice(idx, size=n, replace=False))


def select_indices(labels, balance=False, max_per_class=None, seed=0):
    """Indices kept after optional undersampling to the minority class and an optional per-class cap."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    cap = None
    if balance:
        cap = min(int((labels == c).sum()) for c in classes)
    if max_per_class is not None:
        cap = max_per_class if cap is None else min(cap, max_per_class)
    if cap is None:
        return np.arange(len(labels))
    return np.sort(np.concatenate([_sample(np.flatnonzero(labels == c), cap, rng) for c in classes]))


def load_split(root, split, label_scheme="kl01_vs_2to4", image_size=None, balance=False,
               max_per_class=None, seed=0):
    """Load one split. Sampling is decided on file names before any pixel is read (memory-aware)."""
    if label_scheme not in LABEL_SCHEMES:
        raise ValueError(f"unknown label_scheme {label_scheme!r}; choose from {sorted(LABEL_SCHEMES)}")
    mapping = LABEL_SCHEMES[label_scheme]
    items = list_split(root, split)
    grades_all = np.array([g for _, g in items], dtype=np.int8)
    labels_all = np.array([mapping[int(g)] for g in grades_all], dtype=np.int8)
    keep = select_indices(labels_all, balance=balance, max_per_class=max_per_class, seed=seed)

    first = load_image(items[keep[0]][0], image_size)
    side = first.shape[0]
    X = np.empty((len(keep), side * side), dtype=np.float32)
    for row, i in enumerate(keep):
        img = load_image(items[i][0], image_size)
        if img.shape != (side, side):
            raise ValueError(f"{items[i][0]}: shape {img.shape} differs from {(side, side)}")
        X[row] = img.ravel()
        if (row + 1) % 1000 == 0:
            log.info("%s: loaded %d/%d images", split, row + 1, len(keep))
    y, grades = labels_all[keep], grades_all[keep]
    profile = {
        "split": split,
        "files_found": len(items),
        "grades_found": {str(g): int((grades_all == g).sum()) for g in GRADES},
        "kept": int(len(keep)),
        "grades_kept": {str(g): int((grades == g).sum()) for g in GRADES},
        "labels_kept": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "balanced": bool(balance),
        "max_per_class": max_per_class,
        "side": int(side),
        "memory_mb": round(X.nbytes / 2**20, 1),
    }
    log.info("%s: %d images (%s), %dx%d px, %.1f MB", split, len(keep), profile["labels_kept"], side, side,
             profile["memory_mb"])
    return Split(split, X, y, grades, [str(items[i][0]) for i in keep], side, profile)
