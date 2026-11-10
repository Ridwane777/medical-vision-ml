"""Stateless image feature extractors (numpy only), usable as scikit-learn transformers.

Inputs are flattened square images: X has shape (n, side * side), values in [0, 1].
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _as_images(X):
    X = np.asarray(X, dtype=np.float32)
    side = math.isqrt(X.shape[1])
    if side * side != X.shape[1]:
        raise ValueError(f"{X.shape[1]} features is not a square image")
    return X.reshape(len(X), side, side)


class _Stateless(TransformerMixin, BaseEstimator):
    batch_size = 256

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        out = [self._transform_batch(_as_images(X[i:i + self.batch_size]))
               for i in range(0, len(X), self.batch_size)]
        return np.concatenate(out).astype(np.float32, copy=False)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags


class RawPixels(_Stateless):
    """Identity on flattened pixels (kept as an explicit step for readability)."""

    def _transform_batch(self, imgs):
        return imgs.reshape(len(imgs), -1)


class HOG(_Stateless):
    """Histogram of Oriented Gradients (Dalal & Triggs), vectorized over a batch of images.

    Central-difference gradients, unsigned orientations in [0, 180) split into `bins` bins with linear
    interpolation between neighbouring bins, per-cell histograms weighted by gradient magnitude, then
    overlapping blocks of block x block cells (stride one cell) normalized with L2-Hys (clip at 0.2).
    """

    def __init__(self, cell=8, block=2, bins=9, clip=0.2):
        self.cell = cell
        self.block = block
        self.bins = bins
        self.clip = clip

    def output_shape(self, side):
        c = side // self.cell
        nb = c - self.block + 1
        return nb, nb, self.block * self.block * self.bins

    def _transform_batch(self, imgs):
        n, h, w = imgs.shape
        gx = np.zeros_like(imgs)
        gy = np.zeros_like(imgs)
        gx[:, :, 1:-1] = imgs[:, :, 2:] - imgs[:, :, :-2]
        gy[:, 1:-1, :] = imgs[:, 2:, :] - imgs[:, :-2, :]
        mag = np.hypot(gx, gy)
        ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0
        pos = ang / (180.0 / self.bins) - 0.5
        b0 = np.floor(pos)
        w1 = (pos - b0).astype(np.float32)
        b0 = b0.astype(np.int64) % self.bins
        b1 = (b0 + 1) % self.bins

        c = self.cell
        ch, cw = h // c, w // c
        if ch < self.block or cw < self.block:
            raise ValueError(f"image {h}x{w} too small for cell={c}, block={self.block}")
        sl = (slice(None), slice(0, ch * c), slice(0, cw * c))
        mag, w1, b0, b1 = mag[sl], w1[sl], b0[sl], b1[sl]
        m0, m1 = mag * (1 - w1), mag * w1
        hist = np.empty((n, ch, cw, self.bins), dtype=np.float32)
        for b in range(self.bins):
            contrib = np.where(b0 == b, m0, 0) + np.where(b1 == b, m1, 0)
            hist[..., b] = contrib.reshape(n, ch, c, cw, c).sum(axis=(2, 4))

        k = self.block
        bh, bw = ch - k + 1, cw - k + 1
        blocks = np.concatenate([hist[:, i:i + bh, j:j + bw] for i in range(k) for j in range(k)], axis=-1)
        eps = 1e-6
        blocks = blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + eps ** 2)
        blocks = np.minimum(blocks, self.clip)
        blocks = blocks / np.sqrt((blocks ** 2).sum(-1, keepdims=True) + eps ** 2)
        return blocks.reshape(n, -1)


class LBPHistogram(_Stateless):
    """Normalized 256-bin histogram of 8-neighbour local binary patterns (texture), optionally per grid cell."""

    def __init__(self, grid=1):
        self.grid = grid

    def _transform_batch(self, imgs):
        n, h, w = imgs.shape
        c = imgs[:, 1:-1, 1:-1]
        codes = np.zeros(c.shape, dtype=np.int64)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        for bit, (dy, dx) in enumerate(offsets):
            nb = imgs[:, 1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
            codes |= (nb >= c).astype(np.int64) << bit
        g = self.grid
        hs, ws = codes.shape[1] // g, codes.shape[2] // g
        feats = []
        for i in range(g):
            for j in range(g):
                cell = codes[:, i * hs:(i + 1) * hs, j * ws:(j + 1) * ws].reshape(n, -1)
                rows = np.repeat(np.arange(n), cell.shape[1])
                hist = np.bincount(rows * 256 + cell.ravel(), minlength=n * 256).reshape(n, 256)
                feats.append(hist / cell.shape[1])
        return np.concatenate(feats, axis=1)


class IntensityHistogram(_Stateless):
    def __init__(self, bins=32):
        self.bins = bins

    def _transform_batch(self, imgs):
        idx = np.clip((imgs.reshape(len(imgs), -1) * self.bins).astype(np.int64), 0, self.bins - 1)
        rows = np.repeat(np.arange(len(imgs)), idx.shape[1])
        hist = np.bincount(rows * self.bins + idx.ravel(), minlength=len(imgs) * self.bins)
        return hist.reshape(len(imgs), self.bins) / idx.shape[1]


def make_extractor(kind, hog_params=None):
    """Feature extractor for a feature kind: raw, hog, lbp, hist, or a '+'-joined combination."""
    hog_params = hog_params or {}
    parts = kind.split("+")
    factories = {"raw": RawPixels, "hog": lambda: HOG(**hog_params), "lbp": lambda: LBPHistogram(grid=4),
                 "hist": IntensityHistogram}
    unknown = [p for p in parts if p not in factories]
    if unknown:
        raise ValueError(f"unknown feature kind(s) {unknown}")
    if len(parts) == 1:
        return factories[parts[0]]()
    from sklearn.pipeline import FeatureUnion
    return FeatureUnion([(p, factories[p]()) for p in parts])
