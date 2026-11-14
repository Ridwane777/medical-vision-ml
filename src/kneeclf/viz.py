"""HOG visualization (same code as notebook 01, simplified HOG without block normalisation).

numpy + matplotlib only; works with the JSON model dict or the pickle bundle.
"""
from __future__ import annotations

import numpy as np

from .portable import preprocess


def cell_histograms(image, cell=8, bins=9):
    """Per-cell orientation histograms weighted by gradient magnitude, shape (n, n, bins)."""
    image = np.asarray(image, dtype=np.float32)
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    gy[1:-1, :] = image[2:, :] - image[:-2, :]
    mag, ori = np.hypot(gx, gy), np.rad2deg(np.arctan2(gy, gx)) % 180
    n = image.shape[0] // cell
    H = np.zeros((n, n, bins))
    for i in range(n):
        for j in range(n):
            sl = (slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell))
            H[i, j] = np.histogram(ori[sl], bins=bins, range=(0, 180), weights=mag[sl])[0]
    return H


def draw_hog(ax, H, cell=8):
    """One short line per bin, drawn along the edge (perpendicular to the gradient); longer = stronger."""
    from matplotlib.collections import LineCollection
    ax.set_facecolor("black")
    top = H.max() or 1.0
    bins = H.shape[2]
    segs = []
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            for b, w in enumerate(H[i, j]):
                theta = np.deg2rad(b * 180 / bins + 90 / bins + 90)
                dx, dy = np.cos(theta) * w / top * cell / 2, np.sin(theta) * w / top * cell / 2
                cx, cy = j * cell + cell / 2, i * cell + cell / 2
                segs.append([(cx - dx, cy - dy), (cx + dx, cy + dy)])
    ax.add_collection(LineCollection(segs, colors="white", linewidths=0.8))
    ax.set_xlim(0, H.shape[1] * cell)
    ax.set_ylim(H.shape[0] * cell, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def hog_figure(image, model):
    """Matplotlib figure: the model input (grayscale, model size) and its HOG cells."""
    import matplotlib.pyplot as plt
    side = model["image_size"]
    img = preprocess(image, model).reshape(side, side)
    cell, bins = model["hog"].get("cell", 8), model["hog"].get("bins", 9)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"model input ({side} x {side})")
    axes[0].axis("off")
    draw_hog(axes[1], cell_histograms(img, cell, bins), cell)
    axes[1].set_title(f"HOG cells ({cell} px, {bins} bins)")
    fig.tight_layout()
    return fig
