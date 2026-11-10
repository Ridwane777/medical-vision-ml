"""Tests for kneeclf on synthetic X-ray-like PNGs written to a temporary <root>/<split>/<grade>/ tree."""
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kneeclf import Config, run  # noqa: E402
from kneeclf.data import downscale, load_split, select_indices  # noqa: E402
from kneeclf.evaluate import binary_metrics, choose_threshold  # noqa: E402
from kneeclf.features import HOG, IntensityHistogram, LBPHistogram  # noqa: E402
from kneeclf.train import load_model, predict  # noqa: E402

logging.disable(logging.CRITICAL)

SIDE = 32
# Images per grade for each split: imbalanced on purpose (grade 0 dominates, grade 4 is rare).
COUNTS = {"train": {0: 12, 1: 6, 2: 8, 3: 5, 4: 3}, "val": {0: 5, 1: 3, 2: 3, 3: 2, 4: 2},
          "test": {0: 5, 1: 3, 2: 3, 3: 2, 4: 2}}


def synthetic_image(grade, rng):
    """Smooth background; higher grades get stronger horizontal edges (a crude 'joint space' signal)."""
    img = rng.normal(0.45, 0.05, (SIDE, SIDE))
    if grade >= 2:
        for r in range(4, SIDE, 6):
            img[r:r + 2] += 0.1 * grade
    return np.clip(img, 0, 1)


def write_dataset(root, counts=COUNTS, seed=0):
    rng = np.random.default_rng(seed)
    for split, per_grade in counts.items():
        for g, n in per_grade.items():
            d = Path(root) / split / str(g)
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                Image.fromarray((synthetic_image(g, rng) * 255).astype(np.uint8), mode="L").save(d / f"{i:03d}.png")


class Tmp(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.root = Path(self._d.name) / "data"
        write_dataset(self.root)

    def tearDown(self):
        self._d.cleanup()


class TestData(Tmp):
    def test_loader_shapes_range_and_schemes(self):
        s = load_split(self.root, "train", label_scheme="kl01_vs_2to4")
        self.assertEqual(s.X.shape, (34, SIDE * SIDE))
        self.assertEqual(s.X.dtype, np.float32)
        self.assertTrue(0.0 <= s.X.min() and s.X.max() <= 1.0)
        self.assertEqual(int(s.y.sum()), 8 + 5 + 3)                     # grades 2-4 positive
        self.assertTrue(np.all(s.y == (s.grades >= 2)))
        old = load_split(self.root, "train", label_scheme="kl0_vs_1to4")
        self.assertEqual(int(old.y.sum()), 6 + 8 + 5 + 3)               # grades 1-4 positive
        with self.assertRaises(ValueError):
            load_split(self.root, "train", label_scheme="nope")

    def test_balancing_and_caps_are_deterministic(self):
        a = load_split(self.root, "train", balance=True, seed=3)
        b = load_split(self.root, "train", balance=True, seed=3)
        self.assertEqual(a.paths, b.paths)
        self.assertEqual(a.profile["labels_kept"], {"0": 16, "1": 16})   # 18 negatives undersampled to 16
        c = load_split(self.root, "train", balance=True, max_per_class=5)
        self.assertEqual(c.profile["labels_kept"], {"0": 5, "1": 5})
        idx = select_indices(np.array([0] * 10 + [1] * 3), balance=False)
        self.assertEqual(len(idx), 13)

    def test_downscale_block_mean(self):
        img = np.arange(16, dtype=np.float32).reshape(4, 4)
        np.testing.assert_allclose(downscale(img, 2), [[2.5, 4.5], [10.5, 12.5]])
        self.assertEqual(downscale(np.ones((224, 224), np.float32), 64).shape, (64, 64))
        s = load_split(self.root, "val", image_size=16)
        self.assertEqual(s.X.shape[1], 256)

    def test_balancing_applies_to_train_only(self):
        cfg = Config(image_size=None, candidates=["hog_logreg"], param_grids={"hog_logreg": {"logreg__C": [0.1]}},
                     cv_folds=2, n_jobs=1, verbose=0, hog={"cell": 8, "block": 2, "bins": 9})
        res = run(self.root, cfg, Path(self._d.name) / "runs")
        prof = res.metrics["data_profile"]
        self.assertEqual(prof["train"]["labels_kept"]["0"], prof["train"]["labels_kept"]["1"])
        self.assertEqual(prof["val"]["kept"], prof["val"]["files_found"])    # untouched
        self.assertEqual(prof["test"]["kept"], prof["test"]["files_found"])
        self.assertEqual(res.metrics["test"]["n"], 15)


class TestFeatures(unittest.TestCase):
    def test_hog_shape_and_constant_image(self):
        hog = HOG(cell=8, block=2, bins=9)
        X = np.random.default_rng(0).random((3, 64 * 64), dtype=np.float32)
        out = hog.transform(X)
        self.assertEqual(out.shape, (3, 7 * 7 * 36))
        self.assertEqual(hog.output_shape(64), (7, 7, 36))
        self.assertTrue(np.all(out >= 0) and np.all(out <= 1.0 + 1e-5))
        const = hog.transform(np.full((2, 64 * 64), 0.7, dtype=np.float32))
        self.assertTrue(np.allclose(const, 0))                       # no gradient anywhere

    def test_hog_orientation_and_contrast_invariance(self):
        ramp_x = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (32, 1))
        hog = HOG(cell=8, block=2, bins=9)
        hx = hog.transform(ramp_x.reshape(1, -1)).reshape(3, 3, 4, 9)
        hy = hog.transform(ramp_x.T.reshape(1, -1)).reshape(3, 3, 4, 9)
        # horizontal gradient (0 deg) -> bins 0 and 8 share it; vertical (90 deg) -> bin 4
        self.assertEqual(int(np.argmax(hy[1, 1, 0])), 4)
        self.assertIn(int(np.argmax(hx[1, 1, 0])), (0, 8))
        np.testing.assert_allclose(hog.transform((0.5 * ramp_x).reshape(1, -1)),
                                   hog.transform(ramp_x.reshape(1, -1)), atol=1e-4)

    def test_histograms_sum_to_one(self):
        X = np.random.default_rng(1).random((2, 32 * 32), dtype=np.float32)
        np.testing.assert_allclose(IntensityHistogram(16).transform(X).sum(1), 1, atol=1e-6)
        lbp = LBPHistogram(grid=2).transform(X)
        self.assertEqual(lbp.shape, (2, 4 * 256))
        np.testing.assert_allclose(lbp.sum(1), 4, atol=1e-6)


class TestThreshold(unittest.TestCase):
    def test_target_recall_is_reached(self):
        rng = np.random.default_rng(0)
        y = np.r_[np.zeros(500), np.ones(500)].astype(int)
        s = np.r_[rng.normal(0, 1, 500), rng.normal(1, 1, 500)]
        for target in (0.8, 0.9, 0.95):
            thr = choose_threshold(y, s, "target_recall", target)
            m = binary_metrics(y, s, thr)
            self.assertGreaterEqual(m["recall"], target)
            # the next higher score would fall below the target: threshold is as strict as allowed
            higher = np.min(s[s > thr])
            self.assertLess(binary_metrics(y, s, higher)["recall"], target)

    def test_f2_mode_and_metrics(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.4, 0.35, 0.8])
        thr = choose_threshold(y, s, "f2")
        m = binary_metrics(y, s, thr)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 0, "tp": 2})
        self.assertAlmostEqual(m["roc_auc"], 0.75)


class TestEndToEnd(Tmp):
    def test_quick_training_writes_outputs_and_model_reloads(self):
        cfg = Config(image_size=None, cv_folds=2, n_jobs=1, verbose=0,
                     candidates=["raw_pca_logreg", "hog_logreg", "hog_rf"],
                     param_grids={"raw_pca_logreg": {"pca__n_components": [5], "logreg__C": [0.1]},
                                  "hog_logreg": {"logreg__C": [0.01, 0.1]},
                                  "hog_rf": {"rf__n_estimators": [20], "rf__max_depth": [3]}})
        res = run(self.root, cfg, Path(self._d.name) / "runs")
        d = res.run_dir
        for f in ["metrics.json", "params.json", "config.json", "comparison.csv", "comparison.md", "model.pkl",
                  "train.log", "figures/test_curves.png", "figures/test_confusion.png"]:
            self.assertTrue((d / f).exists(), f)
        m = json.loads((d / "metrics.json").read_text())
        for k in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "f2", "roc_auc",
                  "pr_auc", "confusion_matrix"]:
            self.assertIn(k, m["test"])
        self.assertEqual(set(m["candidates"]), set(cfg.candidates))
        self.assertIn("scikit-learn", m["environment"])
        self.assertGreater(m["validation"]["roc_auc"], 0.8)            # the synthetic signal is learnable
        bundle = load_model(d / "model.pkl")
        test = load_split(self.root, "test")
        pred = predict(bundle, test.X)
        self.assertEqual(pred.shape, test.y.shape)
        cm = m["test"]["confusion_matrix"]
        self.assertEqual(int(pred.sum()), cm["tp"] + cm["fp"])


if __name__ == "__main__":
    unittest.main()
