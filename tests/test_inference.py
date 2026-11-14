"""Tests for the kneeclf CLI subcommands (train / export / predict / download-model) and model checksums.

A tiny model is trained on synthetic PNGs in a temporary folder; no real data and no network are used.
"""
import contextlib
import io
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kneeclf import inference  # noqa: E402
from kneeclf.__main__ import main  # noqa: E402
from test_kneeclf import synthetic_image, write_dataset  # noqa: E402

logging.disable(logging.CRITICAL)

CONFIG = {"image_size": None, "candidates": ["hog_logreg"], "param_grids": {"hog_logreg": {"logreg__C": [0.1]}},
          "cv_folds": 2, "n_jobs": 1, "verbose": 0}


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TestCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._d = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._d.name)
        write_dataset(cls.tmp / "data")
        cfg = cls.tmp / "cfg.json"
        cfg.write_text(json.dumps(CONFIG), encoding="utf-8")
        cls.cfg = cfg
        # Old form (no subcommand) must still train.
        code, out, _ = run_cli(["--data", str(cls.tmp / "data"), "--config", str(cfg), "--output",
                                str(cls.tmp / "runs"), "--quiet"])
        assert code == 0, out
        cls.run_dir = Path(json.loads(out)["run_dir"])
        cls.model = cls.tmp / "dist" / "model.pkl"
        code, out, _ = run_cli(["export", str(cls.run_dir), "--out", str(cls.model)])
        assert code == 0, out
        cls.export_info = json.loads(out)
        rng = np.random.default_rng(1)
        cls.images = []
        for g in (0, 4):
            # A colour JPEG of another size: predict must convert to grayscale and resize.
            arr = (synthetic_image(g, rng) * 255).astype(np.uint8)
            rgb = Image.fromarray(arr, mode="L").resize((300, 300)).convert("RGB")
            p = cls.tmp / f"grade{g}.jpg"
            rgb.save(p)
            cls.images.append(p)

    @classmethod
    def tearDownClass(cls):
        cls._d.cleanup()

    def test_old_form_and_train_subcommand_are_equivalent(self):
        self.assertTrue((self.run_dir / "model.pkl").is_file())
        code, out, _ = run_cli(["train", "--data", str(self.tmp / "data"), "--config", str(self.cfg), "--output",
                                str(self.tmp / "runs2"), "--quiet"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["selected"], "hog_logreg")

    def test_export_is_self_contained_with_checksum(self):
        b = inference.load_bundle(self.model)
        for key in ("pipeline", "threshold", "label_scheme", "image_size", "hog", "versions", "class_names"):
            self.assertIn(key, b)
        self.assertEqual(b["hog"], {"cell": 8, "block": 2, "bins": 9})
        sha = Path(f"{self.model}.sha256").read_text().split()[0]
        self.assertEqual(sha, self.export_info["sha256"])

    def test_predict_json(self):
        code, out, _ = run_cli(["predict", *map(str, self.images), "--model", str(self.model), "--json"])
        self.assertEqual(code, 0)
        res = json.loads(out)
        self.assertIn("not a medical device", res["disclaimer"])
        r0, r4 = res["results"]
        for r in (r0, r4):
            self.assertTrue(0.0 <= r["score"] <= 1.0)
            self.assertIn(r["oa_signs"], ("yes", "no"))
        self.assertGreater(r4["score"], r0["score"])
        self.assertEqual(r4["oa_signs"], "yes")
        self.assertEqual(r0["oa_signs"], "no")

    def test_predict_text_and_bad_image(self):
        bad = self.tmp / "notes.png"
        bad.write_text("not an image")
        code, out, _ = run_cli(["predict", str(self.images[1]), str(bad), "--model", str(self.model)])
        self.assertEqual(code, 1)
        self.assertIn("signs of osteoarthritis (KL>=2): yes", out)
        self.assertIn("error", out)
        self.assertIn("not a medical device", out)

    def test_hog_figure(self):
        fig = inference.hog_figure(self.images[0], inference.load_bundle(self.model))
        self.assertEqual(len(fig.axes), 2)

    def test_checksum_mismatch_is_refused(self):
        bad = self.tmp / "tampered.pkl"
        bad.write_bytes(self.model.read_bytes() + b"x")
        Path(f"{bad}.sha256").write_text(Path(f"{self.model}.sha256").read_text())
        with self.assertRaises(inference.ChecksumError):
            inference.load_bundle(bad)
        code, _, err = run_cli(["predict", str(self.images[0]), "--model", str(bad)])
        self.assertEqual(code, 2)
        self.assertIn("SHA-256", err)

    def test_missing_model_message(self):
        code, _, err = run_cli(["predict", str(self.images[0]), "--model", str(self.tmp / "nope.pkl")])
        self.assertEqual(code, 2)
        self.assertIn("download-model", err)

    def _fake_urlopen(self, files):
        def fake(url, timeout=None):
            return io.BytesIO(files[url.rsplit("/", 1)[1]])
        return fake

    def test_download_verifies_checksum(self):
        name = inference.MODEL_FILENAME
        good = {name: self.model.read_bytes(), name + ".sha256": Path(f"{self.model}.sha256").read_bytes()}
        dest = self.tmp / "cache"
        with mock.patch("urllib.request.urlopen", self._fake_urlopen(good)):
            path = inference.download_model(dest)
        self.assertEqual(path, dest / name)
        inference.load_bundle(path)

        corrupt = {**good, name: good[name][:-10]}
        dest2 = self.tmp / "cache2"
        with mock.patch("urllib.request.urlopen", self._fake_urlopen(corrupt)):
            with self.assertRaises(inference.ChecksumError):
                inference.download_model(dest2)
        self.assertEqual(list(dest2.iterdir()), [])        # nothing left behind


if __name__ == "__main__":
    unittest.main()
