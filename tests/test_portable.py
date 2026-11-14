"""Portable JSON model: numpy predictor (and the JavaScript port in demo/knee.js) == scikit-learn pipeline.

A small HOG + StandardScaler + LogisticRegression pipeline is fitted on synthetic images; no real data is used.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kneeclf import inference, portable  # noqa: E402
from kneeclf.features import HOG  # noqa: E402
from test_kneeclf import synthetic_image  # noqa: E402

SIDE_IN, SIDE = 224, 112


def synthetic_images(n, seed):
    rng = np.random.default_rng(seed)
    grades = rng.integers(0, 5, n)
    imgs = []
    for g in grades:
        small = synthetic_image(int(g), rng)
        imgs.append(np.asarray(Image.fromarray((small * 255).astype(np.uint8), mode="L")
                               .resize((SIDE_IN, SIDE_IN), Image.Resampling.BILINEAR)))
    return imgs, (grades >= 2).astype(int)


class TestPortable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        imgs, y = synthetic_images(60, 0)
        bundle0 = {"input_side": SIDE_IN, "image_size": SIDE}
        X = np.concatenate([portable.preprocess(im, bundle0) for im in imgs])
        pipe = Pipeline([("features", HOG(cell=8, block=2, bins=9)),
                         ("model", Pipeline([("scale", StandardScaler()),
                                             ("logreg", LogisticRegression(C=0.01, max_iter=2000))]))])
        pipe.fit(X, y)
        cls.bundle = inference.make_bundle(pipe, -0.3, "kl01_vs_2to4", SIDE, {"cell": 8, "block": 2, "bins": 9},
                                           "hog_logreg", {"test": {"recall": 0.5}})
        cls.tmp = tempfile.TemporaryDirectory()
        cls.json_path, _ = portable.export_json(cls.bundle, Path(cls.tmp.name) / "model.json")
        cls.model = portable.load_model(cls.json_path)
        cls.test_imgs, _ = synthetic_images(25, 1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_json_is_plain_data(self):
        d = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.assertEqual(d["format"], portable.JSON_FORMAT)
        self.assertEqual(len(d["logreg"]["coef"]), d["n_features"])
        self.assertEqual(len(d["scaler"]["mean"]), d["n_features"])
        self.assertEqual(d["metrics"], {"test": {"recall": 0.5}})

    def test_decision_function_matches_pipeline(self):
        X = np.concatenate([inference.preprocess(im, self.bundle) for im in self.test_imgs])
        ref = self.bundle["pipeline"].decision_function(X)
        got = portable.decision_function(X, self.model)
        self.assertLess(np.abs(ref - got).max(), 1e-6)

    def test_predict_one_matches_pickle_path(self):
        for im in self.test_imgs[:5]:
            pil = Image.fromarray(im, mode="L")
            self.assertEqual(portable.predict_one(pil, self.model), inference.predict_one(pil, self.bundle))

    def test_rejects_other_formats(self):
        with self.assertRaises(ValueError):
            portable.load_model({"format": "something-else"})

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_javascript_port_matches(self):
        cases = [{"px": im.ravel().tolist(), "w": SIDE_IN, "h": SIDE_IN,
                  "py": float(portable.decision_function(portable.preprocess(im, self.model), self.model)[0])}
                 for im in self.test_imgs[:8]]
        cases_path = Path(self.tmp.name) / "cases.json"
        cases_path.write_text(json.dumps(cases), encoding="utf-8")
        script = ("const K=require(process.argv[1]);const m=require(process.argv[2]);"
                  "const c=require(process.argv[3]);let d=0;"
                  "for(const e of c){d=Math.max(d,Math.abs(K.predict(Uint8Array.from(e.px),e.w,e.h,m)"
                  ".decision_score-e.py));}console.log(d);")
        out = subprocess.run(["node", "-e", script, str(ROOT / "demo" / "knee.js"), str(self.json_path),
                              str(cases_path)], capture_output=True, text=True, check=True)
        self.assertLess(float(out.stdout), 1e-4)


if __name__ == "__main__":
    unittest.main()
