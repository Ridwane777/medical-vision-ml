"""The Python glue of the static demo's clustering tab (demo/cluster_app.py) runs under CPython too."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_cluster_app():
    spec = importlib.util.spec_from_file_location("cluster_app", ROOT / "demo" / "cluster_app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDemoCluster(unittest.TestCase):
    def test_sample_csv(self):
        res = json.loads(load_cluster_app().cluster_json(ROOT / "app" / "examples" / "sample_texts.csv"))
        self.assertTrue(res["ok"], res)
        s = res["summary"]
        self.assertEqual((s["text_column"], s["label_column"], s["rows"]), ("description", "topic", 150))
        self.assertGreaterEqual(s["n_clusters"], 2)
        self.assertTrue(res["projection"].startswith("data:image/png;base64,"))
        self.assertEqual(sum(r["size"] for r in res["table"]), 150)

    def test_rejects_too_many_rows(self):
        import tempfile
        app = load_cluster_app()
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("text\n" + "some short text here\n" * (app.MAX_ROWS + 5))
        res = json.loads(app.cluster_json(f.name))
        Path(f.name).unlink()
        self.assertFalse(res["ok"])
        self.assertIn("Too many rows", res["error"])


if __name__ == "__main__":
    unittest.main()
