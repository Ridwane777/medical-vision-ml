"""Text clustering tab of the static demo: runs in the visitor's browser with Pyodide (Python on WebAssembly).

index.html loads numpy, pandas, scikit-learn and matplotlib from the Pyodide distribution, writes the textclust
package (copied from src/textclust by scripts/build_demo.sh) next to this file, then calls `cluster_json`.
Nothing leaves the browser. The same function runs under CPython: tests/test_demo_cluster.py checks it.
"""
import base64
import json
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from textclust import pipeline as tc  # noqa: E402

MAX_ROWS = 2000
MAX_BYTES = 2 * 2**20


def _png(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else None


def cluster_json(csv_path):
    """Cluster a CSV of short texts; returns a JSON string {ok, error | summary, table, projection, sizes}."""
    path = Path(csv_path)
    size = os.path.getsize(path)
    if size > MAX_BYTES:
        return json.dumps({"ok": False, "error": f"File too large ({size / 2**20:.1f} MB > {MAX_BYTES // 2**20} MB)."})
    with open(path, "rb") as f:
        n = sum(1 for _ in f)
    if n > MAX_ROWS + 1:
        return json.dumps({"ok": False, "error": f"Too many rows ({n - 1}); the in-browser demo accepts up to {MAX_ROWS}."})
    out = Path(tempfile.mkdtemp(prefix="textclust-"))
    try:
        res = tc.run(path, config=tc.Config(k_max_cap=12), output_dir=out)
    except (ValueError, RuntimeError, UnicodeDecodeError) as e:
        return json.dumps({"ok": False, "error": f"Could not cluster this file: {e}"})
    m = res.metrics
    ch = m["chosen"]
    summary = {"text_column": res.text_column, "label_column": res.label_column, "rows": len(res.data),
               "language": m["language_used"], "algorithm": ch["algorithm"], "n_clusters": ch["n_clusters"],
               "silhouette": ch["silhouette"], "noise_ratio": ch["noise_ratio"], "external": m.get("external_labels")}
    table = [{"cluster": "noise" if r.cluster == -1 else str(r.cluster), "size": int(r.size),
              "top_terms": r.top_terms, "example": r.example} for r in res.clusters.itertuples()]
    return json.dumps({"ok": True, "summary": summary, "table": table,
                       "projection": _png(res.run_dir / "projection_2d.png"),
                       "sizes": _png(res.run_dir / "cluster_sizes.png")})
