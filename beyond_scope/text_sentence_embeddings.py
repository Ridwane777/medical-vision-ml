"""Beyond the specification: sentence embeddings instead of TF-IDF + SVD for testimony clustering.

Uses sentence-transformers (forbidden by the project specification), so this lives outside src/.
Same rows as the core (src/textclust io + clean), same automatic k / KMeans-vs-HDBSCAN selection
(src/textclust/model.py). Writes only aggregates: no testimony text, no embeddings.

Run from the repository root:
    PYTHONPATH=src:. python beyond_scope/text_sentence_embeddings.py unsupervised_learning/Student_Dataset.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from textclust import clean, features, io, model, report  # noqa: E402

log = logging.getLogger("beyond.text")
MODELS = ("sentence-transformers/all-MiniLM-L6-v2",
          "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def summarize(choice, labels_core=None):
    sizes = {int(c): int((choice.labels == c).sum()) for c in sorted(set(choice.labels))}
    out = {"algorithm": choice.algorithm, "n_clusters": choice.n_clusters,
           "noise_ratio": round(choice.noise_ratio, 4),
           **{k: (round(v, 4) if v is not None else None) for k, v in choice.scores.items()},
           "params": choice.params, "candidates": choice.candidates,
           "k_curve": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                       for r in choice.k_curve],
           "cluster_sizes": sizes}
    if labels_core is not None:
        out["ari_vs_core"] = round(float(adjusted_rand_score(labels_core, choice.labels)), 4)
        out["nmi_vs_core"] = round(float(normalized_mutual_info_score(labels_core, choice.labels)), 4)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--out", default="results/beyond_scope/text")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    loaded = io.load_csv(a.csv)
    df, clean_report = clean.clean_corpus(loaded.frame, loaded.text_column, 2)
    stop = clean.stop_words_for(clean_report["language"])
    # Core reference, identical to `python -m textclust` defaults.
    t = time.perf_counter()
    core_feats = features.build_features(df["clean_text"], stop_words=stop)
    core = model.select_model(core_feats.embedding, random_state=42)
    core_seconds = round(time.perf_counter() - t, 1)
    results = {"n_documents": int(len(df)), "language": clean_report["language"],
               "core_tfidf_svd": {**summarize(core), "seconds": core_seconds, "features": core_feats.params}}
    terms = {"core_tfidf_svd": report.top_terms(core_feats.tfidf, core.labels, core_feats.vectorizer, 8)}

    # Sentence encoders read the original sentence (they rely on word order and punctuation); the row
    # set is exactly the cleaned corpus used by the core.
    texts = df[loaded.text_column].astype(str).tolist()
    for name in MODELS:
        short = name.split("/")[-1]
        t = time.perf_counter()
        enc = SentenceTransformer(name, device="cpu")
        load_s = time.perf_counter() - t
        n_params = sum(p.numel() for p in enc.parameters())
        t = time.perf_counter()
        emb = enc.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        enc_s = time.perf_counter() - t
        t = time.perf_counter()
        ch = model.select_model(np.asarray(emb, dtype=np.float64), random_state=42)
        sel_s = time.perf_counter() - t
        results[short] = {**summarize(ch, core.labels), "embedding_dims": int(emb.shape[1]),
                          "parameters": int(n_params), "weights_mb_fp32": round(n_params * 4 / 2**20, 1),
                          "load_seconds": round(load_s, 1), "encode_seconds": round(enc_s, 1),
                          "selection_seconds": round(sel_s, 1)}
        # Top terms: TF-IDF centroid of each embedding cluster's members (core vocabulary).
        terms[short] = report.top_terms(core_feats.tfidf, ch.labels, core_feats.vectorizer, 8)
        log.info("%s: %s k=%d silhouette=%.4f ARI vs core=%.4f", short, ch.algorithm, ch.n_clusters,
                 ch.scores["silhouette"], results[short]["ari_vs_core"])

    results["environment"] = {"torch": torch.__version__, "threads": a.threads, "device": "cpu"}
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out / "top_terms.json").write_text(json.dumps(terms, indent=2), encoding="utf-8")

    lines = ["| Representation | Algorithm | k | Noise | Silhouette (cosine) | ARI vs core | Encode time |",
             "|---|---|---|---|---|---|---|"]
    for key, r in results.items():
        if isinstance(r, dict) and "algorithm" in r:
            lines.append(f"| {key} | {r['algorithm']} | {r['n_clusters']} | {r['noise_ratio']} | "
                         f"{r['silhouette']} | {r.get('ari_vs_core', '-')} | {r.get('encode_seconds', '-')} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, r in results.items():
        if isinstance(r, dict) and r.get("k_curve"):
            ax.plot([c["k"] for c in r["k_curve"]], [c["silhouette"] for c in r["k_curve"]], marker="o",
                    ms=3, label=key)
    ax.set(xlabel="k (K-Means)", ylabel="cosine silhouette", title="k selection: TF-IDF+SVD vs sentence embeddings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "k_selection.png", dpi=110)
    log.info("done")


if __name__ == "__main__":
    main()
