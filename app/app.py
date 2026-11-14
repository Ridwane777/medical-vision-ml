"""Medical Vision ML: Gradio demo, run locally from the repository (server-side Python).

Local run from the repository (after exporting or downloading the model):
    python app/app.py
The online demo is the static page in demo/ (GitHub Pages); this app is the same demo for people running from source.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
# In the repository the packages live in ../src (or next to this file if copied together).
for cand in (HERE, HERE.parent / "src"):
    if (cand / "kneeclf").is_dir():
        sys.path.insert(0, str(cand))
        break

from kneeclf import inference  # noqa: E402
from textclust import pipeline as tc  # noqa: E402

REPO_URL = "https://github.com/Ridwane777/medical-vision-ml"
EXAMPLES = HERE / "examples"
MAX_ROWS = 5000
MAX_BYTES = 5 * 2**20


def find_model():
    candidates = [os.environ.get("KNEECLF_MODEL"), HERE / inference.MODEL_FILENAME,
                  HERE.parent / "dist" / inference.MODEL_FILENAME, inference.default_model_path()]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    raise FileNotFoundError("model file not found: run "
                            "`python -m kneeclf export ...` or `python -m kneeclf download-model`")


BUNDLE = inference.load_bundle(find_model())
TEST = BUNDLE.get("metrics", {}).get("test", {})
THRESHOLD = inference.threshold_probability(BUNDLE)

DISCLAIMER = (
    "**Research / education demo, not a medical device.** The score is not a diagnosis and must not guide care. "
    "The model (HOG features + logistic regression) was trained on one public dataset (Chen 2018, OAI X-rays) and "
    f"was not validated on any other hospital, scanner or population. On its held-out test set "
    f"({TEST.get('n', '?')} images) it reaches recall **{TEST.get('recall', float('nan')):.3f}** and specificity "
    f"**{TEST.get('specificity', float('nan')):.3f}** (ROC-AUC {TEST.get('roc_auc', float('nan')):.3f}): about "
    "1 knee with OA in 5 is missed and 1 healthy knee in 4 is flagged. Uploaded images are processed in memory "
    "and not stored by this app."
)


def gauge(score, threshold):
    fig, ax = plt.subplots(figsize=(6, 1.3))
    ax.barh([0], [1], color="#e8e8e8", height=0.5)
    ax.barh([0], [score], color="#c0392b" if score >= threshold else "#2e86c1", height=0.5)
    ax.axvline(threshold, color="black", lw=2, ls="--")
    ax.text(threshold, 0.45, f"threshold {threshold:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("model score (probability-like, 0 = no OA signs, 1 = OA signs)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def predict_knee(image):
    """image: PIL image (Gradio). Returns (markdown, gauge figure, HOG figure)."""
    if image is None:
        return "Upload a knee X-ray or pick an example.", None, None
    r = inference.predict_one(image, BUNDLE)
    md = (f"### Signs of osteoarthritis (KL>=2): **{r['oa_signs']}**\n\n"
          f"Score **{r['score']:.3f}** vs. decision threshold {r['threshold']:.3f} "
          f"(threshold set on validation for 85% recall). Predicted class: {r['label']}.\n\n"
          "_Not a diagnosis: see the disclaimer below._")
    return md, gauge(r["score"], r["threshold"]), inference.hog_figure(image, BUNDLE)


def _count_rows(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def cluster_texts(file):
    """file: path of an uploaded CSV. Returns (markdown summary, cluster table, projection image, sizes image)."""
    import pandas as pd
    if file is None:
        return "Upload a CSV file or pick the sample.", None, None, None
    path = Path(file if isinstance(file, (str, Path)) else file.name)
    if path.stat().st_size > MAX_BYTES:
        return f"File too large ({path.stat().st_size / 2**20:.1f} MB > {MAX_BYTES // 2**20} MB).", None, None, None
    n = _count_rows(path)
    if n > MAX_ROWS + 1:
        return f"Too many rows ({n - 1}); this demo accepts up to {MAX_ROWS}.", None, None, None
    out = Path(tempfile.mkdtemp(prefix="textclust-"))
    try:
        res = tc.run(path, config=tc.Config(k_max_cap=12), output_dir=out)
    except (ValueError, UnicodeDecodeError) as e:
        return f"Could not cluster this file: {e}", None, None, None
    m = res.metrics
    ch = m["chosen"]
    lines = [f"- Text column detected: **{res.text_column}**"
             + (f" (reference labels: `{res.label_column}`)" if res.label_column else ""),
             f"- Rows used after cleaning: {len(res.data)}; language: **{m['language_used']}**",
             f"- Algorithm chosen: **{ch['algorithm']}**, **{ch['n_clusters']} clusters**, "
             f"cosine silhouette **{ch['silhouette']:.3f}**, noise {ch['noise_ratio']:.0%}"]
    if m.get("external_labels"):
        e = m["external_labels"]
        lines.append(f"- Agreement with `{res.label_column}`: ARI {e['ari']:.2f}, NMI {e['nmi']:.2f}")
    table = res.clusters.copy()
    table["cluster"] = table["cluster"].map(lambda c: "noise" if c == -1 else str(c))
    table = pd.DataFrame(table)
    proj = res.run_dir / "projection_2d.png"
    sizes = res.run_dir / "cluster_sizes.png"
    return ("\n".join(lines), table, str(proj) if proj.exists() else None,
            str(sizes) if sizes.exists() else None)


ABOUT = f"""
## Medical Vision ML

Classical machine learning on medical data, built to be read and reproduced on a laptop CPU.

* **Knee X-ray classifier**: a hand-written HOG descriptor (NumPy) + logistic regression separates KL 0-1 from
  KL 2-4 (radiographic osteoarthritis). The decision threshold is set on the validation split for 85% recall,
  then the test split is used once. This demo model is the light variant (a few hundred KB); the full study's
  best model (HOG + SVC) reaches test ROC-AUC 0.861 but weighs 200 MB.
* **Text clustering**: TF-IDF + truncated SVD, then K-Means or HDBSCAN chosen automatically by silhouette, with
  automatic detection of the delimiter, header, text column and language. Works on any CSV of short texts.

Source code, notebooks, model card and data cards: [{REPO_URL}]({REPO_URL})
([model card]({REPO_URL}/blob/main/docs/model-card.md), [docs]({REPO_URL}/tree/main/docs)).

Example X-rays: Chen, P. (2018), *Knee Osteoarthritis Severity Grading Dataset*, Mendeley Data,
doi:10.17632/56rmx5bjcr.1, CC BY 4.0. The text sample is synthetic (no real patient text).

Model: `{BUNDLE.get('candidate')}`, scikit-learn {BUNDLE['versions'].get('scikit-learn')}, input
{BUNDLE['image_size']} x {BUNDLE['image_size']} px grayscale.
"""


def build():
    import gradio as gr
    knee_examples = sorted(str(p) for p in EXAMPLES.glob("kl*.png"))
    with gr.Blocks(title="Medical Vision ML") as demo:
        gr.Markdown("# Medical Vision ML\nClassical ML demos: knee X-ray screening score and text clustering.")
        with gr.Tab("Knee X-ray"):
            with gr.Row():
                with gr.Column():
                    img = gr.Image(type="pil", label="Knee X-ray (PNG/JPG)", image_mode="L", height=300)
                    btn = gr.Button("Analyse", variant="primary")
                    gr.Examples(knee_examples, inputs=img, label="Test-split examples (file name = true KL grade)")
                with gr.Column():
                    verdict = gr.Markdown()
                    gauge_plot = gr.Plot(label="Score")
            hog_plot = gr.Plot(label="Model input and its HOG cells")
            gr.Markdown(DISCLAIMER)
            btn.click(predict_knee, img, [verdict, gauge_plot, hog_plot])
            img.upload(predict_knee, img, [verdict, gauge_plot, hog_plot])
        with gr.Tab("Cluster your texts"):
            gr.Markdown(f"Upload a CSV with one short text per row (max {MAX_ROWS} rows, "
                        f"{MAX_BYTES // 2**20} MB). Delimiter, header, text column and language are detected.")
            with gr.Row():
                csv = gr.File(label="CSV file", file_types=[".csv", ".tsv", ".txt"])
                with gr.Column():
                    run_btn = gr.Button("Cluster", variant="primary")
                    gr.Examples([str(EXAMPLES / "sample_texts.csv")], inputs=csv,
                                label="Synthetic sample (150 symptom descriptions, 5 topics)")
            summary = gr.Markdown()
            table = gr.Dataframe(label="Clusters: size, top terms, representative example", wrap=True)
            with gr.Row():
                proj = gr.Image(label="2-D projection", type="filepath")
                sizes = gr.Image(label="Cluster sizes", type="filepath")
            run_btn.click(cluster_texts, csv, [summary, table, proj, sizes], concurrency_limit=2)
        with gr.Tab("About"):
            gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    build().queue(max_size=16).launch()
