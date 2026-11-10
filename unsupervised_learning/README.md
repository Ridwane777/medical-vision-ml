# Unsupervised Learning - Adaptive Testimony Clustering

This folder clusters free-text patient testimonies into themes. The logic lives in the reusable package
[`src/textclust`](../src/textclust); this folder holds the notebook and a thin script wrapper. The pipeline is not tied to
`Student_Dataset.csv`: it adapts to any CSV of texts without code changes. The decision behind this design is recorded in
[ADR-0001](../docs/adr/0001-adaptive-clustering-pipeline.md).

| File | Role |
| --- | --- |
| `clustering.ipynb` | Step-by-step run with explanations. The first cell holds the only parameters: `CSV_PATH`, plus optional overrides. Saved with outputs from the real dataset. |
| `unsupervised_learning.py` | Thin wrapper around the CLI. With no CSV argument it runs on `Student_Dataset.csv` and writes to `../runs/`. |
| `../src/textclust/` | `io.py` (loading), `clean.py` (normalization, language), `features.py` (TF-IDF, SVD), `model.py` (choice of k and algorithm), `report.py` (terms, examples, plots, JSON), `pipeline.py` (`run()`), `__main__.py` (CLI). |
| `../tests/test_textclust.py` | `unittest` suite built on generated CSV files. |

Allowed libraries (project constraint): scikit-learn, pandas, NumPy, Matplotlib, JupyterLab and the Python standard
library. No NLTK, spaCy, embeddings or language-detection package.

## Running

```bash
# from the repository root
PYTHONPATH=src python -m textclust unsupervised_learning/Student_Dataset.csv --output runs
python unsupervised_learning/unsupervised_learning.py          # same, with the default dataset
PYTHONPATH=src python -m textclust any.csv --text-column review --label-column category
python -m unittest discover -s tests -v                         # tests
```

Options: `--text-column` / `--label-column` (name or 0-based index), `--no-labels`, `--k` (force K-Means k),
`--language en|fr|none`, `--ngram-max`, `--output`, `--quiet`. From Python: `textclust.run(csv_path, text_column=None,
label_column=None, config=Config(...), output_dir="runs")` returns a `Result` dataclass.

Each run writes `runs/<timestamp>/`: `metrics.json` (data profile, cleaning report, every parameter used, all candidate
models, the k curve, cluster sizes, top terms, representative examples), `assignments.csv`, `k_selection.png`,
`cluster_sizes.png`, `projection_2d.png`. `runs/` is git-ignored because it contains testimony text.

## Adapts to any CSV

| What | How it is detected | Override |
| --- | --- | --- |
| Encoding | first that decodes among `utf-8-sig`, `utf-8`, `latin-1` | - |
| Delimiter | `csv.Sniffer` on the first 50 lines, fallback: the most consistent of `, ; tab \|` | `load_csv(delimiter=...)` |
| Header | first row made of short, non-numeric names while the body has numeric or much longer values (fallback `Sniffer.has_header`) | `load_csv(header=True/False)` |
| Ragged rows | a row with fewer fields than the table keeps its **longest field as the text** and fills the other columns left to right; a row with more fields has its surplus glued back into the text | - |
| Text column | highest mean length x share of letters | `--text-column` |
| Label column | categorical: 2 to sqrt(n) distinct values, at most n/2, at least 90% filled, not a monotonic ID, not constant | `--label-column`, `--no-labels` |
| Language / stop words | stop-word overlap: scikit-learn English list (+ contraction fragments such as `ve`, `didn`) vs. a hand-written French list | `--language` |
| TF-IDF bounds | `min_df` = 1 (< 50 docs), 2 (< 2,000), 3 (< 10,000), 5; `max_df` = 0.95 (< 100 docs), 0.8 | `features.build_features` |
| SVD size | `min(100, n_terms - 1, n_docs - 1)`, or the fewest components reaching `Config.target_variance` | `Config` |
| k range | 2 .. min(20, max(3, sqrt(n/2))), at most n - 1 | `--k`, `Config.k_max_cap` |
| Algorithm | K-Means (best k) vs. HDBSCAN (`min_cluster_size` = 2% of n, at least 3): best cosine silhouette on non-noise points among candidates with >= 2 clusters and <= 30% noise | `Config.algorithms`, `Config.max_noise` |
| 2D projection | t-SNE (perplexity = min(30, (n-1)/3)) from 30 documents, PCA below | `Config.projection` |

Every random step uses `random_state=42`. When a label column exists, ARI, NMI, homogeneity and completeness are reported.

## Results on `Student_Dataset.csv`

Run of 2026-11-10 (`metrics.json`, same numbers as the saved notebook):

- **Data profile**: 1,011 rows, no header, comma, UTF-8 with BOM. 397 rows had only 2 fields (shifted rows) and were
  repaired by the longest-field rule. Column `col1` (the former `Code`) is **constant** (`0x000000` or empty), so it is
  ignored: **this dataset has no external labels**, and only internal metrics can be computed.
- **Cleaning**: 17 exact duplicates dropped, 994 testimonies kept. Language detected: English (53% of tokens are
  English stop words, 9% French).
- **Features**: 1,271 terms (`min_df=2`, `max_df=0.8`), SVD to 100 components, explained variance **0.415**.
- **Model selection**: K-Means over k = 2..20. Cosine silhouette climbs from 0.049 (k=2) and plateaus around 0.12 from
  k=13; the best is **k = 18, silhouette 0.125** (Calinski-Harabasz 15.5, Davies-Bouldin 3.48). HDBSCAN
  (`min_cluster_size=20`) found 3 clusters with silhouette 0.161 but labelled **77% of the testimonies as noise**, so it
  was not eligible. **Chosen: K-Means, k = 18, noise 0%.**
- **Cluster sizes**: 27 to 96 testimonies.
- **Examples of top terms**: `hot, blister, touched, small, red`; `itchy, rash, new, hives, raised`;
  `wrist, elbow, forearm, hand, palm`; `shoulder, reaching, arm, overhead`; `chest, tightness, tight, pressure`;
  `ear, pain, fever, hear`; `blood, doctor, sugar, high, pressure`; `foot, toe, shoes, pushing`.

Comparison with the previous hard-coded script (k fixed at 14, Euclidean silhouette **0.068** on 1,002 testimonies): the
numbers are not directly comparable, since the silhouette is now computed with the cosine metric and 8 more near
duplicates are removed after normalization. The themes are the same body areas and symptom families.

## Limitations

- The silhouette curve has no clear peak: testimonies form a continuum of overlapping themes. k = 18 is "best among the
  tried values", and k = 20 scores within 0.001. Some clusters are still mixed (cluster 2: `especially, feel, vision,
  night, eye, gums`).
- Bag-of-words cannot link synonyms (`breath` / `breathing` / `winded`): no stemming or embeddings are allowed by the
  project constraint.
- Language detection only knows English and French; any other language falls back to no stop-word list.
- Header detection is a heuristic. When all columns are short words it relies on `csv.Sniffer`; use the overrides if it
  guesses wrong.
