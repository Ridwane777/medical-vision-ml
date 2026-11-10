# ADR-0001: Adaptive clustering pipeline for any CSV of texts

- **Status**: Accepted
- **Date**: 2026-11-10
- **Deciders**: Ridwane777

## Context

The unsupervised part clusters free-text patient testimonies. The first version (`unsupervised_learning.py`, 339
lines) only worked on `Student_Dataset.csv`: hard-coded file name, no header with names `ID`/`Code`/`Testimonies`, a
repair rule for shifted rows based on the `0x` prefix, English stop words only, `min_df=2`/`max_df=0.8`, 100 SVD
components, and k = 14 chosen by eye and then hard-coded in loops (`range(14)`). Any other file, or even a smaller
sample of the same file, required code changes or crashed (for example 100 SVD components on fewer than 100 documents).

Facts about the data: 1,011 rows, 397 of them with 2 fields instead of 3, and a `Code` column that is always
`0x000000`, so there are no labels to evaluate against.

Constraint from the project specification: only scikit-learn, pandas, NumPy, Matplotlib, JupyterLab and the Python
standard library may be imported (Pillow tolerated). This excludes NLTK, spaCy, sentence embeddings, `langdetect`
and `pytest`.

## Options considered

1. **Keep the script, expose its constants as parameters.** Cheap, but the user still has to know the delimiter,
   header, text column, k and the TF-IDF bounds for each new file; small corpora still crash.
2. **Adaptive package with data-derived defaults and overrides** (`src/textclust`): detect the file layout, derive
   every size from the corpus, choose k and the algorithm by an internal metric, write a reproducible JSON report.
3. **Embeddings + HDBSCAN** (sentence-transformers, or a pretrained language model). Better semantic grouping of
   synonyms, but forbidden by the library constraint and heavier to run.
4. **Topic model (LDA with `sklearn.decomposition.LatentDirichletAllocation`).** Allowed, gives interpretable topics,
   but needs a number of topics too and has no single comparable quality score with K-Means.

## Decision

Option 2. The package splits the work into `io` (encoding fallback, `csv.Sniffer` + fallback, header heuristic,
ragged-row repair where the longest field is the text, text/label column detection), `clean` (NFKC, accent-preserving
cleanup, stop-word-overlap language detection for English/French, dropping empty/short/duplicate texts), `features`
(TF-IDF bounds and SVD size from corpus size), `model` (K-Means over k = 2..min(20, max(3, sqrt(n/2))) vs. HDBSCAN with
`min_cluster_size` = 2% of n; best cosine silhouette among candidates with >= 2 clusters and <= 30% noise), `report`
and `pipeline.run()`, with a CLI (`python -m textclust`). Every inferred value is overridable and saved to
`runs/<timestamp>/metrics.json`. Adaptivity is proven by a `unittest` suite on generated CSV files.

## Consequences

Gained:
- ✅ Any CSV of texts runs without code changes (tested: header or not, `,` `;` tab, Latin-1, ragged rows, English or
  French, 15 to 600 rows; a 4-topic synthetic corpus is recovered with ARI 1.0).
- ✅ k and the algorithm are chosen by a stated rule instead of by eye; every parameter is traceable in `metrics.json`.
- ✅ The notebook only has a parameter cell; the code is testable and reused by the CLI and the script.

Lost or accepted:
- ❌ Heuristics can guess wrong (header with short text columns, a categorical column that is not a label); overrides
  exist but the user has to notice.
- ❌ On the real data, the silhouette curve has no clear peak (about 0.12 from k = 13 to 20), so the automatic k = 18
  is a weak optimum; the rule can favour the top of the range.
- ❌ Bag-of-words without stemming or embeddings cannot merge synonyms; option 3 would be the next step if the library
  constraint is lifted.
- ❌ Only English and French stop words are built in.
- ❌ More code to maintain than a single script (about 740 lines across modules plus 170 lines of tests).
