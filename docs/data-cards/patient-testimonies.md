# Data card: patient testimonies (`data/testimonies.csv`)

Short datasheet after Gebru et al., "Datasheets for Datasets" (2021).

## Source and terms

- The CSV is not in the repository (it is git-ignored) because we have neither its license nor permission to
  redistribute it. Place a local copy at `data/testimonies.csv` (see [data/README.md](../../data/README.md)).

## Privacy

- Free-text descriptions of health symptoms (pain, rashes, chest tightness, blood sugar...). Even without names, such
  text is health data in nature. The run outputs (`runs/`) quote testimonies and are git-ignored for that reason;
  only aggregate numbers and top terms are published.
- Nothing in this project tries to link a testimony to a person.

## Composition

Numbers from the clustering run of 2026-11-10 (`runs/<timestamp>/metrics.json`, also shown in
[`notebooks/02_testimony_clustering.ipynb`](../../notebooks/02_testimony_clustering.ipynb)):

- 1,011 rows, UTF-8 with BOM, comma separated, no header. Columns: `ID`, `Code`, free English text.
- `Code` is **constant** (`0x000000` or empty): there are **no labels**, so only internal metrics can be computed.
- 397 rows have 2 fields instead of 3 (shifted rows); repaired by keeping the longest field as the text.

## Preprocessing

Normalization, 17 exact duplicates dropped, **994 testimonies kept**; language detected as English (53% of tokens are
English stop words). TF-IDF (1,271 terms, `min_df=2`, `max_df=0.8`) then TruncatedSVD to 100 components.

## Known issues

- No ground truth: cluster quality can only be judged by silhouette (0.125 for the chosen K-Means, k = 18) and by
  reading the top terms.
- Short, overlapping texts; synonyms are not linked (no stemming or embeddings in the lightweight core; see the
  [deep embeddings comparison](../../experiments/deep_embeddings/README.md)).
- Collection process, population and consent are unknown.
