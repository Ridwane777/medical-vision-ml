# Retrospective

A look at the whole project, from the school version (March 2026) to the solo portfolio refactor (October to
November 2026).

## Context

- **Origin**: a school project with three parts (chest X-ray EDA, knee X-ray classification, clustering of
  patient testimonies) and a restricted library list: NumPy, pandas, Matplotlib, scikit-learn, JupyterLab, Pillow and
  the Python standard library. No deep learning framework.
- **Refactor**: done solo, keeping that constraint, to turn notebooks into tested packages (`src/kneeclf`,
  `src/textclust`) with documented decisions ([ADR index](../adr/README.md)).

## What worked

- **HOG written in NumPy**: the largest gain of the project, +0.07 validation ROC-AUC over raw pixels for the same
  model family (0.857 vs. 0.779 for LogReg, 0.861 vs. 0.792 for SVC).
- **Recall-first protocol with a test set touched once**: the reported test numbers (recall 0.831, ROC-AUC 0.861) are
  honest, and the gap to the 0.85 validation target is visible instead of hidden.
- **Run directory with everything** (config, environment, data profile, timings): every number in the docs can be
  traced to a file.
- **Adaptive text pipeline**: it repaired 397 shifted rows and detected the constant `Code` column automatically,
  where the old script had hard-coded assumptions.

## What we would do differently

- **Held-out test discipline from day one.** The school version first evaluated on test several times (fixed in commit
  "evaluate on the test split only once", 2026-03-29); the earlier numbers (SVM 66.46% accuracy) cannot be compared.
- **Recall-first metric from day one**, instead of accuracy on a balanced validation set, which hid the real class
  ratio.
- **One coherent subject instead of three unrelated sub-projects.** The pneumonia EDA never got a model and stayed in
  `legacy/`; the time would have been better spent on the knee classifier.
- **Text data without labels limits evaluation.** With the `Code` column constant, the clustering can only be judged by
  silhouette (0.125) and reading. A few dozen hand-labelled testimonies would have allowed ARI / NMI, which the pipeline
  already computes when a label column exists.

## Alternatives, when they become preferable, and their cost

| Alternative | Preferable when | Cost |
| --- | --- | --- |
| CNN or transfer learning (e.g. ImageNet-pretrained ResNet fine-tuned on KL grades) | the library policy allows PyTorch or TensorFlow, a GPU is available for fine-tuning, and a ROC-AUC well above the 0.861 baseline is the goal | **Measured** ([beyond_scope](../../beyond_scope/README.md)): *frozen* ResNet-50 + LogReg reaches test ROC-AUC 0.835, recall 0.813, specificity 0.665, below HOG + SVC. Embedding 7,374 images took 1,762 s on 4 CPU threads (EfficientNet-B0: 716 s). Fine-tuning was not run: one forward pass over train alone took 20 min on CPU |
| HOG + LogReg instead of HOG + SVC | inference cost or model size matters (the SVC pickle is about 200 MB) | -0.004 validation ROC-AUC, -0.029 validation specificity at the same recall |
| Full 224 px resolution | RAM >= 16 GB and run time are not limiting | about 4 times more pixels, 26,244 HOG dims, longer searches |
| Sentence embeddings for testimonies | a pretrained embedding library is allowed | **Measured** ([beyond_scope](../../beyond_scope/README.md)): all-MiniLM-L6-v2 (86.6 MB) gives k = 4, silhouette 0.169 in its own space, ARI 0.236 vs the core's k = 18. Multilingual MiniLM-L12 (448.8 MB) gives k = 3, silhouette 0.193, ARI 0.142. Coarser themes, 37 to 76 s vs 6.6 s, and no labels to say which partition is better |
| Stemming / lemmatization | allowed library (e.g. NLTK) or a small hand-written stemmer is worth the time | language-specific rules; would merge `breath` / `breathing` |

## Organisation

- School phase: three loosely related sub-projects, which encouraged parallel notebooks rather than one shared
  pipeline. Reviews were informal.
- Refactor phase: one person, so no peer review; tests (`tests/`) and ADRs replace part of that safety net.

## Hardware limits

- Development on a laptop (no GPU); the full grid search (36 min, 4 workers) was run on a shared 8-core VPS so the
  laptop stayed usable. Memory pushed the full run to 112 px instead of 224 px.
- No GPU at any point, which (with the library list) ruled out deep learning.

## Deadlines

- School phase: the git history shows work from 2026-03-10 to 2026-03-29. Cut or deferred: a model for the pneumonia
  dataset (EDA only) and a clean threshold protocol.
- Refactor: 2026-10-26 (README) to 2026-11-11 (full-run results).

## What we would keep

The library constraint (it forces understanding the features), the recall-first protocol, versioned results without
the heavy model file, and one ADR per decision.

## Next steps (beyond scope)

- LBP texture features and HOG + LBP (grids already in the config: `lbp_logreg`, `hog+lbp_svc`, not run).
- Evaluate on the `auto_test` split (`extra_test_splits`) as a second test set.
- Calibrate the threshold with a confidence interval (bootstrap) to anticipate drift like 0.85 -> 0.831.
- Hand-label a sample of testimonies to measure clustering against ground truth.
