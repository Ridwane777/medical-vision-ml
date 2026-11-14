# Performance

Measured run times and memory. Sources: [`results/knee/train.log`](../../results/knee/train.log),
[`metrics.json`](../../results/knee/metrics.json) (`timings`, `environment`) and
[`comparison.md`](../../results/knee/comparison.md) (`search_seconds`).

## Knee classifier: full run (2026-11-11)

**Environment**: shared 8-core CPU server (VPS), no GPU, `--n-jobs 4`; Linux 6.8.0 x86_64, Python 3.12.3,
scikit-learn 1.9.1, NumPy 2.5.4. Configuration [`configs/full.json`](../../configs/full.json), 112 px, all images.

| Step | Seconds |
| --- | --- |
| Load images (train 4,892 + val 826 + test 1,656) | 33.0 |
| HOG features (6,084 dims) | 33.3 |
| Raw features (12,544 dims) | 2.6 |
| Grid searches (six candidates, below) | 2,013.3 |
| Test prediction (HOG + SVC on 1,656 images) | 48.7 |
| **Total** | **2,188.0 (36.5 min)** |

Grid search per candidate (3-fold CV, then refit on train):

| Candidate | Fits | Search seconds | Val ROC-AUC |
| --- | --- | --- | --- |
| raw pixels + PCA + LogReg | 18 | 132.1 | 0.779 |
| raw pixels + PCA + SVC | 12 | 70.1 | 0.792 |
| HOG + LogReg | 9 | 19.5 | 0.857 |
| HOG + SVC | 12 | 622.9 | 0.861 |
| HOG + RandomForest | 12 | 409.3 | 0.838 |
| HOG + PCA + HistGradientBoosting | 6 | 759.4 | 0.809 |

HOG + LogReg reaches 0.004 ROC-AUC below the SVC for 32 times less search time, and its inference is a dot product
instead of a kernel over the support vectors (test prediction of the SVC: 48.7 s).

**Memory** (measured array sizes from the log, peak not measured): train pixels 234.1 MB, validation 39.5 MB, test
79.2 MB at 112 px float32. HOG features add about 6,084 x 7,374 x 4 bytes = 0.18 GB (estimate). The peak with four
workers was not recorded; the README estimates a few GB, and 16 GB of RAM for 224 px.

## Testimony clustering

Measured on 2026-11-11 with `/usr/bin/time -v` on a laptop (Intel Core i7-1355U, 12 threads, 14 GB RAM, Python
3.14.4, scikit-learn 1.9.1):

```bash
PYTHONPATH=src python -m textclust data/testimonies.csv --output <dir> --quiet
```

| Measure | Value |
| --- | --- |
| Wall-clock time (load, TF-IDF, SVD, K-Means k = 2..20, HDBSCAN, t-SNE, plots) | 11.5 s |
| Peak resident memory | 237 MB |

Single measurement; not a load test. The input is small (1,011 rows), so run time is dominated by the 19 K-Means fits
and t-SNE.
