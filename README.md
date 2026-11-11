# Medical Vision ML

Classical machine learning on medical data, without deep learning: binary classification of knee X-rays (healthy vs. osteoarthritis), an exploratory analysis of chest X-rays (pneumonia), and unsupervised clustering of free-text patient testimonies.

The goal is to show how far scikit-learn baselines go on raw medical images and text, and to make their limits measurable.

**Context**: this started as a school project whose specification restricted the libraries to NumPy, pandas,
Matplotlib, scikit-learn, JupyterLab, Pillow and the Python standard library (no deep learning framework). The solo
refactor into tested packages keeps that constraint. Decisions, model card, data cards, performance and retrospective
are in [`docs/`](docs/README.md).

## Project structure

| Part | Location | What it does |
| --- | --- | --- |
| Supervised learning | `src/kneeclf/`, `configs/`, `supervised_learning/knee_classification.ipynb` | Knee X-rays, KL grades mapped to no OA (KL 0-1) vs. OA (KL 2-4). Raw pixels + PCA and a NumPy HOG descriptor, six scikit-learn candidates tuned by `GridSearchCV`, decision threshold tuned on validation for a target recall, test set used once. Package, CLI, viewer notebook and tests. |
| Legacy EDA | `legacy/dataset_pneumonia_eda.ipynb` | Earlier exploratory analysis of chest X-rays (NORMAL vs. PNEUMONIA): split sizes, class balance, image sizes, samples. |
| Unsupervised learning | `src/textclust/`, `unsupervised_learning/` | Adaptive clustering of any CSV of texts: robust loading, cleaning, language detection, TF-IDF, TruncatedSVD (LSA), automatic choice of k and of K-Means vs. HDBSCAN, t-SNE, JSON report. Package, CLI, notebook, tests and [detailed README](unsupervised_learning/README.md). |
| Tests | `tests/` | `unittest` suites: clustering adapts to different CSV formats; knee pipeline on synthetic images (loader, label schemes, balancing, HOG, threshold, end to end). |
| Documentation | [`docs/`](docs/README.md) | Architecture Decision Records, model card, data cards, performance, retrospective. |

## Results

### Knee osteoarthritis (binary)

> Full run of `configs/full.json` on 2026-11-11 (8-core CPU server, 4 workers, 36 min). Train: 4,892 images balanced
> 2,446 / 2,446; validation: 826 (481 / 345); test: 1,656 (935 / 721). Outputs: [`results/supervised/`](results/supervised/)
> (metrics, comparison, figures; the 200 MB model file is not versioned).

What the pipeline does (`src/kneeclf`):

1. **Labels** ([ADR-0002](docs/adr/0002-binary-label-scheme.md)): KL 0-1 = no OA, KL 2-4 = OA (radiographic
   definition). The previous scheme, KL 0 vs. 1-4, is still available with `--label-scheme kl0_vs_1to4`.
2. **Data**: 224x224 PNGs loaded with Pillow as float32 in [0, 1], downscaled by block mean (112 px for the full run).
   Train is balanced by undersampling; validation and test keep their real class ratio.
3. **Features and models** ([ADR-0003](docs/adr/0003-classical-features-and-recall-first-evaluation.md)): raw pixels
   -> StandardScaler -> PCA, and a HOG descriptor written in NumPy (9 bins, 8x8 cells, 2x2 blocks, L2-Hys). Six
   candidates: raw-PCA-LogReg, raw-PCA-SVC, HOG-LogReg, HOG-SVC, HOG-RandomForest, HOG-PCA-HistGradientBoosting, each
   tuned by `GridSearchCV` (stratified 3-fold, ROC-AUC; recall, F1, balanced accuracy and average precision recorded).
4. **Recall first**: the best candidate (validation ROC-AUC) gets the highest threshold that still reaches a recall of
   0.85 on validation (or the F2-optimal one). The **test set is used once**, at the end: accuracy, balanced accuracy,
   precision, recall (sensitivity), specificity, F1, F2, ROC-AUC, PR-AUC, confusion matrix, ROC/PR curves.

| Metric (test, selected model) | Value |
| --- | --- |
| Selected candidate | HOG + SVC (RBF, C=10) |
| Recall (sensitivity) | **0.831** |
| Specificity | 0.700 |
| Precision | 0.681 |
| F2 | 0.796 |
| ROC-AUC / PR-AUC | **0.861** / 0.854 |
| Balanced accuracy | 0.765 |
| Confusion matrix | TN 654 · FP 281 · FN 122 · TP 599 |

Candidates on validation (threshold set for recall 0.85, so specificity is the comparable number):

| Candidate | CV ROC-AUC | Val ROC-AUC | Val specificity | Search time |
| --- | --- | --- | --- | --- |
| HOG + SVC | 0.861 | **0.861** | **0.674** | 623 s |
| HOG + LogReg | 0.854 | 0.857 | 0.645 | 19 s |
| HOG + RandomForest | 0.822 | 0.838 | 0.634 | 409 s |
| HOG + PCA + HistGradientBoosting | 0.802 | 0.809 | 0.534 | 759 s |
| raw pixels + PCA + SVC | 0.780 | 0.792 | 0.551 | 70 s |
| raw pixels + PCA + LogReg | 0.786 | 0.779 | 0.482 | 132 s |

Takeaways: the hand-written HOG descriptor is the main gain (+0.07 ROC-AUC over raw pixels for the same model
family); HOG + LogReg comes within 0.004 ROC-AUC of the SVC for 30 times less search time, a reasonable choice if
inference cost matters. Test recall (0.831) lands slightly below the 0.85 validation target: the threshold does not
transfer perfectly. This remains a classical-ML baseline, not a diagnostic tool.

![ROC and precision-recall curves on the test set](results/supervised/figures/test_curves.png)

Previous approach, kept for the record (old notebook, KL 0 vs. 1-4, balanced validation, raw 224x224 pixels,
default hyperparameters): validation accuracy SVM 66.46%, Random Forest 59.42%. It is not comparable with the new
metrics (different labels and class ratio).

#### Run it

```bash
# smoke test, about 15 s on a laptop (150 images per class, 64 px, tiny grids; numbers are not results)
PYTHONPATH=src python -m kneeclf --data supervised_learning/Dataset_Knee_Osteoarthritis --config configs/quick.json

# full run (all images, 112 px, medium grids, 4 parallel jobs)
PYTHONPATH=src python -m kneeclf --data /path/to/Dataset_Knee_Osteoarthritis --config configs/full.json --n-jobs 4
```

On a remote server, detach the run and follow the log:

```bash
nohup env PYTHONPATH=src python -m kneeclf --data /path/to/Dataset_Knee_Osteoarthritis \
    --config configs/full.json --n-jobs 4 --output runs > kneeclf_full.log 2>&1 &
tail -f kneeclf_full.log            # or runs/<timestamp>/train.log
# with tmux instead: tmux new -s knee, run the same command without nohup, detach with Ctrl-b d
```

Each run writes `runs/<timestamp>/`: `metrics.json` (config, data profile, environment versions, per-candidate CV and
validation scores, test metrics, timings), `params.json`, `comparison.csv` / `comparison.md`, `model.pkl` (pickle:
feature extractor + tuned pipeline + threshold), `train.log` and `figures/`. Open
`supervised_learning/knee_classification.ipynb` to display a run.

Memory: the full run at 112 px holds about 7,400 images x 12,544 pixels x 4 bytes = 0.37 GB of pixels plus 0.18 GB of
HOG features (6,084 dims), a few GB at peak with four workers. At 224 px (`"image_size": null`) the arrays grow to
1.48 GB + 0.77 GB (26,244 HOG dims) before copies: plan for 16 GB of RAM.

### Chest X-ray pneumonia (EDA only)

| Split | NORMAL | PNEUMONIA | Total |
| --- | --- | --- | --- |
| Train | 3400 | 3400 | 6800 |
| Validation | 850 | 850 | 1700 |
| Test | 234 | 390 | 624 |

No model is trained in this notebook.

### Testimony clustering

The clustering is an adaptive pipeline (`src/textclust`) that works on any CSV of texts without code changes. Results on `Student_Dataset.csv` (run of 2026-11-10, details in the [unsupervised README](unsupervised_learning/README.md)):

- 1,011 rows loaded; 397 shifted rows (2 fields instead of 3) repaired automatically; 17 duplicates dropped, **994 testimonies** kept; language detected: English.
- The `Code` column is constant (`0x000000`), so it is ignored: **no external labels exist**, only internal metrics are reported.
- TF-IDF (1,271 terms) + TruncatedSVD to 100 components: explained variance 0.415.
- Automatic selection between K-Means (k = 2..20) and HDBSCAN: **K-Means, k = 18, cosine silhouette 0.125**, noise 0%. HDBSCAN reached 0.161 but put 77% of the testimonies in noise, so it was rejected.
- Cluster sizes from 27 to 96. Example themes: `itchy, rash, new, hives, raised`; `wrist, elbow, forearm, hand, palm`; `chest, tightness, tight, pressure`; `ear, pain, fever, hear`; `blood, doctor, sugar, high, pressure`.
- The silhouette plateaus around 0.12 from k = 13: the themes overlap, as expected for short free text.

#### Adapts to any CSV

Auto-detected: encoding (UTF-8 with or without BOM, Latin-1), delimiter (`,` `;` tab `|`), header row, ragged rows (the longest field becomes the text), text column, optional label column (categorical, not an ID, not constant), language and stop words (English or French), TF-IDF and SVD sizes, the range of k, and the algorithm. Override with `--text-column`, `--label-column`, `--no-labels`, `--k`, `--language`, or a `Config` object in Python.

```bash
PYTHONPATH=src python -m textclust path/to/file.csv [--text-column C] [--label-column C] [--output runs]
```

Each run writes `runs/<timestamp>/metrics.json` plus plots (git-ignored). The design decision is recorded in [ADR-0001](docs/adr/0001-adaptive-clustering-pipeline.md).

## Data

Datasets are not stored in this repository. Download them and place them as follows (paths are relative to the folder of each notebook, so launch Jupyter from the repository root and open the notebooks in place).

**Knee X-rays**: Kaggle [Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)
(224x224 8-bit grayscale PNG, one folder per KL grade). The dataset is not stored here; pass its root with `--data`.
Expected layout (`auto_test` is ignored unless listed in `extra_test_splits`):

```
<root>/{train,val,test,auto_test}/{0,1,2,3,4}/*.png
```

The default location `supervised_learning/Dataset_Knee_Osteoarthritis/` is git-ignored. Train counts per grade:
2286 / 1046 / 1516 / 757 / 173.

**Chest X-rays**: Kaggle [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Expected layout:

```
legacy/dataset_pneumonia/{train,val,test}/{NORMAL,PNEUMONIA}/
```

Note: the counts above (balanced 3400/3400 train, 850/850 validation) show the notebook was run on a re-split and balanced version of this dataset, not on the original Kaggle split.

**Patient testimonies**: `Student_Dataset.csv` is not distributed. Place it at `unsupervised_learning/Student_Dataset.csv`. The original file has no header and the columns `ID`, `Code` (always `0x000000`) and free English text, with some rows missing the `Code` field. Any other CSV of texts works too: the pipeline detects the layout (see "Adapts to any CSV" above).

## Setup

Requires Python 3.9 or newer and scikit-learn 1.3 or newer (for `HDBSCAN`).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Then open the notebooks, or run the pipelines from the repository root:

```bash
PYTHONPATH=src python -m textclust unsupervised_learning/Student_Dataset.csv
python unsupervised_learning/unsupervised_learning.py   # same, default dataset
PYTHONPATH=src python -m kneeclf --data supervised_learning/Dataset_Knee_Osteoarthritis --config configs/quick.json
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Libraries (project constraint, nothing else is imported): NumPy, pandas, Matplotlib, scikit-learn, JupyterLab, Pillow, and the Python standard library.

## Author

Author: [Ridwane777](https://github.com/Ridwane777)
