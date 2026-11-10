# Medical Vision ML

Classical machine learning on medical data, without deep learning: binary classification of knee X-rays (healthy vs. osteoarthritis), an exploratory analysis of chest X-rays (pneumonia), and unsupervised clustering of free-text patient testimonies.

The goal is to show how far scikit-learn baselines go on raw medical images and text, and to make their limits measurable.

## Project structure

| Part | Location | What it does |
| --- | --- | --- |
| Supervised learning | `supervised_learning/dataset_knee_osteoarthritis.ipynb` | Knee X-rays regrouped into Healthy (KL grade 0) vs. Sick (grades 1-4), balanced, loaded as 224x224 arrays, flattened and normalized. Compares SVM, Random Forest and Logistic Regression, then tunes them with `GridSearchCV`. |
| Legacy EDA | `legacy/dataset_pneumonia_eda.ipynb` | Earlier exploratory analysis of chest X-rays (NORMAL vs. PNEUMONIA): split sizes, class balance, image sizes, samples. |
| Unsupervised learning | `src/textclust/`, `unsupervised_learning/` | Adaptive clustering of any CSV of texts: robust loading, cleaning, language detection, TF-IDF, TruncatedSVD (LSA), automatic choice of k and of K-Means vs. HDBSCAN, t-SNE, JSON report. Package, CLI, notebook, tests and [detailed README](unsupervised_learning/README.md). |
| Tests | `tests/` | `unittest` suite proving the clustering adapts to different CSV formats and corpora. |
| Decisions | `docs/adr/` | Architecture Decision Records. |

## Results

All figures below come from the saved notebook outputs.

### Knee osteoarthritis (binary)

Balanced dataset after regrouping grades 1-4 into "Sick":

| Split | Healthy | Sick | Total |
| --- | --- | --- | --- |
| Train | 2286 | 2284 | 4570 |
| Validation | 328 | 325 | 653 |
| Test | 639 | 637 | 1276 |

Each image becomes a vector of 50,176 features (224 x 224 pixels).

Baseline accuracy on the validation set (default hyperparameters):

| Model | Validation accuracy |
| --- | --- |
| SVM (`SVC`, default RBF kernel) | 66.46% |
| Random Forest | 59.42% |
| Logistic Regression (`saga`) | not printed in the saved outputs |

The GridSearchCV, test-set evaluation and recall cells were not executed in the saved notebook, so no tuned parameters or test metrics are reported here. Run the notebook to reproduce them.

Takeaway: on flattened raw pixels, classical models stay well below what convolutional networks reach on this task; the notebook is a baseline, not a diagnostic tool.

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

**Knee X-rays**: Kaggle [Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity). Extract it so that the grade folders are at:

```
supervised_learning/Dataset_Knee_Osteoarthritis/{train,val,test}/{0,1,2,3,4}/
```

The first notebook cell builds `supervised_learning/Dataset_Knee_Osteoarthritis_Balanced/{train,val,test}/{healthy,sick}/` from it.

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

Then open the notebooks, or run the clustering from the repository root:

```bash
PYTHONPATH=src python -m textclust unsupervised_learning/Student_Dataset.csv
python unsupervised_learning/unsupervised_learning.py   # same, default dataset
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Libraries (project constraint, nothing else is imported): NumPy, pandas, Matplotlib, scikit-learn, JupyterLab, Pillow, and the Python standard library.

## Author

Author: [Ridwane777](https://github.com/Ridwane777)
