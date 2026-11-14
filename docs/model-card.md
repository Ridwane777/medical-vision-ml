# Model card: knee osteoarthritis classifier (HOG + SVC)

Structure after Mitchell et al., "Model Cards for Model Reporting" (2019). Every number comes from
[`results/knee/metrics.json`](../results/knee/metrics.json) and
[`comparison.md`](../results/knee/comparison.md) (full run of 2026-11-11).

## Model details

- **Developer**: [Ridwane777](https://github.com/Ridwane777).
- **Version**: full run `configs/full.json`, 2026-11-11, package `src/kneeclf`.
- **Type**: binary classifier. Features: a HOG descriptor written in NumPy (9 bins, 8x8 cells, 2x2 blocks, L2-Hys,
  6,084 dims at 112 px). Model: `StandardScaler` + `SVC` (RBF kernel, `C=10`, `gamma="scale"`), chosen among six
  candidates by validation ROC-AUC.
- **Output**: SVC decision score; OA if score >= **-0.2003** (threshold tuned on validation).
- **Software**: Python 3.12.3, NumPy 2.5.4, scikit-learn 1.9.1, pandas 3.1.1, Matplotlib 3.11.2, Pillow 13.0.0.
- **License of the code**: see the repository. The model file (`model.pkl`, about 200 MB) is not versioned.

## Intended use

- Research, teaching and portfolio: showing how far classical, non-deep-learning features go on knee X-rays, and how
  to report a medical classifier with a recall-first protocol.

## Not intended use

- **Not a medical device and not a diagnostic tool.** It must not be used to diagnose, triage or screen patients, nor
  to support any clinical decision.
- Not validated on any image source other than the Kaggle dataset below, nor on other joints, views or populations.

## Factors

- **Label**: KL 0-1 = no OA (0), KL 2-4 = OA (1), the usual radiographic definition (definite osteophyte), so the
  metrics can be read against the literature. KL 1 ("doubtful") sits on the border and is expected to carry most
  errors; the alternative scheme KL 0 vs. KL 1-4 is available with `--label-scheme kl0_vs_1to4`.
- No demographic metadata (age, sex, site, device) is available in the dataset, so no subgroup analysis is possible.

## Training and evaluation data

Kaggle [Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity),
224x224 grayscale PNGs, one folder per KL grade. See the [data card](data-cards/knee-xrays.md).

| Split | Files | Kept | No OA (KL 0-1) | OA (KL 2-4) | Note |
| --- | --- | --- | --- | --- | --- |
| Train | 5,778 | 4,892 | 2,446 | 2,446 | balanced by undersampling KL 0-1 |
| Validation | 826 | 826 | 481 | 345 | real ratio, used for model and threshold choice |
| Test | 1,656 | 1,656 | 935 | 721 | real ratio, used **once** at the end |

Images are downscaled to 112x112 by block mean before feature extraction.

## Metrics

Recall (sensitivity) is the primary metric: a missed OA case costs more than a false alarm. The threshold is the
highest one that still reaches **recall 0.85 on validation**; ROC-AUC ranks the candidates because it does not depend
on the threshold.

| Metric | Validation | Test |
| --- | --- | --- |
| Recall (sensitivity) | 0.852 | **0.831** |
| Specificity | 0.674 | 0.700 |
| Precision | 0.652 | 0.681 |
| F1 | 0.739 | 0.748 |
| F2 | 0.803 | 0.796 |
| Accuracy | 0.748 | 0.757 |
| Balanced accuracy | 0.763 | 0.765 |
| ROC-AUC | 0.861 | **0.861** |
| PR-AUC | 0.837 | 0.854 |
| Confusion (TN / FP / FN / TP) | 324 / 157 / 51 / 294 | 654 / 281 / 122 / 599 |

Cross-validation on train (3 folds) for the selected model: ROC-AUC 0.861, average precision 0.877.

Candidates on validation (each at its own recall-0.85 threshold):

| Candidate | CV ROC-AUC | Val ROC-AUC | Val specificity |
| --- | --- | --- | --- |
| HOG + SVC (selected) | 0.861 | 0.861 | 0.674 |
| HOG + LogReg | 0.854 | 0.857 | 0.645 |
| HOG + RandomForest | 0.822 | 0.838 | 0.634 |
| HOG + PCA + HistGradientBoosting | 0.802 | 0.809 | 0.534 |
| raw pixels + PCA + SVC | 0.780 | 0.792 | 0.551 |
| raw pixels + PCA + LogReg | 0.786 | 0.779 | 0.482 |

Figures: [test curves](../results/knee/figures/test_curves.png),
[test confusion](../results/knee/figures/test_confusion.png),
[validation curves](../results/knee/figures/val_curves.png).

## Limitations

- **Classical features only** (a design choice: CPU-only, interpretable, small dependency set): HOG captures edges and
  texture, not the joint space width a radiologist measures. ROC-AUC 0.861 is a baseline, below what fine-tuned CNNs
  report on the same task. A frozen ImageNet ResNet-50 did worse (test ROC-AUC 0.835, see
  [deep embeddings](../experiments/deep_embeddings/README.md)).
- **112 px**: half the native resolution, chosen for run time and memory. Fine osteophytes may be lost.
- **Single dataset, single source**: no external validation. Possible acquisition bias (devices, sites, cropping,
  exposure) that the model may exploit and that would not transfer.
- **Threshold drift**: 0.85 recall on validation became **0.831** on test. A threshold must be re-checked on every new
  population; recall is not guaranteed.
- **About 30% false positives** among healthy knees on test (specificity 0.700).
- Train is undersampled (886 KL 0-1 images dropped), so the model saw less of the healthy class variety.

## Ethical considerations

- Medical images of real patients, redistributed by Kaggle under CC BY 4.0; no identity data in the files. This
  repository does not store the dataset; only six test-split images are copied as demo examples, with attribution
  ([`app/examples/ATTRIBUTION.md`](../app/examples/ATTRIBUTION.md)).
- The main harm would be misuse as a diagnostic aid: a missed OA case (122 of 721 on test) or a false alarm (281 of
  935). Hence the "not intended use" section. The public demo (CLI `predict`, the static web demo and the local Gradio app) prints a
  "not a medical device" disclaimer with the test recall and specificity next to every output.
- Fairness across age, sex or ethnicity cannot be measured: the dataset has no such metadata.

## How to reproduce

```bash
PYTHONPATH=src python -m kneeclf --data data/knee --config configs/full.json --n-jobs 4
```

Configuration: [`configs/full.json`](../configs/full.json) (copy used: [`results/knee/config.json`](../results/knee/config.json)),
seed 42. Run time: **2,188 s (about 36 min)** on an 8-core CPU server with 4 workers, no GPU
([performance](performance/README.md)). The viewer notebook
[`notebooks/01_knee_classification.ipynb`](../notebooks/01_knee_classification.ipynb) displays the
versioned results without the dataset or the model file.

## Demo model (HOG + logistic regression, distributed)

The 200 MB SVC above is not distributed. The `predict` command and the online demo use a lighter model trained with
the same protocol, [`configs/demo.json`](../configs/demo.json) (only the `hog_logreg` candidate; same label scheme,
112 px, balanced train, recall target 0.85 on validation, seed 42). Numbers from
[`results/knee-demo/metrics.json`](../results/knee-demo/metrics.json) (run of 2026-11-14):

| | Value |
| --- | --- |
| Pipeline | HOG (6,084 dims) + `StandardScaler` + `LogisticRegression(C=0.0003, class_weight="balanced")` |
| Threshold | decision score >= -0.4418, i.e. probability >= 0.391 |
| Validation | ROC-AUC 0.857, recall 0.852, specificity 0.645 |
| **Test (1,656 images, used once)** | **recall 0.811, specificity 0.719**, precision 0.690, F2 0.784, ROC-AUC 0.854, PR-AUC 0.845, balanced accuracy 0.765 |
| Test confusion matrix | TN 672 · FP 263 · FN 136 · TP 585 |
| Files | `kneeclf-hog-logreg.pkl`, 172,307 bytes (168 KB), and the portable `kneeclf-hog-logreg.json`, 376,862 bytes (368 KB), each with a `.sha256` file |
| Training time | 25.1 s total (26.8 s wall clock, 1.0 GB peak memory), laptop, 12 logical CPUs, 3 workers |
| Software | Python 3.14.4, NumPy 2.5.4, scikit-learn 1.9.1 |

Compared with HOG + SVC on the same test set: ROC-AUC 0.854 vs. 0.861, recall 0.811 vs. 0.831, specificity
0.719 vs. 0.700, for a file about 1,200 times smaller and a 25 s training run.

The exported file is a dict (`format: "kneeclf-model/1"`) holding the scikit-learn pipeline (HOG extractor + scaler
+ logistic regression), the HOG parameters, the image size, the label scheme, the threshold, a metrics summary and
the library versions. It is a **pickle**: loading it can run code, so only load it from a trusted source (your own
export or the project's GitHub release). `kneeclf.inference.load_bundle` refuses the file when its `.sha256` does
not match, and `download-model` verifies the checksum before moving the file into place. Load it with the same
scikit-learn version (1.9.1); other versions may warn or fail.

Reproduce and export:

```bash
PYTHONPATH=src python -m kneeclf train --data data/knee --config configs/demo.json --n-jobs 3
PYTHONPATH=src python -m kneeclf export runs/<timestamp> --out dist/kneeclf-hog-logreg.pkl
PYTHONPATH=src python -m kneeclf export-json      # dist/kneeclf-hog-logreg.pkl -> dist/kneeclf-hog-logreg.json
```

### Portable JSON model (no pickle)

`kneeclf-hog-logreg.json` (`format: "kneeclf-portable/1"`) holds only plain numbers and strings: HOG parameters
(cell 8, block 2, 9 bins, L2-Hys clip 0.2), input and model image sizes (224 then 112 px), label scheme and class
names, the `StandardScaler` mean and scale (6,084 values each), the logistic regression coefficients and intercept
(with the dtype they were fitted in, float32), the threshold, the metrics summary and the training library
versions. Reading it cannot execute code and needs no scikit-learn.

Two predictors read it:

- `src/kneeclf/portable.py` (numpy + Pillow): reproduces the pipeline's `decision_function` with the same
  arithmetic. On the 1,656 real test images its scores are **identical** to the pickle pipeline's (maximum absolute
  difference 0.0, checked locally on 2026-11-14), so recall 0.8114 and specificity 0.7187 are unchanged.
  `tests/test_portable.py` checks the same on synthetic images (difference below 1e-6).
- `demo/knee.js` (plain JavaScript, used by the static web demo): a port of the same code in float64. On the six
  example images its decision scores differ from Python's by at most 4.3e-6, far from the threshold;
  `tests/test_portable.py` checks the port with Node.js when it is installed (tolerance 1e-4). Images whose side is
  not 224 px are resized with a JavaScript box filter that can differ from Pillow's by one gray level.
