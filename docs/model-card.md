# Model card: knee osteoarthritis classifier (HOG + SVC)

Structure after Mitchell et al., "Model Cards for Model Reporting" (2019). Every number comes from
[`results/supervised/metrics.json`](../results/supervised/metrics.json) and
[`comparison.md`](../results/supervised/comparison.md) (full run of 2026-11-11).

## Model details

- **Developer**: [Ridwane777](https://github.com/Ridwane777). Portfolio refactor of a school project.
- **Version**: full run `configs/full.json`, 2026-11-11, package `src/kneeclf`.
- **Type**: binary classifier. Features: a HOG descriptor written in NumPy (9 bins, 8x8 cells, 2x2 blocks, L2-Hys,
  6,084 dims at 112 px). Model: `StandardScaler` + `SVC` (RBF kernel, `C=10`, `gamma="scale"`), chosen among six
  candidates by validation ROC-AUC ([ADR-0003](adr/0003-classical-features-and-recall-first-evaluation.md)).
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

- **Label**: KL 0-1 = no OA (0), KL 2-4 = OA (1), the usual radiographic definition
  ([ADR-0002](adr/0002-binary-label-scheme.md)). KL 1 ("doubtful") sits on the border and is expected to carry most
  errors.
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

Figures: [test curves](../results/supervised/figures/test_curves.png),
[test confusion](../results/supervised/figures/test_confusion.png),
[validation curves](../results/supervised/figures/val_curves.png).

## Limitations

- **Classical features only** (project constraint): HOG captures edges and texture, not the joint space width a
  radiologist measures. ROC-AUC 0.861 is a baseline, far from what CNNs report on the same task.
- **112 px**: half the native resolution, chosen for run time and memory. Fine osteophytes may be lost.
- **Single dataset, single source**: no external validation. Possible acquisition bias (devices, sites, cropping,
  exposure) that the model may exploit and that would not transfer.
- **Threshold drift**: 0.85 recall on validation became **0.831** on test. A threshold must be re-checked on every new
  population; recall is not guaranteed.
- **About 30% false positives** among healthy knees on test (specificity 0.700).
- Train is undersampled (886 KL 0-1 images dropped), so the model saw less of the healthy class variety.

## Ethical considerations

- Medical images of real patients, redistributed by Kaggle under CC BY 4.0; no identity data in the files. This
  repository does not store the images.
- The main harm would be misuse as a diagnostic aid: a missed OA case (122 of 721 on test) or a false alarm (281 of
  935). Hence the "not intended use" section and no packaged inference service.
- Fairness across age, sex or ethnicity cannot be measured: the dataset has no such metadata.

## How to reproduce

```bash
PYTHONPATH=src python -m kneeclf --data /path/to/Dataset_Knee_Osteoarthritis --config configs/full.json --n-jobs 4
```

Configuration: [`configs/full.json`](../configs/full.json) (copy used: [`results/supervised/config.json`](../results/supervised/config.json)),
seed 42. Run time: **2,188 s (about 36 min)** on an 8-core CPU server with 4 workers, no GPU
([performance](performance/README.md)). The viewer notebook
[`supervised_learning/knee_classification.ipynb`](../supervised_learning/knee_classification.ipynb) displays the
versioned results without the dataset or the model file.
