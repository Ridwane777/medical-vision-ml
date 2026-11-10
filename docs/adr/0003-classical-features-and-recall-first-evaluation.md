# ADR-0003: Classical image features and recall-first evaluation

- **Status**: Accepted
- **Date**: 2026-11-10
- **Deciders**: Ridwane777

## Context

Constraint from the project specification: only scikit-learn, pandas, NumPy, Matplotlib, JupyterLab, Pillow and the
Python standard library may be imported. No PyTorch/TensorFlow (so no CNN or pretrained network), no scikit-image or
OpenCV (so no ready-made HOG, LBP or CLAHE).

The previous notebook fed 50,176 raw pixels to SVM / Random Forest / Logistic Regression, chose models by
**accuracy**, and planned to evaluate on the test set inside the tuning loop. For a disease classifier, a missed OA
case (false negative) costs more than a false alarm, and accuracy hides that trade-off. Tuning anything on the test
set also biases the reported numbers.

## Options considered

Features:
1. **Raw pixels only** (previous). Simple, but pixels are not shift-invariant and 50k dimensions make kernels slow.
2. **Raw pixels + StandardScaler + PCA.** Compresses to 100-200 components, keeps the global shape.
3. **HOG implemented in NumPy** (central-difference gradients, 9 unsigned orientation bins with linear
   interpolation, 8x8-pixel cells, 2x2-cell blocks with L2-Hys normalization). Captures edges of the joint space and
   osteophytes, is robust to contrast changes, and is the standard pre-deep-learning descriptor for X-rays.
4. **Texture histograms (LBP, intensity)** in NumPy. Cheap; kept as optional candidates (`lbp_logreg`,
   `hog+lbp_svc`), not in the default set.
5. **CNN / transfer learning.** Best expected accuracy on this dataset but forbidden by the constraint.

Selection and threshold:
- A. Select by accuracy at the default 0.5 threshold (previous).
- B. Select by **ROC-AUC** in stratified cross-validation (threshold-independent), record recall, F1, balanced
  accuracy and average precision alongside, then **tune the decision threshold on validation** for a target recall
  (default 0.85) or for maximal F2.
- C. Weight classes or costs only (`class_weight`), keep 0.5. Partial: no explicit control of sensitivity.

## Decision

Features 2 and 3 (with 4 available), selection B. Six default candidates: raw->PCA->LogReg, raw->PCA->SVC (RBF),
HOG->LogReg, HOG->SVC (RBF), HOG->RandomForest, HOG->PCA->HistGradientBoosting, all with `class_weight="balanced"`,
each tuned by `GridSearchCV` (StratifiedKFold, refit on ROC-AUC). The best candidate is chosen by **validation**
ROC-AUC; its threshold is the highest one that still reaches the target recall on validation. The **test set is used
once**, after the model and threshold are frozen; all its metrics (accuracy, balanced accuracy, precision, recall,
specificity, F1, F2, ROC-AUC, PR-AUC, confusion matrix) and figures are written to `metrics.json` and `figures/`.

Feature extraction is stateless, so it is computed once per feature type instead of once per fold and grid point; the
saved model re-attaches the extractor in front of the tuned pipeline so it takes raw images.

Full runs use 112x112 images (block mean of 2x2 pixels). Memory estimate for the default scheme (train balanced to
2 x 2446 = 4892, val 826, test 1656, 7374 images): raw float32 7374 x 12544 x 4 B = 0.37 GB, HOG 6084 dims = 0.18 GB;
at 224 px it would be 1.48 GB raw and 26,244 HOG dims (0.77 GB) before the copies made by scaling, PCA and four
parallel workers. 112 px keeps the run comfortably under 8 GB of RAM.

## Consequences

Gained:
- ✅ Sensitivity is an explicit, configurable target instead of a by-product of accuracy.
- ✅ Test numbers are unbiased estimates: nothing is tuned on them.
- ✅ Six comparable candidates with timings in one table (`comparison.csv` / `.md`), reproducible from a JSON config.
- ✅ HOG is unit-tested (shape, zero output on a constant image, orientation, contrast invariance).

Lost or accepted:
- ❌ Classical features stay well below published CNN results on this dataset; this is a baseline.
- ❌ A threshold tuned for 0.85 recall on 826 validation images can transfer imperfectly (test recall a few points
  lower or higher).
- ❌ Ranking by validation ROC-AUC uses validation twice (model choice and threshold); a nested scheme or a separate
  calibration split would be cleaner.
- ❌ Downscaling to 112 px loses fine osteophyte detail; 224 px is available (`"image_size": null`) at about 4x the
  memory and time.
- ❌ Our NumPy HOG is slower and less optimized than scikit-image's, and without CLAHE the contrast varies between
  images.
