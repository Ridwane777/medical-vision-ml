# ADR-0002: Binary label scheme for knee X-rays (KL >= 2 vs. KL >= 1)

- **Status**: Accepted
- **Date**: 2026-11-10
- **Deciders**: Ridwane777

## Context

The Kaggle "Knee Osteoarthritis Dataset with Severity Grading" gives each 224x224 grayscale knee X-ray a
Kellgren-Lawrence (KL) grade from 0 to 4 (train: 2286 / 1046 / 1516 / 757 / 173 images; val: 826; test: 1656).
The project asks for a binary healthy vs. sick classifier, so the five grades must be grouped.

The first notebook (`supervised_learning/dataset_knee_osteoarthritis.ipynb`, now removed and replaced by
`knee_classification.ipynb` and the `kneeclf` package) used **KL 0 = healthy, KL 1-4 = sick**, balanced every split
(train, validation and test) by undersampling, flattened the 224x224 pixels, and compared SVM, Random Forest and
Logistic Regression. Best validation accuracy: SVM 66.46% (Random Forest 59.42%). Grid search and test evaluation
were never executed in the saved outputs.

KL 1 means "doubtful joint space narrowing and possible osteophytic lipping". In epidemiology and in most imaging
studies, radiographic osteoarthritis is defined as **KL >= 2** (definite osteophyte). KL 1 images are visually close to
KL 0, so labelling them "sick" puts the decision border inside the noisiest grade.

## Options considered

1. **KL 0 vs. KL 1-4** (previous). Keeps all early signs as positives (maximal sensitivity to "anything"), but the
   positive class mixes doubtful cases with severe OA, and the border KL0/KL1 is where graders disagree most.
   Train split: 2286 negatives vs. 3492 positives.
2. **KL 0-1 vs. KL 2-4** (clinical definition). Matches the radiographic OA definition used in the literature, so
   results are comparable and clinically meaningful. Train split: 3332 negatives vs. 2446 positives.
3. **Drop KL 1 entirely** (KL 0 vs. KL 2-4). Cleaner classes and easier task, but the model never sees the hardest
   real cases and its reported metrics would be optimistic for real use.
4. **Multi-class (5 grades) or ordinal regression.** Closest to the data, but out of the binary scope of the project
   and much harder with classical features (KL 3/4 vs. 4 has only 173 training images for grade 4).

## Decision

Option 2 is the default (`label_scheme = "kl01_vs_2to4"`). Option 1 stays available (`"kl0_vs_1to4"`,
`--label-scheme kl0_vs_1to4`) so the previous approach can be reproduced and compared. In both cases, balancing by
undersampling is applied to **train only**: validation and test keep their natural class ratio, unlike the previous
notebook, because a balanced test set misstates precision and the expected number of false alarms.

## Consequences

Gained:
- ✅ The positive class means "radiographic OA" as clinicians define it; metrics can be read against the literature.
- ✅ Validation and test reflect the real prevalence of the dataset, so precision and specificity are honest.
- ✅ The scheme is a configuration value, tested on synthetic data for both mappings.

Lost or accepted:
- ❌ KL 1 knees (doubtful early OA) count as negatives: a screening use case that wants to catch them must switch the
  scheme or lower the threshold.
- ❌ The KL1/KL2 border is still ambiguous; most errors are expected there.
- ❌ Numbers are not directly comparable with the previous 66.46% accuracy (different labels, unbalanced validation,
  different metric).
