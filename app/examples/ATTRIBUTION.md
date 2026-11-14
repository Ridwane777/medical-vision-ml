# Example images: source and license

The six knee X-rays in this folder are unmodified copies of images from the **test split** of:

> Chen, P. (2018). *Knee Osteoarthritis Severity Grading Dataset*. Mendeley Data, V1.
> doi:[10.17632/56rmx5bjcr.1](https://doi.org/10.17632/56rmx5bjcr.1). Redistributed via Kaggle.
> License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The images derive from the Osteoarthritis Initiative (OAI) cohort. They are shown here only to demonstrate the
model; the demo model never saw them during training or threshold selection (they come from the test split).

| File | True KL grade | Binary label (KL>=2) | Demo model score (threshold 0.391) | Model output |
| --- | --- | --- | --- | --- |
| `kl0_9003815R.png` | 0 | no OA | 0.177 | no (correct) |
| `kl1_9008934R.png` | 1 | no OA | 0.553 | yes (false positive) |
| `kl1_9018291L.png` | 1 | no OA | 0.117 | no (correct) |
| `kl2_9008884L.png` | 2 | OA | 0.293 | no (false negative) |
| `kl3_9037494L.png` | 3 | OA | 0.887 | yes (correct) |
| `kl4_9101066R.png` | 4 | OA | 0.991 | yes (correct) |

The examples were picked by position in the sorted file list (5th or 10th file of each grade folder), not by
model output: the two errors are kept on purpose, they show what a 0.81 recall / 0.72 specificity model does.
