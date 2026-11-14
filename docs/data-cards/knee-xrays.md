# Data card: knee X-rays (KL graded)

Short datasheet after Gebru et al., "Datasheets for Datasets" (2021).

## Source and license

- Kaggle: [Knee Osteoarthritis Dataset with Severity Grading](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity),
  uploaded by Shashwat Tiwari. License shown on Kaggle: **Attribution 4.0 International (CC BY 4.0)** (checked through
  the Kaggle API on 2026-11-11).
- Acknowledged origin: Chen, Pingjun (2018), "Knee Osteoarthritis Severity Grading Dataset", Mendeley Data, V1,
  doi: [10.17632/56rmx5bjcr.1](https://doi.org/10.17632/56rmx5bjcr.1).
- Not stored in this repository (git-ignored); download it into `data/knee/` (see [data/README.md](../../data/README.md))
  or pass its root with `--data`.

## Composition

224x224 8-bit grayscale PNGs of knee joints, one folder per Kellgren-Lawrence grade, layout
`<root>/{train,val,test,auto_test}/{0,1,2,3,4}/*.png`. Counts from the data profile of the full run:

| Split | KL 0 | KL 1 | KL 2 | KL 3 | KL 4 | Total |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 2,286 | 1,046 | 1,516 | 757 | 173 | 5,778 |
| Validation | 328 | 153 | 212 | 106 | 27 | 826 |
| Test | 639 | 296 | 447 | 223 | 51 | 1,656 |

`auto_test` is not used. No patient metadata (age, sex, side, site, device) comes with the images.

## Preprocessing in this project

1. Binary label: KL 0-1 = no OA, KL 2-4 = OA, the radiographic definition of OA (definite osteophyte). KL 0 vs. 1-4
   remains available as `--label-scheme kl0_vs_1to4`.
2. Train balanced by random undersampling of the majority class (seed 42): 4,892 images kept, 2,446 per class
   (KL 0: 1,689 of 2,286 kept; KL 1: 757 of 1,046). Validation and test are kept whole, with their real ratio.
3. Pixels as float32 in [0, 1], downscaled to 112x112 by block mean.

## Known issues

- KL grading is subjective; KL 1 is "doubtful" by definition, so the 0-1 / 2-4 border carries label noise.
- Grade 4 is rare (173 train images).
- One source only: acquisition and cropping biases are unknown and cannot be checked without metadata.
- The provenance chain (Kaggle copy of a Mendeley record) means the original split procedure is not documented here;
  patient overlap between splits cannot be ruled out from the files alone.
