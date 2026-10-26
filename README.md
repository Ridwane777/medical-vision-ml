# Medical Vision ML

Classical machine learning on medical data, without deep learning: binary classification of knee X-rays (healthy vs. osteoarthritis), an exploratory analysis of chest X-rays (pneumonia), and unsupervised clustering of free-text patient testimonies.

The goal is to show how far scikit-learn baselines go on raw medical images and text, and to make their limits measurable.

## Project structure

| Part | Location | What it does |
| --- | --- | --- |
| Supervised learning | `supervised_learning/dataset_knee_osteoarthritis.ipynb` | Knee X-rays regrouped into Healthy (KL grade 0) vs. Sick (grades 1-4), balanced, loaded as 224x224 arrays, flattened and normalized. Compares SVM, Random Forest and Logistic Regression, then tunes them with `GridSearchCV`. |
| Legacy EDA | `legacy/dataset_pneumonia_eda.ipynb` | Earlier exploratory analysis of chest X-rays (NORMAL vs. PNEUMONIA): split sizes, class balance, image sizes, samples. |
| Unsupervised learning | `unsupervised_learning/` | Text cleaning, TF-IDF, TruncatedSVD (LSA), L2 normalization, K-Means (k = 2..15, elbow + silhouette), t-SNE visualization. Script, notebook and [detailed README](unsupervised_learning/README.md). |

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

- 1002 testimonies after cleaning (deduplication, removal of empty rows).
- TF-IDF + TruncatedSVD to 100 components: explained variance 0.413.
- Final K-Means with **k = 14**: silhouette score **0.068** (low: clusters overlap, as expected for short free text).
- Cluster sizes range from 39 to 130 testimonies.
- Clusters are thematically readable, for example: `knee, ache, deep, leg, hip`; `wrist, forearm, hand, thumb, finger`; `itchy, rash, new, hives, raised`; `breath, chest, short, cough, feel`; `pressure, blood, chest, tightness, constant`.

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

**Patient testimonies**: `Student_Dataset.csv` is not distributed. To run the clustering, provide a CSV at `unsupervised_learning/Student_Dataset.csv` with no header and three columns:

| Column | Content |
| --- | --- |
| `ID` | integer identifier |
| `Code` | hexadecimal code such as `0x000000` (may be missing) |
| `Testimonies` | free English text describing symptoms |

Malformed rows are handled by the cleaning step (see the [unsupervised README](unsupervised_learning/README.md)).

## Setup

Requires Python 3.9 or newer.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open the notebooks, or run the clustering script from its folder:

```bash
cd unsupervised_learning && python unsupervised_learning.py
```

Main libraries: NumPy, pandas, Matplotlib, Pillow, scikit-learn, Jupyter.

## Author

Author: [Ridwane777](https://github.com/Ridwane777)
