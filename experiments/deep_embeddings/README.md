# Classical vs. deep embeddings

The core (`src/kneeclf`, `src/textclust`) uses classical features on purpose: CPU-only, interpretable, and a small
dependency set (NumPy, pandas, Matplotlib, scikit-learn, Pillow, JupyterLab). This folder measures what deep embeddings
would add, with PyTorch, torchvision and sentence-transformers (`requirements-deep.txt`).

Rules: `src/` never imports from here; the scripts here import from `src/` so they use the same data loading,
evaluation and model selection as the core. The two runs are comparable, and the core and its tests never need torch.

## Run

```bash
python -m venv .venv-deep && .venv-deep/bin/pip install -r requirements-deep.txt
PYTHONPATH=src:. .venv-deep/bin/python experiments/deep_embeddings/knee_cnn_embeddings.py --data data/knee --threads 4
PYTHONPATH=src:. .venv-deep/bin/python experiments/deep_embeddings/text_sentence_embeddings.py data/testimonies.csv
```

Outputs go to `results/deep_embeddings/{knee,text}/` (metrics, comparison, figures). Embeddings are cached as `.npy`
under `runs/deep_cache/`, which is git-ignored. Model weights, embeddings and testimony text are not versioned.

## Knee X-rays: frozen ImageNet CNN + scikit-learn head

`knee_cnn_embeddings.py`: frozen ResNet-50 (IMAGENET1K_V2, 2048-d) and EfficientNet-B0 (IMAGENET1K_V1, 1280-d).
Grayscale copied to 3 channels, ImageNet normalization, native 224 px. The CNNs are used only as feature extractors.
Heads: StandardScaler + LogisticRegression (C in 0.001..1) and RBF SVC (C 1/10, gamma scale/1e-4), GridSearchCV with 3
folds on ROC-AUC. The rest of the protocol is the core's: KL 0-1 vs 2-4, train balanced (4,892), val 826, test 1,656,
threshold for recall 0.85 on validation, test used once for the candidate with the best validation ROC-AUC.

Validation (threshold at recall 0.85, so specificity is the number to compare):

| Candidate | Val ROC-AUC | Val specificity | Search time |
| --- | --- | --- | --- |
| Core: HOG + SVC (112 px) | **0.861** | **0.674** | 623 s |
| ResNet-50 + LogReg | 0.839 | 0.626 | 25 s |
| EfficientNet-B0 + LogReg | 0.838 | 0.607 | 9 s |
| EfficientNet-B0 + SVC | 0.837 | 0.632 | 161 s |
| ResNet-50 + SVC | 0.832 | 0.613 | 259 s |

Test (selected model only):

| Metric | Core HOG + SVC | ResNet-50 + LogReg |
| --- | --- | --- |
| Recall | **0.831** | 0.813 |
| Specificity | **0.700** | 0.665 |
| Precision | **0.681** | 0.652 |
| F2 | **0.796** | 0.775 |
| ROC-AUC / PR-AUC | **0.861 / 0.854** | 0.835 / 0.821 |
| Confusion matrix | TN 654 · FP 281 · FN 122 · TP 599 | TN 622 · FP 313 · FN 135 · TP 586 |

Cost on the 8-core CPU server (4 threads, `nice 19`, shared machine): embedding all 7,374 images took 1,762 s with
ResNet-50 (1,229 s for train) and 716 s with EfficientNet-B0. The whole script took 3,019 s (50 min), against 2,188 s
for the six-candidate core run. Model sizes without the classifier: ResNet-50 23.5 M parameters (89.7 MB fp32, 97.8 MB
download), EfficientNet-B0 4.0 M parameters (15.3 MB fp32). The core's HOG + SVC pickle is about 200 MB, because it
stores the support vectors.

Fine-tuning was **not run**: a forward pass alone over the training set took 20 min for ResNet-50 on this CPU. One
epoch of training the last block would add a backward pass on top of that, and several epochs are needed, which is
more than the 1 h budget. Without a GPU, fine-tuning is not practical here.

## Testimonies: sentence embeddings instead of TF-IDF + SVD

`text_sentence_embeddings.py`: the 994 testimonies kept by the core cleaning (`textclust.io` + `textclust.clean`). The
encoders read the original sentence, not the lowercased and stop-word-filtered one, because they use word order.
Vectors are L2-normalized, then the core's `model.select_model` runs on them: K-Means over k = 2..20 against HDBSCAN,
best cosine silhouette, at most 30% noise. Top terms come from the TF-IDF centroid of each cluster (core vocabulary).

| Representation | Dims | Chosen | k | Silhouette (cosine) | HDBSCAN (rejected) | ARI / NMI vs core | Encode time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Core TF-IDF + SVD | 100 | K-Means | 18 | 0.125 | 3 clusters, 77% noise | - | 6.6 s (whole selection) |
| all-MiniLM-L6-v2 (22.7 M params, 86.6 MB fp32) | 384 | K-Means | 4 | 0.169 | 4 clusters, 70% noise | 0.236 / 0.508 | 25.3 s |
| paraphrase-multilingual-MiniLM-L12-v2 (117.7 M params, 448.8 MB fp32) | 384 | K-Means | 3 | 0.193 | 3 clusters, 71% noise | 0.142 / 0.421 | 56.0 s |

Cluster sizes: MiniLM-L6 238 / 278 / 290 / 188; multilingual 279 / 521 / 194; core 27 to 96.
MiniLM-L6 themes (top TF-IDF terms): `feel, vision, pain, doctor, pressure, blood`; `cough, throat, cold, ear, fever,
chest`; `ache, sore, shoulder, sharp, deep`; `itchy, rash, skin, red, blister`. The multilingual model merges the
respiratory and general themes into one cluster of 521.

The silhouettes are computed in different spaces, so 0.169 vs 0.125 does **not** show that the embedding clusters are
better. What the numbers do show: the sentence encoders produce a few broad themes (body pain, respiratory/ENT, skin,
general) that the core's 18 clusters split finer (NMI 0.51). The partitions agree only weakly (ARI 0.14 to 0.24).

## Takeaway

- **Knees**: frozen ImageNet features did **worse** than the hand-written HOG on every test metric (ROC-AUC 0.835 vs
  0.861, specificity 0.665 vs 0.700 at similar recall). Natural-image features do not transfer well to the narrow
  joint-space cues of KL grading without fine-tuning, and fine-tuning needs a GPU. In the frozen setup, the deep stack brought no measured gain.
- **Text**: sentence embeddings give coarser, cleaner-looking themes and a higher silhouette in their own space, for
  an 87 to 449 MB download and about 6 to 11 times the run time (encode + selection: 37 s and 76 s vs 6.6 s). Whether they are "better" cannot be decided without labels;
  for exploration, the core's TF-IDF terms are more transparent.
