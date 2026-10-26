# Unsupervised Learning - Student Dataset Clustering

This folder contains a complete unsupervised clustering pipeline on student testimonies. The main script is `unsupervised_learning.py`; it reproduces all the work originally done in the notebook `Student_Dataset_preprocessing.ipynb`.

## Objective

The goal is to detect thematic groups in text testimonies related to medical conditions. The pipeline goes from text preprocessing to K-Means clustering, then to identifying dominant themes and a t-SNE visualization.

## Workflow structure

### 1. Loading and cleaning the data
- Initial read of the CSV `./Student_Dataset.csv` for a first overview.
- Reload forcing 3 columns (`ID`, `Code`, `Testimonies`) with `on_bad_lines='skip'` to ignore corrupted lines.
- Fix shifted rows: if `Code` does not contain `0x`, the value is moved to `Testimonies` and `Code` is set to `0x000000`.
- Remove rows without a testimony (`dropna`) and exact text duplicates (`drop_duplicates`).
- Fill remaining missing codes with `0x000000`.

### 2. Text preprocessing (`clean_text`)
- Returns `0x000000` if the input is not a valid string or is empty.
- Lowercasing.
- Removal of every non-alphabetic character via `re.sub(r'[^a-z\s]', ' ', text)`.
- Normalization of multiple spaces.
- Tokenization with `split()` and removal of English stop words using sklearn's `ENGLISH_STOP_WORDS` list.
- If the resulting text is empty, it is replaced by `0x000000`.
- The cleaned dataset is saved to `Student_Dataset_Cleaned.csv`.

### 2.a Definitions (basic NLP)
- `Tokenization`: splitting a text stream into minimal units of meaning called tokens (words). It is the structuring step that turns a raw character string into a sequence an algorithm can use.
- `Stop words`: very frequent words with no informative value (e.g. "the", "is", "and") that are removed to keep only meaningful terms.

### 2.b Why these steps?
- Removing case and punctuation normalizes the text.
- Removing stop words prevents uninformative words from carrying weight.

### 3. Vectorization
- TF-IDF with `TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')`:
  - `max_df=0.8`: ignores words present in more than 80% of documents (too common).
  - `min_df=2`: ignores words present in fewer than 2 documents (too rare).
- Visualization of the 10 words with the highest and the lowest mean TF-IDF weight (side-by-side charts).
- Dimensionality reduction with `TruncatedSVD` to 100 components (~41% explained variance, 0.413 in the notebook run).
- L2 normalization with `normalize(..., norm='l2')` to make distances consistent (implicit cosine similarity).

### 3.a Technical definitions
- `TF-IDF (Term Frequency-Inverse Document Frequency)`: statistical weighting method that measures how important a word is in a document relative to the whole corpus. It multiplies local frequency (TF) by global rarity (IDF), filtering out overly common words in favor of discriminative terms.

- `TruncatedSVD (Singular Value Decomposition)`: linear dimensionality reduction technique optimized for sparse matrices (such as TF-IDF). It projects the data into a lower-dimensional latent space, keeping only the most significant components. In NLP this approach is often called LSA (Latent Semantic Analysis).

- `L2 (Euclidean) normalization`: vector transformation that divides each component by the vector's Euclidean norm, bringing its total length to 1. It ensures that comparing two documents depends on their topic, not on their raw length.

- `Silhouette Score`: metric that evaluates the quality of a partition by measuring whether each point fits well in its cluster (internal cohesion vs. external separation). A higher score indicates better clustering.

### 4. Choosing the number of clusters
- Loop over k=2..15 with K-Means (`n_init=10`, `random_state=42`).
- Two complementary methods:
  - **Inertia (elbow/knee method)**: looks for an elbow in the inertia curve.
  - **Silhouette score**: evaluates cluster cohesion and separation.
- Both curves are plotted side by side.

### 5. Final K-Means (k=14)
- After analyzing both curves over k=2..15, **k=14** is kept as the best balance between compactness and separation.
- Each testimony receives a `Cluster_Predit` label.
- The final silhouette score is computed on the selected model.
- For each cluster: mean TF-IDF scores are computed and the **5 most representative words** are extracted.

### 6. t-SNE visualization
- 2D reduction with `TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)`.
- Scatter plot colored by cluster with the `tab20` palette (14 distinct colors) and a colorbar numbered 0 to 13.

## Libraries used

- `pandas`: tabular data manipulation
- `numpy`: numerical computing
- `matplotlib`: visualization
- `re`: regular expressions for text cleaning
- `scikit-learn`: TF-IDF, stop words, SVD, normalization, KMeans, silhouette score, t-SNE

## Running

1. Activate the virtual environment (if needed):
   ```bash
   source venv/bin/activate
   ```
2. Run the script:
   ```bash
   python unsupervised_learning/unsupervised_learning.py
   ```
3. Check `Student_Dataset_Cleaned.csv` and the plots.

## Code review notes

- Explain clearly why cleaning happens at the dataset level (`0x000000`, misaligned columns, duplicate removal).
- Show that TF-IDF vectorization is needed to turn text into numerical features.
- Mention that SVD is used to reduce dimensionality and avoid meaningless clustering in an overly large space.
- Present the silhouette score as the clustering quality metric.
- If needed, suggest an extension: testing other algorithms (DBSCAN, Agglomerative) for comparison.

---
