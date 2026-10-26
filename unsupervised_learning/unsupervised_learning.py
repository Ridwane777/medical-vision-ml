# =============================================================================
# UNSUPERVISED LEARNING - STUDENT DATASET PREPROCESSING & CLUSTERING
# =============================================================================
# This script explores a CSV dataset of student testimonies related to
# medical conditions. It performs the following steps:
#   1. Loading and cleaning the dataset
#   2. TF-IDF vectorization of the testimonies
#   3. Dimensionality reduction (SVD)
#   4. K-Means clustering with selection of the optimal k
#   5. Cluster visualization with t-SNE
# =============================================================================

# ─────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────

import pandas as pd                                       # Tabular data manipulation
import numpy as np                                        # Numerical computing
import matplotlib.pyplot as plt                           # Plotting
import re                                                 # Regular expressions for text cleaning

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize               # L2 normalization of vectors
from sklearn.decomposition import TruncatedSVD            # Dimensionality reduction (LSA)
from sklearn.cluster import KMeans                        # K-Means clustering algorithm
from sklearn.metrics import silhouette_score              # Cluster quality metric
from sklearn.manifold import TSNE                         # 2D reduction for visualization


# =============================================================================
# SECTION 2: LOADING AND OVERVIEW OF THE RAW DATASET
# =============================================================================

# Initial load of the CSV file for a first overview
# The dataset has irregularities: some rows have 2 columns,
# others 3, with "0x000000" values and missing data.
df = pd.read_csv('./Student_Dataset.csv')

print("Preview of the first rows (df.head):")
print(df.head())


# =============================================================================
# SECTION 3: ROBUST DATASET CLEANING
# =============================================================================
# Cleaning strategy:
#   - Fix malformed rows (shifted columns)
#   - Normalize missing codes to '0x000000'
#   - Remove empty testimonies and duplicates
#   - Lowercase, remove punctuation and non-alphabetic characters
#   - Remove English stop words (via sklearn)
#   - Replace remaining empty testimonies with '0x000000'
# ─────────────────────────────────────────────

# Read the CSV while handling malformed rows
# - header=None: no header in the source file
# - names=['ID', 'Code', 'Testimonies']: name the 3 columns manually
# - on_bad_lines='skip': silently ignore corrupted rows
df_clean = pd.read_csv(
    './Student_Dataset.csv',
    header=None,
    names=['ID', 'Code', 'Testimonies'],
    on_bad_lines='skip'
)

# ── Fixing shifted rows ──────────────────────────────────────────
# Some rows only have 2 columns: the ID and the testimony (no 0x code).
# Detect them: the 'Code' column does not contain '0x'.
mask_decalage = ~df_clean['Code'].astype(str).str.contains('0x', na=False)

# For these rows: move the 'Code' value to 'Testimonies'
# and set the 'Code' column to the neutral value '0x000000'
df_clean.loc[mask_decalage, 'Testimonies'] = df_clean.loc[mask_decalage, 'Code']
df_clean.loc[mask_decalage, 'Code'] = '0x000000'

# ── Removing missing values and duplicates ──────────────────────
# Drop rows without a testimony (NaN in 'Testimonies')
df_final = df_clean.dropna(subset=['Testimonies']).copy()

# Drop duplicate testimonies (same text)
df_final = df_final.drop_duplicates(subset=['Testimonies'])

# Fill remaining missing codes with the default value
df_final['Code'] = df_final['Code'].fillna('0x000000')


# ── Text cleaning function ──────────────────────────────────────────
def clean_text(text):
    """
    Clean a text testimony:
    - Check that the input is a non-empty string
    - Convert to lowercase
    - Remove every non-alphabetic character (punctuation, digits, etc.)
    - Normalize multiple spaces
    - Tokenize with a simple split
    - Remove sklearn's English stop words
    - Return '0x000000' if the result is empty
    """
    # If the value is not a valid string, return the empty code
    if not isinstance(text, str) or text.strip() == '':
        return '0x000000'

    # Lowercase
    text = text.lower()

    # Keep only letters and spaces (remove punctuation, digits...)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization: split on spaces and remove stop words
    tokens = [
        tok for tok in text.split(' ')
        if tok and tok not in ENGLISH_STOP_WORDS
    ]

    # Rebuild the cleaned text
    cleaned = ' '.join(tokens).strip()

    # If the text is empty after cleaning, return the default code
    return cleaned if cleaned else '0x000000'


# Apply the cleaning to the whole 'Testimonies' column
df_final['Testimonies'] = df_final['Testimonies'].apply(clean_text)

# Save the cleaned dataset to a new CSV file
df_final.to_csv('Student_Dataset_Cleaned.csv', index=False)

# Cleaning metrics
print('Number of rows after cleaning:', len(df_final))
print('Distinct codes:', df_final['Code'].nunique())
print(df_final.head())


# =============================================================================
# SECTION 4: TF-IDF VECTORIZATION AND DIMENSIONALITY REDUCTION
# =============================================================================
# Turn the cleaned texts into numerical vectors with TF-IDF,
# then reduce dimensionality with TruncatedSVD (LSA)
# and apply L2 normalization to prepare cosine-distance-based K-Means.
# ─────────────────────────────────────────────

# ── TF-IDF Vectorization ────────────────────────────────────────────────────
# max_df=0.8: ignore words present in more than 80% of documents (too common)
# min_df=2  : ignore words present in fewer than 2 documents (too rare)
# stop_words='english': remove built-in English stop words
vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df_final['Testimonies'])

# ── Dimensionality reduction with truncated SVD (LSA) ─────────────────────
# Reduce the TF-IDF space (potentially thousands of dimensions)
# to 100 components while keeping ~41% of the variance (0.413 in the notebook run)
n_components = 100
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(X)

# ── L2 normalization ─────────────────────────────────────────────────────────
# K-Means works better with cosine similarity on text.
# L2 normalization turns vectors into unit vectors, which amounts
# to using cosine distance as an implicit metric.
X_normalized = normalize(X_reduced, norm='l2', axis=1)

print(f"Normalized TF-IDF matrix: {X_normalized.shape}")
print(f"Variance explained by the {n_components} components: {svd.explained_variance_ratio_.sum():.3f}")


# =============================================================================
# SECTION 5: TF-IDF WEIGHT VISUALIZATION (TOP 10 / BOTTOM 10)
# =============================================================================
# Show the 10 words with the highest mean TF-IDF score (significant)
# and the 10 words with the lowest score (very common or very rare).
# ─────────────────────────────────────────────

# Get feature names (words of the TF-IDF vocabulary)
feature_names = vectorizer.get_feature_names_out()

# Compute the mean weight of each word over the whole corpus
mean_weights = np.asarray(X.mean(axis=0)).ravel()

# Build a DataFrame to make sorting easier
df_weights = pd.DataFrame({'word': feature_names, 'weight': mean_weights})

# Select the 10 most and least important words
df_top = df_weights.sort_values(by='weight', ascending=False).head(10)
df_bottom = df_weights.sort_values(by='weight', ascending=True).head(10)

# ── Plot both charts side by side ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Chart 1: most significant words (high weight)
ax1.barh(df_top['word'], df_top['weight'], color='skyblue')
ax1.set_title('Top 10: Words with the most weight (Significant)')
ax1.invert_yaxis()   # Invert so the most important word is on top
ax1.set_xlabel('Mean TF-IDF score')

# Chart 2: least significant words (low weight)
ax2.barh(df_bottom['word'], df_bottom['weight'], color='salmon')
ax2.set_title('Bottom 10: Words with the least weight (Common/Rare)')
ax2.invert_yaxis()
ax2.set_xlabel('Mean TF-IDF score')

plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 6: FINDING THE OPTIMAL NUMBER OF CLUSTERS (k)
# =============================================================================
# Two complementary methods to choose k:
#   - Elbow / Knee method: look for the inflection point of the inertia
#   - Silhouette score: measures internal cohesion and separation between clusters
#     (the higher the score, the better the separation)
# ─────────────────────────────────────────────

inertie = []             # Inertia (sum of intra-cluster distances) for each k
silhouette_scores = []   # Silhouette score for each k
K_range = range(2, 16)   # Test k=2 to k=15

for k in K_range:
    # Train the K-Means model for k clusters
    # n_init=10: run 10 random initializations and keep the best one
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_normalized)

    # Record the model inertia
    inertie.append(model.inertia_)

    # Compute the silhouette score (Euclidean metric on normalized vectors)
    score = silhouette_score(X_normalized, labels, metric='euclidean')
    silhouette_scores.append(score)

# ── Plot both methods side by side ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Chart 1: elbow method
ax1.plot(K_range, inertie, 'bx-', markersize=8, linewidth=2)
ax1.set_xticks(K_range)
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Knee Method for determining the optimal k (up to 15)')
ax1.grid(True)

# Chart 2: silhouette score
ax2.plot(K_range, silhouette_scores, 'rx-', markersize=8, linewidth=2)
ax2.set_xticks(K_range)
ax2.set_xlabel('Number of clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis for Optimal k (up to 15)')
ax2.grid(True)

plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 7: FINAL CLUSTERING WITH k=14
# =============================================================================
# After analyzing the elbow and silhouette methods over k=2..15,
# k=14 gives the best balance between cluster compactness and separation.
# ─────────────────────────────────────────────

# Train the final K-Means model with 14 clusters
kmeans_final = KMeans(n_clusters=14, random_state=42, n_init=10)

# Assign each testimony to a cluster ('Cluster_Predit' column)
df_final['Cluster_Predit'] = kmeans_final.fit_predict(X_normalized)

# Show the distribution of testimonies per cluster
print("Distribution of testimonies across the 14 clusters:")
print(df_final['Cluster_Predit'].value_counts().sort_index().to_string())

# Overall silhouette score of the final model
final_silhouette = silhouette_score(X_normalized, df_final['Cluster_Predit'], metric='euclidean')
print(f"\nFinal Silhouette Score: {final_silhouette:.3f}")

# ── Thematic analysis: top 5 words per cluster ──────────────────────────────
# For each cluster, compute the mean TF-IDF scores of its testimonies
# and extract the 5 most representative words.
distances = kmeans_final.transform(X_normalized)
feature_names = vectorizer.get_feature_names_out()

for i in range(14):
    cluster_indices = df_final['Cluster_Predit'] == i
    if cluster_indices.sum() > 0:
        # Mean TF-IDF vector for the testimonies of this cluster
        cluster_tfidf = X[cluster_indices.values].mean(axis=0).A1

        # Indices of the 5 words with the highest mean scores
        top_indices = cluster_tfidf.argsort()[-5:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"CLUSTER {i} : {', '.join(top_words)}")
    else:
        print(f"CLUSTER {i} : No items found")

print("\n" + "-" * 60)


# =============================================================================
# SECTION 8: CLUSTER VISUALIZATION WITH t-SNE
# =============================================================================
# t-SNE (t-distributed Stochastic Neighbor Embedding) reduces the normalized
# vectors to 2 dimensions for a readable view of the clusters.
# - perplexity=30: balance between local and global structure (recommended 5-50)
# - max_iter=1000: number of iterations for convergence
# - cmap='tab20' : 20-color palette to distinguish the 14 clusters
# ─────────────────────────────────────────────

# 2D reduction with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
coords = tsne.fit_transform(X_normalized)

# Color labels: cluster index for each point
clusters_couleurs = df_final['Cluster_Predit']

plt.figure(figsize=(14, 10))

# Scatter plot colored by cluster
scatter = plt.scatter(
    coords[:, 0], coords[:, 1],
    c=clusters_couleurs,
    cmap='tab20',       # Palette with at least 14 distinct colors
    alpha=0.8,          # Slight transparency to show overlaps
    edgecolors='w',     # White edge to tell points apart
    s=70                # Point size
)

plt.title("Visualization of the 14 Testimonial Clusters (t-SNE)")
plt.xlabel("t-SNE axis 1")
plt.ylabel("t-SNE axis 2")

# Colorbar numbered 0 to 13 to identify each cluster
ticks = list(range(14))
cb = plt.colorbar(scatter, ticks=ticks, label="Cluster Number")
cb.set_ticklabels(ticks)

plt.grid(True, linestyle='--', alpha=0.3)
plt.show()
