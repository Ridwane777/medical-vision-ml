##
## Ridwane777 PROJECT, 2026
## medical-vision-ml
## File description:
## unsupervised_learning
##

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


def clean_text(text):
    """Nettoie une chaîne textuelle:
    - convertit en minuscules
    - supprime les caractères non alphabétiques
    - tokenize
    - retire les stopwords
    - lemmatise les tokens
    """
    # Cas d'une valeur manquante ou vide : placeholder
    if not isinstance(text, str) or text.strip() == '':
        return '0x000000'

    # Mise en minuscule de la string
    text = text.lower()
    # Remplacement des signes de ponctuation et chiffres par des espaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Tokenization avec NLTK
    tokens = word_tokenize(text)
    stop_words_set = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Suppression des stopwords et des tokens courts de taille 1
    cleaned_tokens = [lemmatizer.lemmatize(tok) for tok in tokens
                      if tok not in stop_words_set and len(tok) > 1]

    cleaned = ' '.join(cleaned_tokens).strip()
    # Si tout est supprimé, garder le token placeholder
    return cleaned if cleaned else '0x000000'


def load_and_clean(input_csv_path):
    """Charge le dataset, corrige le format et nettoie les textes."""
    # Lecture sécurisée pour lignes malformées
    df_clean = pd.read_csv(input_csv_path, header=None,
                           names=['ID', 'Code', 'Testimonies'], on_bad_lines='skip')

    # Certaines lignes peuvent inverser Code et Testimonies ; on corrige si Code n'a pas le format attendu
    mask_decalage = ~df_clean['Code'].astype(str).str.contains('0x', na=False)
    df_clean.loc[mask_decalage, 'Testimonies'] = df_clean.loc[mask_decalage, 'Code']
    df_clean.loc[mask_decalage, 'Code'] = '0x000000'

    # Supprimer les lignes avec témoignage manquant et doublons pour éviter duplications artificielles
    df_final = df_clean.dropna(subset=['Testimonies']).copy()
    df_final = df_final.drop_duplicates(subset=['Testimonies'])
    df_final['Code'] = df_final['Code'].fillna('0x000000')

    # Application du nettoyage NLTK ligne par ligne
    df_final['Testimonies'] = df_final['Testimonies'].apply(clean_text)

    return df_final


def vectorize_data(df_text, max_df=0.8, min_df=2, n_components=100):
    """TF-IDF + SVD + normalisation.

    - `max_df` et `min_df` filtrent les mots trop fréquents ou trop rares.
    - Truncated SVD réduit la dimensionnalité pour atténuer le bruit.
    - Normalisation L2 standardise la longueur des vecteurs pour KMeans.
    """
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english')
    X = vectorizer.fit_transform(df_text)

    # Réduction des dimensions via SVD pour conserver l'essentiel de l'information
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)

    # Normalisation L2 : utile pour calcul de distance cohérente (cosine-like)
    X_normalized = normalize(X_reduced, norm='l2', axis=1)

    return X, X_normalized, vectorizer, svd


def choose_k(X_normalized, max_k=15):
    """Calcule l'inertie et silhouette pour k dans [2, max_k]."""
    inertie = []
    silhouette_scores = []
    K_range = list(range(2, max_k + 1))

    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_normalized)
        inertie.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X_normalized, labels, metric='euclidean'))

    return K_range, inertie, silhouette_scores


def plot_k_selection(K_range, inertie, silhouette_scores):
    """Affiche les courbes elbow et silhouette côte à côte."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(K_range, inertie, 'bx-', markersize=8, linewidth=2)
    ax1.set_xticks(K_range)
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Knee Method (Elbow)')
    ax1.grid(True)

    ax2.plot(K_range, silhouette_scores, 'rx-', markersize=8, linewidth=2)
    ax2.set_xticks(K_range)
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def kmeans_and_report(X_normalized, X_tfidf, df_final, n_clusters=7, vectorizer=None):
    """Exécute KMeans, calcule le score silhouette, et imprime des thèmes par cluster."""
    # Entraînement KMeans sur les vecteurs normalisés
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X_normalized)
    df_final['Cluster_Predit'] = cluster_labels

    # Distribution des points par cluster
    print('Distribution des clusters :')
    print(df_final['Cluster_Predit'].value_counts().sort_index())

    # Indicateur qualité
    final_silhouette = silhouette_score(X_normalized, cluster_labels, metric='euclidean')
    print(f'Final Silhouette Score: {final_silhouette:.3f}')

    # Thèmes par cluster (top mots TF-IDF)
    if vectorizer is not None:
        feature_names = vectorizer.get_feature_names_out()
        print('\nThèmes représentatifs par cluster :')
        for i in range(n_clusters):
            mask = df_final['Cluster_Predit'] == i
            if mask.sum() > 0:
                cluster_tfidf = X_tfidf[mask.values].mean(axis=0).A1
                top_idx = cluster_tfidf.argsort()[-5:][::-1]
                top_words = [feature_names[idx] for idx in top_idx]
                print(f'CLUSTER {i} : {", ".join(top_words)}')
            else:
                print(f'CLUSTER {i} : no data')
    else:
        print('Vectorizer non fourni : impossible d\'afficher les thèmes.')

    return kmeans_final, df_final


def tsne_visualization(X_normalized, df_final, n_clusters=7):
    """Visualisation t-SNE des clusters."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(X_normalized)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=df_final['Cluster_Predit'], cmap='tab10',
                          alpha=0.8, edgecolors='w', s=60)
    plt.title(f'Visualization of the {n_clusters} clusters (t-SNE)')
    plt.xlabel('t-SNE axis 1')
    plt.ylabel('t-SNE axis 2')
    plt.colorbar(scatter, ticks=range(n_clusters), label='Cluster Number')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()


def main():
    # Téléchargements NLTK nécessaires (une fois) pour tokenization/lemmatisation
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    # Chargement et pré-processing des données
    csv_path = '../Student_Dataset.csv'
    df_final = load_and_clean(csv_path)

    print('Nombre de lignes après nettoyage :', len(df_final))
    print('Codes distincts :', df_final['Code'].nunique())

    # Vectorisation et réduction de dimension
    X_tfidf, X_normalized, vectorizer, svd = vectorize_data(df_final['Testimonies'])
    print(f'TF-IDF réduit et normalisé : {X_normalized.shape}')
    print(f'Variance expliquée by SVD : {svd.explained_variance_ratio_.sum():.3f}')

    # Méthode d'évaluation pour choisir k (2..15)
    K_range, inertie, silhouette_scores = choose_k(X_normalized, max_k=15)
    plot_k_selection(K_range, inertie, silhouette_scores)

    # K final sélectionné suite à l'analyse des graphes
    k = 7

    # Appliquer KMeans, afficher rapport et thèmes
    kmeans_final, df_final = kmeans_and_report(X_normalized, X_tfidf, df_final,
                                               n_clusters=k, vectorizer=vectorizer)

    # Visualisation en 2D via t-SNE
    tsne_visualization(X_normalized, df_final, n_clusters=k)

    # Enregistrement du dataset avec clusters
    df_final.to_csv('Student_Dataset_Clustered.csv', index=False)


if __name__ == '__main__':
    main()
