# =============================================================================
# UNSUPERVISED LEARNING - STUDENT DATASET PREPROCESSING & CLUSTERING
# =============================================================================
# Ce script explore un dataset CSV d'étudiants contenant des témoignages
# liés à des pathologies. Il réalise les étapes suivantes :
#   1. Chargement et nettoyage du dataset
#   2. Vectorisation TF-IDF des témoignages
#   3. Réduction de dimensionnalité (SVD)
#   4. Clustering K-Means avec sélection du k optimal
#   5. Visualisation des clusters avec t-SNE
# =============================================================================

# ─────────────────────────────────────────────
# SECTION 1 : IMPORTS
# ─────────────────────────────────────────────

import pandas as pd                                       # Manipulation de données tabulaires
import numpy as np                                        # Calculs numériques
import matplotlib.pyplot as plt                           # Visualisations graphiques
import re                                                 # Expressions régulières pour le nettoyage de texte

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize               # Normalisation L2 des vecteurs
from sklearn.decomposition import TruncatedSVD            # Réduction de dimensionnalité (LSA)
from sklearn.cluster import KMeans                        # Algorithme de clustering K-Means
from sklearn.metrics import silhouette_score              # Métrique de qualité des clusters
from sklearn.manifold import TSNE                         # Réduction 2D pour la visualisation


# =============================================================================
# SECTION 2 : CHARGEMENT ET APERÇU DU DATASET BRUT
# =============================================================================

# Chargement initial du fichier CSV pour un premier aperçu
# Le dataset contient des irrégularités : certaines lignes ont 2 colonnes,
# d'autres 3, avec des valeurs "0x000000" et des données manquantes.
df = pd.read_csv('../Student_Dataset.csv')

print("Aperçu des premières lignes (df.head) :")
print(df.head())


# =============================================================================
# SECTION 3 : NETTOYAGE ROBUSTE DU DATASET
# =============================================================================
# Stratégie de nettoyage :
#   - Corriger les lignes mal formées (colonnes décalées)
#   - Normaliser les codes manquants en '0x000000'
#   - Supprimer les témoignages vides et les doublons
#   - Passer en minuscules, supprimer la ponctuation et les caractères non-alpha
#   - Retirer les stop words anglais (via sklearn)
#   - Remplacer les témoignages vides restants par '0x000000'
# ─────────────────────────────────────────────

# Lecture du CSV avec gestion des lignes mal formatées
# - header=None : pas d'en-tête dans le fichier source
# - names=['ID', 'Code', 'Testimonies'] : on nomme manuellement les 3 colonnes
# - on_bad_lines='skip' : ignorer silencieusement les lignes corrompues
df_clean = pd.read_csv(
    '../Student_Dataset.csv',
    header=None,
    names=['ID', 'Code', 'Testimonies'],
    on_bad_lines='skip'
)

# ── Correction des lignes décalées ──────────────────────────────────────────
# Certaines lignes n'ont que 2 colonnes : l'ID et le témoignage (sans code 0x).
# On détecte ces lignes : la colonne 'Code' ne contient pas '0x'.
mask_decalage = ~df_clean['Code'].astype(str).str.contains('0x', na=False)

# Pour ces lignes : on déplace la valeur de 'Code' vers 'Testimonies'
# et on force la colonne 'Code' à la valeur neutre '0x000000'
df_clean.loc[mask_decalage, 'Testimonies'] = df_clean.loc[mask_decalage, 'Code']
df_clean.loc[mask_decalage, 'Code'] = '0x000000'

# ── Suppression des valeurs manquantes et des doublons ──────────────────────
# On supprime les lignes sans témoignage (NaN dans 'Testimonies')
df_final = df_clean.dropna(subset=['Testimonies']).copy()

# On supprime les témoignages en double (même texte)
df_final = df_final.drop_duplicates(subset=['Testimonies'])

# Remplir les codes manquants restants par la valeur par défaut
df_final['Code'] = df_final['Code'].fillna('0x000000')


# ── Fonction de nettoyage du texte ──────────────────────────────────────────
def clean_text(text):
    """
    Nettoie un témoignage textuel :
    - Vérifie que l'entrée est bien une chaîne non vide
    - Convertit en minuscules
    - Supprime tout caractère non alphabétique (ponctuation, chiffres, etc.)
    - Normalise les espaces multiples
    - Tokenise par split simple
    - Supprime les stop words anglais de sklearn
    - Retourne '0x000000' si le résultat est vide
    """
    # Si la valeur n'est pas une chaîne valide, retourner le code vide
    if not isinstance(text, str) or text.strip() == '':
        return '0x000000'

    # Mise en minuscules
    text = text.lower()

    # Garder uniquement les lettres et les espaces (supprimer ponctuation, chiffres...)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Normaliser les espaces multiples en un seul espace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenisation : séparation par espace et suppression des stop words
    tokens = [
        tok for tok in text.split(' ')
        if tok and tok not in ENGLISH_STOP_WORDS
    ]

    # Reconstruction du texte nettoyé
    cleaned = ' '.join(tokens).strip()

    # Si après nettoyage le texte est vide, retourner le code par défaut
    return cleaned if cleaned else '0x000000'


# Application du nettoyage à toute la colonne 'Testimonies'
df_final['Testimonies'] = df_final['Testimonies'].apply(clean_text)

# Sauvegarde du dataset nettoyé dans un nouveau fichier CSV
df_final.to_csv('Student_Dataset_Cleaned.csv', index=False)

# Métriques de nettoyage
print('Nombre de lignes après nettoyage :', len(df_final))
print('Codes distincts :', df_final['Code'].nunique())
print(df_final.head())


# =============================================================================
# SECTION 4 : VECTORISATION TF-IDF ET RÉDUCTION DE DIMENSIONNALITÉ
# =============================================================================
# On transforme les textes nettoyés en vecteurs numériques via TF-IDF,
# puis on réduit la dimensionnalité avec TruncatedSVD (LSA)
# et on normalise avec L2 pour préparer le K-Means basé sur la cosine distance.
# ─────────────────────────────────────────────

# ── TF-IDF Vectorization ────────────────────────────────────────────────────
# max_df=0.8 : ignorer les mots présents dans plus de 80% des documents (trop communs)
# min_df=2   : ignorer les mots présents dans moins de 2 documents (trop rares)
# stop_words='english' : supprimer les stop words anglais intégrés
vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
X = vectorizer.fit_transform(df_final['Testimonies'])

# ── Réduction de dimensionnalité par SVD tronquée (LSA) ─────────────────────
# On réduit l'espace TF-IDF (potentiellement des milliers de dimensions)
# à 100 composantes en conservant l'essentiel de la variance (≈30%)
n_components = 100
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_reduced = svd.fit_transform(X)

# ── Normalisation L2 ─────────────────────────────────────────────────────────
# K-Means fonctionne mieux avec la similarité cosinus sur les textes.
# La normalisation L2 convertit les vecteurs en unitaires, ce qui revient
# à utiliser la distance cosinus comme métrique implicite.
X_normalized = normalize(X_reduced, norm='l2', axis=1)

print(f"Matrice TF-IDF normalisée : {X_normalized.shape}")
print(f"Variance expliquée par les {n_components} composantes : {svd.explained_variance_ratio_.sum():.3f}")


# =============================================================================
# SECTION 5 : VISUALISATION DES POIDS TF-IDF (TOP 10 / BOTTOM 10)
# =============================================================================
# Affichage des 10 mots avec le score TF-IDF moyen le plus élevé (significatifs)
# et des 10 mots avec le score le plus bas (très communs ou très rares).
# ─────────────────────────────────────────────

# Récupération des noms de features (mots du vocabulaire TF-IDF)
feature_names = vectorizer.get_feature_names_out()

# Calcul du poids moyen de chaque mot sur l'ensemble du corpus
mean_weights = np.asarray(X.mean(axis=0)).ravel()

# Construction d'un DataFrame pour faciliter le tri
df_weights = pd.DataFrame({'word': feature_names, 'weight': mean_weights})

# Sélection des 10 mots les plus et les moins importants
df_top = df_weights.sort_values(by='weight', ascending=False).head(10)
df_bottom = df_weights.sort_values(by='weight', ascending=True).head(10)

# ── Affichage côte à côte des deux graphiques ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Graphique 1 : mots les plus significatifs (poids élevé)
ax1.barh(df_top['word'], df_top['weight'], color='skyblue')
ax1.set_title('Top 10: Words with the most weight (Significant)')
ax1.invert_yaxis()   # Inverser pour afficher le mot le plus important en haut
ax1.set_xlabel('Score TF-IDF moyen')

# Graphique 2 : mots les moins significatifs (poids faible)
ax2.barh(df_bottom['word'], df_bottom['weight'], color='salmon')
ax2.set_title('Bottom 10: Words with the least weight (Common/Rare)')
ax2.invert_yaxis()
ax2.set_xlabel('Score TF-IDF moyen')

plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 6 : DÉTERMINATION DU NOMBRE OPTIMAL DE CLUSTERS (k)
# =============================================================================
# Deux méthodes complémentaires pour choisir k :
#   - Méthode du coude (Elbow / Knee) : on cherche le point d'inflexion de l'inertie
#   - Score Silhouette : mesure la cohésion interne et la séparation entre clusters
#     (plus le score est élevé, meilleure est la séparation)
# ─────────────────────────────────────────────

inertie = []             # Stocke l'inertie (somme des distances intra-cluster) pour chaque k
silhouette_scores = []   # Stocke le score silhouette pour chaque k
K_range = range(2, 16)   # Tester de k=2 à k=15

for k in K_range:
    # Entraînement du modèle K-Means pour k clusters
    # n_init=10 : effectue 10 initialisations aléatoires et garde la meilleure
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_normalized)

    # Enregistrement de l'inertie du modèle
    inertie.append(model.inertia_)

    # Calcul du score silhouette (métrique euclidienne sur vecteurs normalisés)
    score = silhouette_score(X_normalized, labels, metric='euclidean')
    silhouette_scores.append(score)

# ── Affichage des deux méthodes côte à côte ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Graphique 1 : Méthode du coude
ax1.plot(K_range, inertie, 'bx-', markersize=8, linewidth=2)
ax1.set_xticks(K_range)
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Knee Method for determining the optimal k (up to 15)')
ax1.grid(True)

# Graphique 2 : Score Silhouette
ax2.plot(K_range, silhouette_scores, 'rx-', markersize=8, linewidth=2)
ax2.set_xticks(K_range)
ax2.set_xlabel('Number of clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis for Optimal k (up to 15)')
ax2.grid(True)

plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 7 : CLUSTERING FINAL AVEC k=14
# =============================================================================
# Après analyse des méthodes du coude et silhouette sur k=2..15,
# k=14 offre le meilleur équilibre entre compacité et séparation des clusters.
# ─────────────────────────────────────────────

# Entraînement du modèle K-Means final avec 14 clusters
kmeans_final = KMeans(n_clusters=14, random_state=42, n_init=10)

# Assignation de chaque témoignage à un cluster (colonne 'Cluster_Predit')
df_final['Cluster_Predit'] = kmeans_final.fit_predict(X_normalized)

# Affichage de la distribution des témoignages par cluster
print("Distribution of testimonies across the 14 clusters:")
print(df_final['Cluster_Predit'].value_counts().sort_index().to_string())

# Score silhouette global du modèle final
final_silhouette = silhouette_score(X_normalized, df_final['Cluster_Predit'], metric='euclidean')
print(f"\nFinal Silhouette Score: {final_silhouette:.3f}")

# ── Analyse thématique : top 5 mots par cluster ──────────────────────────────
# Pour chaque cluster, on calcule la moyenne des scores TF-IDF des témoignages
# qui lui appartiennent, et on extrait les 5 mots les plus représentatifs.
distances = kmeans_final.transform(X_normalized)
feature_names = vectorizer.get_feature_names_out()

for i in range(14):
    cluster_indices = df_final['Cluster_Predit'] == i
    if cluster_indices.sum() > 0:
        # Moyenne des vecteurs TF-IDF pour les témoignages de ce cluster
        cluster_tfidf = X[cluster_indices.values].mean(axis=0).A1

        # Indices des 5 mots avec les scores moyens les plus élevés
        top_indices = cluster_tfidf.argsort()[-5:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"CLUSTER {i} : {', '.join(top_words)}")
    else:
        print(f"CLUSTER {i} : No items found")

print("\n" + "-" * 60)


# =============================================================================
# SECTION 8 : VISUALISATION DES CLUSTERS AVEC t-SNE
# =============================================================================
# t-SNE (t-distributed Stochastic Neighbor Embedding) réduit les vecteurs
# normalisés en 2 dimensions pour permettre une visualisation lisible des clusters.
# - perplexity=30 : équilibre entre structure locale et globale (recommandé 5-50)
# - max_iter=1000 : nombre d'itérations pour la convergence
# - cmap='tab20'  : palette de 20 couleurs pour distinguer les 14 clusters
# ─────────────────────────────────────────────

# Réduction 2D avec t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
coords = tsne.fit_transform(X_normalized)

# Labels de couleur : indice de cluster pour chaque point
clusters_couleurs = df_final['Cluster_Predit']

plt.figure(figsize=(14, 10))

# Nuage de points coloré par cluster
scatter = plt.scatter(
    coords[:, 0], coords[:, 1],
    c=clusters_couleurs,
    cmap='tab20',       # Palette avec au moins 14 couleurs distinctes
    alpha=0.8,          # Légère transparence pour voir les superpositions
    edgecolors='w',     # Bordure blanche pour mieux distinguer les points
    s=70                # Taille des points
)

plt.title("Visualization of the 14 Testimonial Clusters (t-SNE)")
plt.xlabel("t-SNE axis 1")
plt.ylabel("t-SNE axis 2")

# Barre de couleur numérotée de 0 à 13 pour identifier chaque cluster
ticks = list(range(14))
cb = plt.colorbar(scatter, ticks=ticks, label="Cluster Number")
cb.set_ticklabels(ticks)

plt.grid(True, linestyle='--', alpha=0.3)
plt.show()
