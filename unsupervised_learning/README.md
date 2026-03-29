# Unsupervised Learning - Student Dataset Clustering

Ce dossier contient un pipeline complet de regroupement (clustering) non supervisé sur des témoignages d'étudiants. Le script principal est `unsupervised_learning.py` et il reproduit l'ensemble du travail initialement réalisé dans le notebook `Student_Dataset_preprocessing.ipynb`.

## Objectif

L'objectif est de détecter des groupes thématiques dans des témoignages textuels liés à des pathologies. On passe d'une tâche de prétraitement textuel à un clustering K-Means, puis à l'identification de thèmes dominants et à une visualisation t-SNE.

## Structure du workflow

### 1. Chargement et nettoyage des données
- Lecture initiale du CSV `../Student_Dataset.csv` pour un premier aperçu.
- Rechargement en forçant 3 colonnes (`ID`, `Code`, `Testimonies`) avec `on_bad_lines='skip'` pour ignorer les lignes corrompues.
- Correction des lignes décalées : si `Code` ne contient pas `0x`, la valeur est déplacée dans `Testimonies` et `Code` est fixé à `0x000000`.
- Suppression des lignes sans témoignage (`dropna`) et des doublons textuels exacts (`drop_duplicates`).
- Remplissage des codes manquants restants par `0x000000`.

### 2. Prétraitement textuel (`clean_text`)
- Retourne `0x000000` si l'entrée n'est pas une chaîne valide ou est vide.
- Passage en minuscules.
- Suppression de tout caractère non alphabétique via `re.sub(r'[^a-z\s]', ' ', text)`.
- Normalisation des espaces multiples.
- Tokenisation par `split()` et suppression des stop words anglais via la liste `ENGLISH_STOP_WORDS` de sklearn.
- Si le texte résultant est vide, remplacement par `0x000000`.
- Sauvegarde du dataset nettoyé dans `Student_Dataset_Cleaned.csv`.

### 2.a Définitions (NLP de base)
- `Tokenisation` : Processus de segmentation d'un flux textuel en unités minimales de sens appelées tokens (mots). C'est l'étape de structuration indispensable qui transforme une chaîne de caractères brute en une séquence exploitable par un algorithme.
- `Stop words` : Mots très fréquents et sans valeur informative (ex: "the", "is", "and") que l'on retire pour ne conserver que les termes porteurs de sens.

### 2.b Pourquoi ces étapes ?
- Suppression de la casse et de la ponctuation pour normaliser le texte.
- Retrait des stop words pour ne pas faire pondérer des mots sans valeur informative.

### 3. Vectorisation
- TF-IDF avec `TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')` :
  - `max_df=0.8` : ignore les mots présents dans plus de 80% des documents (trop communs).
  - `min_df=2` : ignore les mots présents dans moins de 2 documents (trop rares).
- Visualisation des 10 mots avec le poids TF-IDF moyen le plus élevé et le plus faible (graphiques côte à côte).
- Réduction de dimension par `TruncatedSVD` à 100 composantes (~30% de variance expliquée).
- Normalisation L2 avec `normalize(..., norm='l2')` pour rendre les distances cohérentes (similitude cosinus implicite).

### 3.a Définitions techniques
- `TF-IDF (Term Frequency-Inverse Document Frequency)` : Méthode de pondération statistique qui évalue l'importance d'un mot au sein d'un document relativement à un corpus complet. Elle multiplie la fréquence locale (TF) par la rareté globale (IDF), permettant ainsi de filtrer les mots trop communs au profit des termes discriminants.

- `TruncatedSVD (Singular Value Decomposition)` : Technique de réduction de dimension linéaire optimisée pour les matrices creuses (type TF-IDF). Elle projette les données dans un espace latent de dimension réduite en ne conservant que les composantes les plus significatives. Dans le cadre du NLP, on appelle souvent cette approche LSA (Latent Semantic Analysis).

- `Normalisation L2 (Euclidienne)` : Transformation vectorielle consistant à diviser chaque composante d'un vecteur par sa norme euclidienne, ramenant sa longueur totale à 1. Elle garantit que la comparaison entre deux documents repose sur la thématique et non sur la longueur brute du texte.

- `Silhouette Score` : Métrique qui permet d'évaluer la qualité d'un partitionnement en mesurant si chaque point est bien à sa place dans son cluster (cohésion interne vs séparation externe). Un score plus élevé indique un meilleur clustering.

### 4. Sélection du nombre de clusters
- Boucle k=2..15 sur K-Means (`n_init=10`, `random_state=42`).
- Deux méthodes complémentaires :
  - **Inertie (elbow/knee method)** : recherche un coude dans la courbe d'inertie.
  - **Score silhouette** : évalue la cohésion et la séparation des clusters.
- Visualisation des deux courbes côte à côte.

### 5. K-Means final (k=14)
- Après analyse des deux courbes sur k=2..15, **k=14** est retenu comme offrant le meilleur équilibre entre compacité et séparation.
- Chaque témoignage reçoit une étiquette `Cluster_Predit`.
- Calcul du score silhouette final sur le modèle retenu.
- Pour chaque cluster : calcul de la moyenne des scores TF-IDF et extraction des **5 mots les plus représentatifs**.

### 6. Visualisation t-SNE
- Réduction en 2D avec `TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)`.
- Scatter plot coloré par cluster avec palette `tab20` (14 couleurs distinctes) et barre de couleur numérotée de 0 à 13.

## Librairies utilisées

- `pandas` : manipulation de données tabulaires
- `numpy` : calculs numériques
- `matplotlib` : visualisation
- `re` : expressions régulières pour le nettoyage textuel
- `scikit-learn` : TF-IDF, stop words, SVD, normalisation, KMeans, silhouette score, t-SNE

## Exécution

1. Active l'environnement virtuel (si nécessaire) :
   ```bash
   source venv/bin/activate
   ```
2. Lance le script :
   ```bash
   python unsupervised_learning/unsupervised_learning.py
   ```
3. Vérifie `Student_Dataset_Cleaned.csv` et les graphiques.

## Conseils pour la revue de code

- Explique clairement pourquoi on nettoie au niveau du dataset (`0x000000`, mauvais alignement colonnes, suppression des doublons).
- Montre que la vectorisation TF-IDF est nécessaire pour transformer le texte en features numériques.
- Mentionne que SVD est utilisé pour réduire la dimensionnalité et éviter un clustering peu pertinent dans un espace trop grand.
- Présente le score silhouette comme métrique de qualité de regroupement.
- Si besoin, propose une extension : test d'autres algorithmes (DBSCAN, Agglomerative) pour comparaison.

---
