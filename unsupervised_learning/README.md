# Unsupervised Learning - Student Dataset Clustering

Ce dossier contient un pipeline complet de regroupement (clustering) non supervisé sur des témoignages d\'étudiants. Le script principal est `unsupervised_learning.py` et il reproduit l\'ensemble du travail initialement réalisé dans le notebook `Student_Dataset_preprocessing.ipynb`.

## Objectif

L'objectif est de détecter des groupes thématiques dans des témoignages textuels liés à des pathologies. On passe d'une tâche de prétraitement textuel à un clustering k-means, puis à l'identification de thèmes dominants et à une visualisation t-SNE.

## Structure du workflow

### 1. Chargement et nettoyage des données (`load_and_clean`)
- Lecture du CSV `../Student_Dataset.csv` (en tête non standard, îl peut contenir des lignes mal formatées).
- Correction des lignes où la colonne `Code` ne correspond pas à un code `0x...` (cas décalés où le texte est dans `Code`).
- Suppression des lignes sans témoignage et suppression des doublons (mêmes textes).
- Remplacement des valeurs manquantes `Code` par `0x000000`.

### 2. Prétraitement textuel (`clean_text`)
- Passage en minuscules.
- Suppression de tout caractère non alphabétique (chiffres, ponctuation). 
- Tokenization avec NLTK (`word_tokenize`).
- Suppression des stopwords anglais (liste NLTK `stopwords`).
- Lemmatisation avec `WordNetLemmatizer` (réduit les mots à leur racine).
- Si le texte devient vide, remplacement par le placeholder `0x000000`.

### 2.a Définitions (NLP de base)
- `Tokenisation` : Processus de segmentation d'un flux textuel en unités minimales de sens appelées tokens (mots, caractères ou sous-mots). C'est l'étape de structuration indispensable qui transforme une chaîne de caractères brute en une séquence exploitable par un algorithme.
- `Lemmatisation` : Analyse morphosyntaxique visant à ramener un mot à sa forme de dictionnaire, appelée lemme (infinitif pour les verbes, masculin singulier pour les adjectifs/noms). Contrairement au stemming, elle s'appuie sur un dictionnaire et préserve la catégorie grammaticale, garantissant que le résultat est un mot existant et porteur de sens.

### 2.b Pourquoi ces étapes ?
- Suppression de la casse/ponctuation pour normaliser le texte.
- Retrait des stopwords pour ne pas faire pondérer des mots sans valeur informative.
- Lemmatisation pour regrouper des mots appartenant au même concept (run/running, patient/patients).

### 3. Vectorisation (`vectorize_data`)
- TF-IDF avec `TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')` :
  - max_df = enlève les mots très fréquents (bruit), min_df = enlève les mots trop rares.
- Réduction de dimension par `TruncatedSVD` à 100 composantes pour capter signal principal et réduire le bruit.
- Normalisation L2 avec `normalize(..., norm='l2')` pour rendre les distances cohérentes (recommandé pour la similitude cosinus).

Le TF-IDF crée la carte d'identité du texte, la TruncatedSVD en simplifie la lecture en regroupant les concepts proches, et la Normalisation L2 remet tous les documents sur un pied d'égalité pour la comparaison.

### 3.a Définitions techniques
- `TF-IDF (Term Frequency-Inverse Document Frequency)` : Méthode de pondération statistique qui évalue l'importance d'un mot au sein d'un document relativement à un corpus complet. Elle multiplie la fréquence locale (TF) par la rareté globale (IDF), permettant ainsi de filtrer les "mots outils" (le, la, de) au profit des termes porteurs de sens et discriminants.

- `TruncatedSVD (Singular Value Decomposition)` : Technique de réduction de dimension linéaire optimisée pour les matrices creuses (type TF-IDF). Elle projette les données dans un espace latent de dimension réduite en ne conservant que les composantes les plus significatives. Dans le cadre du NLP, on appelle souvent cette approche LSA (Latent Semantic Analysis) car elle permet de capturer des relations sémantiques entre les mots.

- `Normalisation L2 (Euclidienne)` : Transformation vectorielle consistant à diviser chaque composante d'un vecteur par sa norme euclidienne, ramenant sa longueur totale à 1. En recherche d'information, elle garantit que la comparaison entre deux documents (souvent via la similarité cosinus) repose sur la thématique et non sur la longueur brute du texte. 

- `Silhouette_score` : C'est une métrique qui permet d'évaluer la qualité d'un partitionnement (clustering) en mesurant si chaque point est bien à sa place.

### 4. Sélection du nombre de clusters (`choose_k` et `plot_k_selection`)
- Boucle k=2..15 sur K-means (`n_init=10`, `random_state=42`).
- Méthodes:
  - Inertie (elbow/knee method) : recherche un coude dans la courbe d\'inertie.
  - Score silhouette : évalue la cohésion et séparation des clusters.
- Visualisation en deux graphiques côte à côte.

### 5. K-means final et thème des clusters (`kmeans_and_report`)
- K-means final choisi avec k=7 (selon analyse elbow/silhouette).
- Étiquette `Cluster_Predit` pour chaque ligne.
- Mesure finale : score silhouette sur le clustering.
- Pour chaque cluster, recherche des 5 mots TF-IDF moyens les plus hauts pour identifier un thème.

### 6. Visualisation (`tsne_visualization`)
- Réduction en 2D avec `TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)`.
- Scatter plot coloré selon cluster.

### 7. Export de résultats
- Fichier `Student_Dataset_Clustered.csv` contenant `ID`, `Code`, `Testimonies`, `Cluster_Predit`.

## Thèmes identifiés (exemple)

D'après l'exécution actuelle, les clusters reflètent approximativement :
- Cluster 0 : symptômes dermatologiques (peau, brûlure, rougeur)
- Cluster 1 : douleurs, oreilles, maux de tête
- Cluster 2 : poitrine, respiration, toux
- Cluster 3 : démangeaisons, éruption, allergie
- Cluster 4 : sang, médecin, soif, perte de poids
- Cluster 5 : douleurs générales (courbatures, sensation de malaise)
- Cluster 6 : gorge/nasal, toux légère

> Renote que ces thèmes sont suggérés par les mots les plus forts TF-IDF, ce n'est pas une étiquette médicale certifiée.

## Cas d'utilisation

- Analyse exploratoire de corpus de récits/patients.
- Classification préliminaire et regroupement de types de symptômes.
- Aide à la segmentation des témoignages pour un travail médical/épidémiologique.
- Création de rapports thématiques par regroupement documentaire.

## Librairies utilisées

- `pandas` : manipulation de tables
- `numpy` : calculs numériques
- `matplotlib` : visualisation
- `nltk` : prétraitement textuel (tokenization, stopwords, lemmatisation)
- `scikit-learn` : TF-IDF, SVD, normalization, KMeans, silhouette_score, TSNE

## Exécution

1. Active l'environnement virtuel (si nécessaire) :
   ```bash
   source venv/bin/activate
   ```
2. Lance le script :
   ```bash
   python unsupervised_learning/unsupervised_learning.py
   ```
3. Vérifie `Student_Dataset_Clustered.csv` et les graphiques.

## Conseils pour la revue de code

- Explique clairement pourquoi on clean au niveau du dataset (`0x000000`, mauvais alignement colonnes, suppression des doublons).
- Montre que la vectorisation TF-IDF est nécessaire pour transformer texte en features numériques.
- Mentionne que SVD est utilisé pour réduire la dimensionnalité et éviter overfitting/clustering futile.
- Présente le score silhouette comme métrique de qualité de regroupement.
- Si besoin, propose une extension : test de `k` additionnels (7, 8, 9), ou autres algos (DBSCAN, Agglomerative) pour comparaison.

---
