import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.stats import spearmanr, shapiro, levene, f_oneway, kruskal, kstest
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # Nécessaire pour activer IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def test_shapiro_safe(data, max_sample_size=5000):
    """Test de Shapiro-Wilk sur un échantillon limité à max_sample_size pour éviter l'erreur de taille."""
    sample = data.sample(n=min(len(data), max_sample_size), random_state=42)
    return shapiro(sample)


def plot_pca_kmeans_projection(df, variables_pc_x, variables_pc_y, n_clusters=3, pc_x=0, pc_y=1):
    """
    Fonction pour réaliser la projection des individus sur un plan factoriel défini par PCX et PCY,
    en appliquant une ACP (PCA) sur un ensemble de variables et un clustering K-means.

    :param df: DataFrame pandas contenant les données brutes.
    :param variables_pc_x: Liste des variables utilisées pour l'axe X dans la PCA.
    :param variables_pc_y: Liste des variables utilisées pour l'axe Y dans la PCA.
    :param n_clusters: Nombre de clusters pour le K-means (par défaut 3).
    :param pc_x: Index de la composante principale à afficher sur l'axe X (ex: 0 pour PC1, 2 pour PC3).
    :param pc_y: Index de la composante principale à afficher sur l'axe Y (ex: 1 pour PC2, 3 pour PC4).
    """

    # Sélectionner les colonnes pour X et Y
    df_selected = df[variables_pc_x + variables_pc_y]

    # Imputation des valeurs manquantes par la moyenne
    imputer = SimpleImputer(strategy='mean')
    df_imputed = imputer.fit_transform(df_selected)

    # Standardisation des données
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_imputed)

    # Vérifier combien de composantes peuvent être extraites
    max_components = min(df_scaled.shape)  # min(n_samples, n_features)

    # Vérifier que les indices de composantes à afficher sont valides
    if pc_x >= max_components or pc_y >= max_components:
        raise ValueError(f"Impossible d'afficher PC{pc_x+1} et PC{pc_y+1}. "
                         f"Le maximum possible est PC{max_components}.")

    # Appliquer la PCA avec seulement le nombre de composantes nécessaires
    pca = PCA(n_components=max_components)
    pca_components = pca.fit_transform(df_scaled)

    # Appliquer K-means sur les données après PCA
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_components)

    # Affichage de la projection des individus
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_components[:, pc_x], pca_components[:, pc_y], c=clusters, cmap='viridis', alpha=0.7)
    plt.title(f"Projection des individus sur PC{pc_x+1} et PC{pc_y+1} avec K-means")
    plt.xlabel(f'PC{pc_x+1} (basé sur {", ".join(variables_pc_x)})')
    plt.ylabel(f'PC{pc_y+1} (basé sur {", ".join(variables_pc_y)})')
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.show()

def plot_pca_individuals(df, plot_correlation_circle, plot_scree_plot, color_col=None):
    """
    Effectue une ACP et affiche le nuage de points des individus selon F1 et F2.
    Agrège également les variables corrélées fortement avec F1 et F2 en nouvelles variables synthétiques.
    
    Paramètres :
    - df : DataFrame contenant les données.
    - plot_correlation_circle : Fonction pour afficher le cercle de corrélation.
    - plot_scree_plot : Fonction pour afficher le graphique du coude.
    - color_col : Nom de la colonne pour coloriser les points (optionnel).
    """
    # Remplacement des NaN par la médiane de chaque colonne
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='median', missing_values=np.nan)
    df_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)
    
    # Standardisation des données
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_imputed)
    
    # Application de l'ACP
    pca = PCA()
    pca_result = pca.fit_transform(df_scaled)
    
    # Calcul des composantes principales
    pcs = pd.DataFrame(pca.components_.T, index=numeric_cols, 
                        columns=[f'PC{i+1}' for i in range(len(pca.components_))])
    
    # Identifier les variables fortement corrélées avec F1 et F2
    threshold = 0.5  # Seuil de corrélation significative
    f1_vars = pcs.loc[abs(pcs['PC1']) > threshold, 'PC1'].index.tolist()
    f2_vars = pcs.loc[abs(pcs['PC2']) > threshold, 'PC2'].index.tolist()
    
    # Création des variables synthétiques
    df['F1_synth'] = df_imputed[f1_vars].mean(axis=1) if f1_vars else np.zeros(len(df))
    df['F2_synth'] = df_imputed[f2_vars].mean(axis=1) if f2_vars else np.zeros(len(df))
    
    # Affichage du cercle de corrélation
    plot_correlation_circle(pca, numeric_cols)
    
    # Affichage du graphique du coude
    plot_scree_plot(pca)
    
    # Nuage de points des individus
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                           c=df[color_col] if color_col and color_col in df.columns else 'blue', 
                           cmap='viridis', alpha=0.7)
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.title("Nuage de points des individus (F1 vs F2)")
    if color_col and color_col in df.columns:
        plt.colorbar(scatter, label=color_col)
    plt.grid()
    plt.show()
    
    return df[['F1_synth', 'F2_synth']]

def get_normal_columns(df, columns=None, alpha=0.05, max_sample_size=5000):
    """
    Identifie les colonnes normalement distribuées parmi un sous-ensemble de colonnes spécifiées.

    :param df: DataFrame pandas
    :param columns: Liste des colonnes à tester
    :param alpha: Seuil de p-value pour accepter la normalité
    :param max_sample_size: Taille maximale de l'échantillon pour éviter les avertissements sur la taille de l'échantillon
    :return: Liste des colonnes normalement distribuées
    """
    normal_columns = []

    # Si aucune liste de colonnes n'est fournie, prendre toutes les colonnes numériques
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()

    for col in columns:
        # Filtrer les valeurs numériques (en cas de non numériques dans la colonne)
        data = df[col].dropna()

        # Vérifier que la colonne contient des données numériques
        if data.empty or not pd.api.types.is_numeric_dtype(data):
            continue

        # Limiter la taille de l'échantillon pour éviter le warning
        sample = data.sample(n=min(len(data), max_sample_size), random_state=42)

        # Test de Shapiro-Wilk pour la normalité
        stat, p_value = shapiro(sample)
        
        # Si p > alpha, on considère que la colonne suit une distribution normale
        if p_value > alpha:
            normal_columns.append(col)

    return normal_columns

def analyze_variables(df, target_col, features, max_sample_size=5000):
    """
    Analyse plusieurs variables quantitatives par rapport à une variable catégorielle.
    
    :param df: DataFrame pandas
    :param target_col: Variable catégorielle (ex: 'nutrition_grade_fr')
    :param features: Liste des variables quantitatives à analyser
    :param max_sample_size: Taille max pour le test de Shapiro-Wilk
    """
    # Vérifier que les colonnes existent
    missing_cols = [col for col in [target_col] + features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Les colonnes suivantes sont absentes du DataFrame : {missing_cols}")

    # Boucle sur les variables sélectionnées
    for feature_col in features:
        print(f"\n🔹 Analyse pour {feature_col} :")

        # Suppression des NaN
        df_clean = df[[target_col, feature_col]].dropna()

        # Vérification si assez de données
        if df_clean.shape[0] < 10:
            print("⚠️ Trop peu de données pour analyser cette variable.")
            continue

        # Test de normalité (Shapiro-Wilk avec sous-échantillonnage)
        stat, p_shapiro = test_shapiro_safe(df_clean[feature_col], max_sample_size)
        normal = p_shapiro > 0.05
        print(f"  - Test de Shapiro-Wilk : p-value = {p_shapiro:.5f} {'✅ Normal' if normal else '❌ Non normal'}")

        # Test de Kolmogorov-Smirnov (optionnel)
        stat_ks, p_ks = kstest(df_clean[feature_col], 'norm')
        print(f"  - Test de Kolmogorov-Smirnov : p-value = {p_ks:.5f}")

        # Regroupement par catégorie de la variable cible
        groups = [group[feature_col].values for _, group in df_clean.groupby(target_col)]

        # Test d'homogénéité des variances (Levene)
        if len(groups) > 1:
            stat, p_levene = levene(*groups)
            homogene = p_levene > 0.05
            print(f"  - Test de Levene (homogénéité) : p-value = {p_levene:.5f} {'✅ Homogène' if homogene else '❌ Non homogène'}")
        else:
            print("⚠️ Impossible de tester Levene (une seule catégorie présente).")
            homogene = False

        # Sélection du test statistique
        if normal and homogene:
            stat, p = f_oneway(*groups)
            test_type = "ANOVA"
        else:
            stat, p = kruskal(*groups)
            test_type = "Kruskal-Wallis"

        # Résultat du test statistique
        print(f"  - {test_type} : p-value = {p:.5f} {'⚠️ Différence significative !' if p < 0.05 else '✅ Aucune différence significative.'}")

def plot_nutrition_scatter(df, x_col, y_col, color_col='nutrition-score-fr_100g', cmap='coolwarm'):
    """
    Crée un nuage de points avec une colorisation en fonction du score nutritionnel.
    
    :param df: DataFrame pandas contenant les données
    :param x_col: Nom de la colonne pour l'axe des X
    :param y_col: Nom de la colonne pour l'axe des Y
    :param color_col: Nom de la colonne pour la colorisation (par défaut, 'nutrition-score-fr_100g')
    :param cmap: Colormap utilisée pour la colorisation des points (par défaut, 'coolwarm')
    """
    
    # Vérification que les colonnes existent et sont numériques
    for col in [x_col, y_col, color_col]:
        if col not in df.columns:
            raise ValueError(f"La colonne '{col}' n'existe pas dans le DataFrame.")
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValueError(f"La colonne '{col}' doit être numérique.")
    
    # Suppression des valeurs NaN pour éviter les erreurs
    df_filtered = df[[x_col, y_col, color_col]].dropna()
    
    # Création du nuage de points avec seaborn
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df_filtered[x_col], df_filtered[y_col], 
        c=df_filtered[color_col], cmap=cmap, edgecolors='k', alpha=0.7
    )
    
    # Ajout de la barre de couleur
    cbar = plt.colorbar(scatter)
    cbar.set_label(color_col)
    
    # Ajout des titres et labels
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Nuage de points colorisé par {color_col}')
    
    # Affichage du graphique
    plt.show()

def remove_duplicates(df, id_column):
    """
    Supprime les doublons en se basant uniquement sur un identifiant unique.

    Parameters:
    df (pd.DataFrame): Le DataFrame à nettoyer.
    id_column (str): Nom de la colonne contenant l'identifiant unique.

    Returns:
    pd.DataFrame: Un DataFrame sans doublons.
    """
    if id_column not in df.columns:
        raise ValueError(f"La colonne '{id_column}' n'existe pas dans le DataFrame.")
    
    df_cleaned = df.drop_duplicates(subset=[id_column])

    return df_cleaned

def filter_duplicates(df, columns=None):
    """
    Filtre le DataFrame pour afficher uniquement les lignes dupliquées sur une ou plusieurs colonnes.

    Parameters:
    df (pd.DataFrame): Le DataFrame à analyser.
    columns (list or str, optional): Nom d'une colonne ou liste de colonnes à vérifier. 
                                      Si None, vérifie tous les doublons sur toutes les colonnes.

    Returns:
    pd.DataFrame: DataFrame contenant uniquement les lignes en double.
    """
    if columns:
        if isinstance(columns, str):
            columns = [columns]  # Convertir en liste si une seule colonne est donnée
        duplicates = df[df.duplicated(subset=columns, keep=False)]  # Garder toutes les occurrences en doublon
    else:
        duplicates = df[df.duplicated(keep=False)]  # Vérifier les doublons sur toutes les colonnes

    return duplicates

def calculate_mode(df, columns=None):
    """
    Calcule le mode d'un DataFrame ou des colonnes spécifiques renseignées.

    Parameters:
    df (pd.DataFrame): Le DataFrame à analyser.
    columns (list, optional): Liste des colonnes sur lesquelles calculer le mode. Si None, calcule le mode sur toutes les colonnes.

    Returns:
    pd.Series: Mode des colonnes spécifiées.
    """
    if columns:
        return df[columns].mode()
    else:
        return df.mode()

def calculate_mean(df, columns=None):
    """
    Calcule la moyenne d'un DataFrame ou des colonnes spécifiques renseignées.

    Parameters:
    df (pd.DataFrame): Le DataFrame à analyser.
    columns (list, optional): Liste des colonnes sur lesquelles calculer la moyenne. Si None, calcule la moyenne sur toutes les colonnes.

    Returns:
    pd.Series: Moyenne des colonnes spécifiées.
    """
    if columns:
        return df[columns].mean()
    else:
        return df.mean()

def calculate_median(df, columns=None):
    """
    Calcule la médiane d'un DataFrame ou des colonnes spécifiques renseignées.

    Parameters:
    df (pd.DataFrame): Le DataFrame à analyser.
    columns (list, optional): Liste des colonnes sur lesquelles calculer la médiane. Si None, calcule la médiane sur toutes les colonnes.

    Returns:
    pd.Series: Médiane des colonnes spécifiées.
    """
    if columns:
        return df[columns].median()
    else:
        return df.median()


def replace_nan_with_stat(df, columns, stat='mode'):
    """
    Remplace les NaN dans les colonnes spécifiées par une statistique choisie (mode, mean ou median).

    :param df: DataFrame pandas
    :param columns: Liste des colonnes à traiter
    :param stat: Statistique utilisée pour remplacer les NaN ('mode', 'mean', 'median')
    :return: DataFrame avec valeurs remplacées
    """
    df = df.copy()  # Éviter les modifications de l'original

    for col in columns:
        if stat == 'mode':
            value = df[col].mode().dropna().iloc[0] if not df[col].mode().dropna().empty else np.nan
        elif stat == 'mean':
            value = df[col].mean(skipna=True)
        elif stat == 'median':
            value = df[col].median(skipna=True)
        else:
            raise ValueError("Stat must be 'mode', 'mean', or 'median'")

        # Remplacer NaN avec la valeur calculée
        df[col] = df[col].fillna(value)

    return df


def display_boxplot_with_stats(dataframe, column_name):
    """
    Affiche un graphique box plot pour une colonne donnée d'un DataFrame
    et ajoute des lignes pour la moyenne, la médiane et l'écart-type.
    
    Parameters:
        dataframe (pd.DataFrame): Le DataFrame contenant les données.
        column_name (str): Le nom de la colonne à afficher dans le boxplot.
    """
    # Calcul des statistiques
    mean = dataframe[column_name].mean()
    median = dataframe[column_name].median()
    std = dataframe[column_name].std()

    # Box plot
    dataframe.boxplot(column=column_name)

    # Ajouter la ligne de la moyenne
    plt.axhline(y=mean, color='r', linestyle='--', label='Moyenne')

    # Ajouter la ligne de la médiane
    plt.axhline(y=median, color='b', linestyle='-', label='Médiane')

    # Ajouter les lignes de l'écart-type
    plt.axhline(y=mean + std, color='g', linestyle=':', label='Écart-type (+)')
    plt.axhline(y=mean - std, color='g', linestyle=':', label='Écart-type (-)')

    # Ajouter des annotations textuelles avec des positions ajustées
    # Moyenne (au-dessus)
    plt.annotate(f'Moyenne: {mean:.2f}', 
                 xy=(0.5, mean + 0.05), xycoords='data',  # Légèrement au-dessus de la ligne
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white'),
                 color='red', fontsize=10, ha='center')

    # Médiane (au-dessous)
    plt.annotate(f'Médiane: {median:.2f}', 
                 xy=(0.85, median), xycoords='data',  # Légèrement en dessous de la ligne
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='white'),
                 color='blue', fontsize=10, ha='center')

    # Écart-type (à droite)
    plt.annotate(f'Écart-type: {std:.2f}', 
                 xy=(1.15, mean + std), xycoords='data',  # À droite de la ligne
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='white'),
                 color='green', fontsize=10, ha='left')

    # Titre et labels
    plt.title(f'Box Plot {column_name}')
    plt.ylabel(column_name)

    # Ajouter une légende
    plt.legend()

    # Ajuster la mise en page
    plt.tight_layout()

    # Afficher le graphique
    plt.show()


def detect_outliers(df, columns, method="IQR", plausibility_check=True):
    """
    Détecte les valeurs aberrantes dans les colonnes spécifiées du DataFrame.
    
    Args:
    - df (pd.DataFrame): Le DataFrame contenant les données.
    - columns (list): Liste des colonnes à analyser.
    - method (str): Méthode à utiliser ("IQR" ou "Z-score").
    - plausibility_check (bool): Si True, applique des seuils de plausibilité définis.

    Returns:
    - pd.DataFrame: Un DataFrame indiquant les valeurs aberrantes (True/False).
    """
    outliers = pd.DataFrame(False, index=df.index, columns=columns)

    # Seuils de plausibilité
    plausibility_thresholds = {
        'omega-3-fat_100g': (0, 30),
        'omega-6-fat_100g': (0, 50),
        'iron_100g': (0, 15),
        'calcium_100g': (0, 1500),
        'energy-from-fat_100g': (0, 900)
    }

    for col in columns:
        if df[col].dtype in [np.float64, np.int64]:  
            col_data = df[col].dropna()

            # Détection par IQR
            if method == "IQR":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

            # Détection par Z-score (calcul manuel)
            elif method == "Z-score":
                mean = col_data.mean()
                std = col_data.std()
                z_scores = (df[col] - mean) / std
                outliers[col] = np.abs(z_scores) > 3

            # Vérification de plausibilité
            if plausibility_check and col in plausibility_thresholds:
                min_val, max_val = plausibility_thresholds[col]
                outliers[col] |= (df[col] < min_val) | (df[col] > max_val)

    return outliers

def replace_outliers(df, outliers_detected, columns, strategy="median", replace_with_nan=False):
    """
    Remplace les outliers détectés dans un DataFrame par une valeur spécifique ou NaN.
    
    Paramètres :
    - df : pd.DataFrame -> DataFrame contenant les données.
    - outliers_detected : dict -> Dictionnaire des outliers avec colonnes et indices.
    - columns : list -> Liste des colonnes à traiter.
    - strategy : str -> Stratégie de remplacement ('median', 'mean', 'mode').
    - replace_with_nan : bool -> Si True, remplace les outliers par NaN au lieu d'une statistique.
    
    Retourne :
    - pd.DataFrame -> DataFrame avec les valeurs aberrantes remplacées.
    """
    df_cleaned = df.copy()

    for col in columns:
        if col in df_cleaned.columns:
            if replace_with_nan:
                replacement_value = np.nan
            else:
                if strategy == "median":
                    replacement_value = df_cleaned[col].median()
                elif strategy == "mean":
                    replacement_value = df_cleaned[col].mean()
                elif strategy == "mode":
                    replacement_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else np.nan
                else:
                    raise ValueError("La stratégie doit être 'median', 'mean' ou 'mode'.")
            
            print(f"🔹 {col} - Valeur utilisée pour remplacer les outliers : {replacement_value}")
            
            # Remplacement des outliers détectés
            df_cleaned.loc[outliers_detected[col], col] = replacement_value

            # Vérification : Remplir les NaN restants avec la même stratégie si replace_with_nan est False
            if not replace_with_nan:
                df_cleaned[col] = df_cleaned[col].fillna(replacement_value)
    
    return df_cleaned

def plot_histogram(df, column):
    """
    Affiche un histogramme pour une variable donnée d'un DataFrame avec estimation KDE.
    Trace également les seuils des outliers selon la méthode interquartile (IQR).
    
    :param df: DataFrame pandas
    :param column: Nom de la colonne à visualiser
    """
    # Suppression des valeurs NaN
    data = df[column].dropna()
    
    # Calcul des quartiles et de l'IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # Seuils des outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Création du graphique
    plt.figure(figsize=(8, 5))
    sns.histplot(data, bins=30, kde=True, color="royalblue")
    
    # Ajout des lignes verticales pour les seuils des outliers
    plt.axvline(lower_bound, color='red', linestyle='dashed', linewidth=2, label=f"Seuil bas ({lower_bound:.2f})")
    plt.axvline(upper_bound, color='red', linestyle='dashed', linewidth=2, label=f"Seuil haut ({upper_bound:.2f})")
    
    # Personnalisation
    plt.title(f'Histogramme de {column} avec détection des outliers (IQR)')
    plt.xlabel(column)
    plt.ylabel('Fréquence')
    plt.legend()
    
    # Affichage du graphique
    plt.show()

def plot_scatter(df, column_x, column_y):
    """
    Affiche un diagramme de dispersion pour deux variables d'un DataFrame.
    :param df: DataFrame pandas
    :param column_x: Nom de la colonne pour l'axe X
    :param column_y: Nom de la colonne pour l'axe Y
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[column_x], y=df[column_y])
    plt.title(f'Diagramme de dispersion entre {column_x} et {column_y}')
    plt.xlabel(column_x)
    plt.ylabel(column_y)
    plt.show()

def plot_heatmap(df, columns=None, method='spearman', figsize=(8,6), cmap="coolwarm", annot=True):
    """
    Génère une heatmap de corrélation pour les colonnes sélectionnées d'un DataFrame.

    :param df: pd.DataFrame - Le DataFrame contenant les données.
    :param columns: list - Liste des colonnes à inclure dans la heatmap (par défaut, toutes les colonnes numériques).
    :param method: str - Méthode de corrélation ('spearman', 'pearson', 'kendall').
    :param figsize: tuple - Taille de la figure (par défaut : (8,6)).
    :param cmap: str - Palette de couleurs pour la heatmap.
    :param annot: bool - Afficher ou non les coefficients dans les cases.
    """
    
    # Sélection des colonnes à analyser
    if columns is None:
        df_selected = df.select_dtypes(include=['number'])  # Sélectionne seulement les colonnes numériques
    else:
        df_selected = df[columns]
    
    # Calcul de la matrice de corrélation
    corr_matrix = df_selected.corr(method=method)
    
    # Création de la heatmap avec Seaborn
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=".2f", linewidths=0.5, square=True, cbar=True)

    # Ajout d'un titre
    plt.title(f"Heatmap des corrélations ({method.capitalize()})")
    plt.show()

def test_spearman(df, col1, col2):
    """
    Effectue le test de Spearman entre deux colonnes et affiche le coefficient de corrélation et la p-value.
    :param df: DataFrame contenant les données
    :param col1: Nom de la première colonne
    :param col2: Nom de la deuxième colonne
    """
    coef, p_value = spearmanr(df[col1], df[col2], nan_policy='omit')

    print(f"Test de Spearman entre {col1} et {col2}:")
    print(f"📊 Coefficient de corrélation : {coef:.3f}")
    print(f"📉 P-value : {p_value:.5f}")
    
    if p_value < 0.05:
        print("✅ La corrélation est significative (p < 0.05).")
    else:
        print("❌ La corrélation n'est pas significative (p >= 0.05).")
    print("-" * 50)

def plot_elbow_curve(data_scaled):
    """
    Applique une ACP sur les données standardisées et trace la courbe du coude.

    Paramètres :
    - data_scaled : array numpy, données prétraitées et standardisées

    Retour :
    - pca : objet PCA ajusté
    - explained_variance : array, variance expliquée par chaque composante
    """
    # Appliquer l'ACP
    pca = PCA()
    pca.fit(data_scaled)

    # Calculer les valeurs propres (variance expliquée)
    explained_variance = pca.explained_variance_ratio_

    # Tracer la courbe du coude
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), 
             marker='o', linestyle='--', color='b')
    plt.xlabel("Nombre de composantes principales")
    plt.ylabel("Variance expliquée cumulée")
    plt.title("Méthode du coude pour choisir le nombre optimal de composantes")
    plt.grid()
    plt.show()

    return pca, explained_variance

def plot_elbow_curve2(data_scaled):
    """
    Applique une ACP sur les données standardisées et trace la courbe du coude.

    Paramètres :
    - data_scaled : array numpy, données prétraitées et standardisées

    Retour :
    - pca : objet PCA ajusté
    """
    # Appliquer l'ACP
    pca = PCA()
    pca.fit(data_scaled)

    # Calculer les valeurs propres (variance expliquée)
    explained_variance = pca.explained_variance_ratio_

    # Tracer la courbe du coude
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), 
             marker='o', linestyle='--', color='b')
    plt.xlabel("Nombre de composantes principales")
    plt.ylabel("Variance expliquée cumulée")
    plt.title("Méthode du coude pour choisir le nombre optimal de composantes")
    plt.grid()
    plt.show()

    # Retourner uniquement l'objet PCA
    return pca


def plot_correlation_circle(pca, columns_for_pca):
    """
    Tracer le cercle des corrélations (ACP) avec les composantes principales.

    Parameters:
    pca : modèle PCA
        Modèle PCA entraîné qui contient les informations des composantes principales.
    columns_for_pca : list
        Liste des noms des variables utilisées pour l'ACP.
    """
    # Calcul des coordonnées des variables dans l'espace des composantes principales
    components = pca.components_

    # Créer la figure et les axes
    fig, ax = plt.subplots(figsize=(7, 7))

    # Ajouter le cercle unité
    circle = patches.Circle((0, 0), radius=1, edgecolor='black', facecolor='none', linestyle='dashed')
    ax.add_patch(circle)

    # Tracer les flèches pour chaque variable
    for i, column in enumerate(columns_for_pca):
        plt.arrow(0, 0, components[0, i], components[1, i], 
                  head_width=0.05, head_length=0.05, color='r')
        plt.text(components[0, i], components[1, i], column, fontsize=12)

    # Limites et lignes de référence
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    # Titre et affichage
    plt.title("Cercle des corrélations (ACP)")
    plt.grid()
    plt.show()

def plot_pca_projection(pca, data_scaled, df_clean, column_for_color):
    """
    Trace la projection des individus dans l'espace des 2 premières composantes principales (PCA),
    avec colorisation basée sur une variable spécifique.

    Paramètres :
    - pca : modèle PCA ajusté
    - data_scaled : array numpy, données standardisées
    - df_clean : DataFrame contenant les données nettoyées
    - column_for_color : string, nom de la colonne à utiliser pour la colorisation
    """
    # Extraire les coordonnées des individus dans l'espace des composantes principales
    df_pca = pca.transform(data_scaled)

    # Ajouter la colonne pour la colorisation
    df_clean[column_for_color] = df_clean[column_for_color]

    # Tracer la projection
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df_clean[column_for_color], palette="Set1", alpha=0.7)
    plt.xlabel("Composante 1")
    plt.ylabel("Composante 2")
    plt.title("Projection des individus (colorisé par " + column_for_color + ")")
    plt.legend(title=column_for_color,  loc='upper right')
    plt.grid()
    plt.show()

def impute_missing_values(df, columns=None, estimator=BayesianRidge(), max_iter=10, random_state=42):
    """
    Remplace les valeurs NaN dans certaines colonnes ou dans tout le DataFrame avec IterativeImputer.
    
    :param df: DataFrame pandas contenant des valeurs NaN
    :param columns: Liste des colonnes à imputer (par défaut, toutes les colonnes numériques)
    :param estimator: Modèle utilisé pour l'imputation (par défaut, BayesianRidge)
    :param max_iter: Nombre maximal d'itérations pour l'imputation
    :param random_state: Graine aléatoire pour la reproductibilité
    :return: Nouveau DataFrame avec les valeurs NaN remplacées
    """
    df_imputed = df.copy()  # Création d'une copie du DataFrame original
    
    # Sélection des colonnes concernées
    if columns is None:
        columns = df_imputed.select_dtypes(include=[np.number]).columns  # Colonnes numériques uniquement
    
    # Vérification qu'il y a bien des colonnes sélectionnées
    if len(columns) == 0:
        raise ValueError("Aucune colonne valide pour l'imputation.")
    
    # Initialisation de l'IterativeImputer
    imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=random_state)
    
    # Imputation sur les colonnes sélectionnées
    df_imputed[columns] = imputer.fit_transform(df_imputed[columns])
    
    return df_imputed