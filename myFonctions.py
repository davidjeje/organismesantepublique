import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

def check_unique_columns(dataframe, columns=None):
    """
    Vérifie si une ou plusieurs colonnes d'un DataFrame contiennent des valeurs uniques.

    Parameters:
        dataframe (pd.DataFrame): Le DataFrame à analyser.
        columns (list or str, optional): Liste des colonnes à vérifier ou une seule colonne.
                                         Si None, toutes les colonnes du DataFrame sont analysées.

    Returns:
        dict: Un dictionnaire indiquant pour chaque colonne si elle contient des valeurs uniques.
              Clé : nom de la colonne, Valeur : True (unique) ou False (non unique).
    """
    # Si aucune colonne n'est spécifiée, vérifier toutes les colonnes
    if columns is None:
        columns = dataframe.columns
    elif isinstance(columns, str):
        columns = [columns]
    
    # Vérifier les colonnes spécifiées
    uniqueness = {}
    for column in columns:
        if column in dataframe.columns:
            uniqueness[column] = dataframe[column].is_unique
        else:
            raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")
    
    return uniqueness

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

def find_unique_identifier(df):
    """
    Trouve la colonne qui pourrait être un identifiant unique.

    Parameters:
    df (pd.DataFrame): Le DataFrame à analyser.

    Returns:
    list: Liste des colonnes qui sont des identifiants uniques.
    """
    unique_cols = [col for col in df.columns if df[col].nunique() == len(df) and df[col].isnull().sum() == 0]
    return unique_cols

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

def filter_duplicates_index(df):
    """
    Filtre le DataFrame pour afficher uniquement les lignes avec des index dupliqués.
    
    Parameters:
    df (pd.DataFrame): Le DataFrame à analyser.
    
    Returns:
    pd.DataFrame: DataFrame contenant uniquement les lignes avec des index dupliqués.
    """
    duplicates = df[df.index.duplicated(keep=False)]  # Garder toutes les occurrences d'un index dupliqué
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

def display_boxplot_with_stats3(dataframe, column_name):
    """
    Affiche un graphique box plot pour une colonne donnée d'un DataFrame
    et ajoute des lignes pour la moyenne, la médiane et l'écart-type.
    Optimisé pour de grands ensembles de données.
    
    Parameters:
        dataframe (pd.DataFrame): Le DataFrame contenant les données.
        column_name (str): Le nom de la colonne à afficher dans le boxplot.
    """
    # Calcul des statistiques
    data = dataframe[column_name].dropna()
    mean = data.mean()
    median = data.median()
    std = data.std()

    # Création de la figure et des axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Box plot avec gestion des outliers
    bp = ax.boxplot(data, showfliers=False, whis=1.5)

    # Ajouter la ligne de la moyenne
    ax.axhline(y=mean, color='r', linestyle='--', label='Moyenne')

    # Ajouter la ligne de la médiane (déjà présente dans le boxplot, mais on la rend plus visible)
    ax.axhline(y=median, color='b', linestyle='-', label='Médiane')

    # Ajouter les lignes de l'écart-type
    ax.axhline(y=mean + std, color='g', linestyle=':', label='Écart-type (+)')
    ax.axhline(y=mean - std, color='g', linestyle=':', label='Écart-type (-)')

    # Ajouter des annotations textuelles avec des positions ajustées
    ax.annotate(f'Moyenne: {mean:.2f}', 
                xy=(1.1, mean), xycoords=('axes fraction', 'data'),
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white'),
                color='red', fontsize=10, ha='left', va='center')

    ax.annotate(f'Médiane: {median:.2f}', 
                xy=(1.1, median), xycoords=('axes fraction', 'data'),
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='blue', facecolor='white'),
                color='blue', fontsize=10, ha='left', va='center')

    ax.annotate(f'Écart-type: {std:.2f}', 
                xy=(1.1, mean + std), xycoords=('axes fraction', 'data'),
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='green', facecolor='white'),
                color='green', fontsize=10, ha='left', va='center')

    # Titre et labels
    ax.set_title(f'Box Plot {column_name}')
    ax.set_ylabel(column_name)
    ax.set_xticks([])  # Supprime les ticks de l'axe x

    # Ajouter une légende
    ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1))

    # Ajuster la mise en page
    plt.tight_layout()

    # Afficher le graphique
    plt.show()


def detect_outliers(df, columns, method="IQR", plausibility_check=True):
    """
    Détecte les valeurs aberrantes dans les colonnes spécifiées du DataFrame sans utiliser scipy.
    
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

def replace_outliers2(df, outliers_detected, columns, strategy="median"):
    """
    Remplace les outliers détectés dans un DataFrame par une valeur spécifique.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        outliers_detected (pd.DataFrame): DataFrame booléen indiquant la présence d'outliers.
        columns (list): Liste des colonnes à traiter.
        strategy (str): Stratégie de remplacement ("median", "mean", "mode").
    
    Returns:
        pd.DataFrame: DataFrame avec les valeurs aberrantes remplacées.
    """
    df_cleaned = df.copy()

    for col in columns:
        if col in df_cleaned.columns:
            if strategy == "median":
                replacement_value = df_cleaned[col].median()
            elif strategy == "mean":
                replacement_value = df_cleaned[col].mean()
            elif strategy == "mode":
                replacement_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else np.nan
            else:
                raise ValueError("La stratégie doit être 'median', 'mean' ou 'mode'.")


            # Remplacement des outliers détectés
            df_cleaned.loc[outliers_detected[col], col] = replacement_value

            # Vérification : Remplir les NaN restants avec la même stratégie
            df_cleaned[col] = df_cleaned[col].fillna(replacement_value)

    return df_cleaned

def display_boxplot_with_stats2(dataframe, column_name):
    """
    Affiche un graphique box plot pour une colonne donnée d'un DataFrame
    et ajoute des lignes pour la moyenne, la médiane, l'écart-type, ainsi que les moustaches du boxplot.
    
    Parameters:
        dataframe (pd.DataFrame): Le DataFrame contenant les données.
        column_name (str): Le nom de la colonne à afficher dans le boxplot.
    """
    # Calcul des statistiques
    mean = dataframe[column_name].mean()
    median = dataframe[column_name].median()
    std = dataframe[column_name].std()

    # Box plot avec personnalisation
    ax = dataframe.boxplot(column=column_name, patch_artist=True, 
                           boxprops=dict(facecolor='lightblue', color='blue'),
                           whiskerprops=dict(color='green', linewidth=2),
                           capprops=dict(color='red', linewidth=2),
                           flierprops=dict(marker='o', color='purple', markersize=6))

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

    # Accéder aux moustaches du boxplot (whiskers)
    whiskers = ax.artists[0].get_paths()  # Obtenir les moustaches du boxplot
    for i, whisker in enumerate(whiskers):
        whisker_y = whisker.vertices[:, 1]  # Extraire les coordonnées Y des moustaches
        whisker_position = whisker_y[0]  # Position du whisker (valeur à l'extrémité)

        # Ajouter des annotations pour les moustaches
        plt.annotate(f'Moustache {i+1}: {whisker_position:.2f}', 
                     xy=(1.05, whisker_position), xycoords='data',  # Placer à droite des moustaches
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

def plot_histogram(df, column):
    """
    Affiche un histogramme pour une variable donnée d'un DataFrame.
    :param df: DataFrame pandas
    :param column: Nom de la colonne à visualiser
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column].dropna(), bins=30, kde=True)
    plt.title(f'Histogramme de {column}')
    plt.xlabel(column)
    plt.ylabel('Fréquence')
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
