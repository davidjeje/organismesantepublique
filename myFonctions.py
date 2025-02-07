import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    df = df.copy()  # Éviter les problèmes de modification sur une copie

    if stat == 'mode':
        stats = df[columns].mode().iloc[0]  # Prendre la première valeur du mode
    elif stat == 'mean':
        stats = df[columns].mean()
    elif stat == 'median':
        stats = df[columns].median()
    else:
        raise ValueError("Stat must be 'mode', 'mean', or 'median'")

    # Remplacer NaN correctement en utilisant une assignation explicite
    df[columns] = df[columns].fillna(stats)

    return df  # Retourne le DataFrame modifié

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

def replace_outliers(df, outliers_detected, columns, strategy="median"):
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

            print(f"🔹 {col} - Valeur utilisée pour remplacer les outliers : {replacement_value}")

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
