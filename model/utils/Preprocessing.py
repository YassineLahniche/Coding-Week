import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo  

# 1. Chargement du dataset
dataset = fetch_ucirepo(id=544)
df = dataset.data.original.copy()

# 2. Vérification des valeurs manquantes et suppression
df.dropna(inplace=True)

# 3. Encodage des variables catégoriques
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_columns}
for col in categorical_columns:
    df[col] = label_encoders[col].transform(df[col])

# 4. Suppression des valeurs aberrantes (outliers)
def remove_outliers(data, columns, lower=0.01, upper=0.99):
    """Supprime les valeurs extrêmes en utilisant les percentiles."""
    for col in columns:
        q1, q99 = data[col].quantile([lower, upper])
        data = data[(data[col] >= q1) & (data[col] <= q99)]
    return data

df = remove_outliers(df, df.select_dtypes(include=['float64', 'int64']).columns)

# 5. Suppression des colonnes fortement corrélées (> 0.85)
corr_matrix = df.corr()
threshold = 0.85
to_drop = [col for col in corr_matrix.columns if any(corr_matrix[col] > threshold) and col != "NObeyesdad"]
df.drop(columns=to_drop, inplace=True)

# 6. Normalisation des variables numériques
scaler = MinMaxScaler()
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

