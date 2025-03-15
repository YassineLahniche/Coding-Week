import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
import time
from scipy.stats import randint, uniform
import pickle


# Load the dataset
dataset = fetch_ucirepo(id=544)
df = dataset.data.original.copy()

# Drop rows with missing values
df.dropna(inplace=True)

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Initialize LabelEncoder for each categorical column
label_encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_columns}

# Apply LabelEncoder to each categorical column
for col in categorical_columns:
    df[col] = label_encoders[col].transform(df[col])

# Create a dictionary to store the mapping of original values to encoded values
value_to_code_mapping = {}

# Iterate over each categorical column and its corresponding LabelEncoder
for col, le in label_encoders.items():
    # Create a dictionary for the current column
    col_mapping = {original_value: encoded_value for original_value, encoded_value in
                   zip(le.classes_, le.transform(le.classes_))}

    # Add the column's mapping to the main dictionary
    value_to_code_mapping[col] = col_mapping

# Function to remove outliers
def remove_outliers(data, columns, lower=0.01, upper=0.99):
    """Remove extreme values using percentiles."""
    for col in columns:
        q1, q99 = data[col].quantile([lower, upper])
        data = data[(data[col] >= q1) & (data[col] <= q99)]
    return data

# Remove outliers from numerical columns
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
df = remove_outliers(df, numerical_columns)

# Compute the correlation matrix
corr_matrix = df.corr().abs()

# Identify highly correlated features (above the threshold)
threshold = 0.85
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find columns to drop (keeping only one from each correlated pair)
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

# Drop the selected columns
df.drop(columns=to_drop, inplace=True)

#Optimize the dataset
float_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
int_cols = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"]
def optimize_memory(df, float_cols, int_cols):
  df[float_cols] = df[float_cols].astype("float32")
  df[int_cols] = df[int_cols].astype("int32")
  return df
df_opt = optimize_memory(df.copy(), float_cols, int_cols)

#Compare the old and new memory usage
def measure_memory_usage(dataset):
    return sys.getsizeof(dataset)

def demonstrate_memory_improvement(df, df_32):
    memory_64 = measure_memory_usage(df)
    memory_32 = measure_memory_usage(df_32)

    improvement = memory_64 - memory_32
    improvement_percentage = (improvement / memory_64) * 100

    print(f"Memory usage of float64 dataset: {memory_64} bytes")
    print(f"Memory usage of mixed (float32 + int32) dataset: {memory_32} bytes")
    print(f"Memory improvement: {improvement} bytes ({improvement_percentage:.2f}%)")

demonstrate_memory_improvement(df, df_opt)

#Separate features and results
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

#Undersampling data to remove biased distributions
undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X, y = undersampler.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest Classifier
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# CatBoost Classifier
print("Training CatBoost Classifier...")
catboost_model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_state=42, verbose=False)
catboost_model.fit(X_train, y_train)

# Baseline XGBoost Classifier
print("Training Baseline XGBoost Classifier...")
xgb_baseline = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=6,  # Specify the number of classes
    random_state=42
)
xgb_baseline.fit(X_train, y_train)

#Tuning the xgboost classifier parameters
def get_parameter_distribution():
    param_dist = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5),
        'min_child_weight': randint(1, 10),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1),
        'scale_pos_weight': uniform(0.5, 2.5),
    }
    return param_dist


# Fine-tune XGBoost with RandomizedSearchCV
def tune_xgboost(X_train, y_train, scoring='f1_weighted', n_iter=50, cv=5, n_jobs=-1, verbose=2):
    """
    Fine-tune XGBoost classifier using RandomizedSearchCV.

    Args:
        X_train: Training features
        y_train: Training target
        scoring: Scoring metric for optimization (default: f1_weighted)
        n_iter: Number of parameter settings sampled (default: 50)
        cv: Number of cross-validation folds (default: 5)
        n_jobs: Number of jobs to run in parallel (default: -1, all CPUs)
        verbose: Controls verbosity (default: 2)

    Returns:
        best_model: Tuned XGBoost model
        search: RandomizedSearchCV object with results
    """
    print(f"Starting XGBoost tuning with RandomizedSearchCV for {scoring}...")
    start_time = time.time()

    # Define the base model
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic' if len(np.unique(y_train)) == 2 else 'multi:softprob',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Define the parameter space
    param_dist = get_parameter_distribution()

    # Set up the cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Set up RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_strategy,
        verbose=verbose,
        random_state=42,
        n_jobs=n_jobs,
        return_train_score=True
    )

    # Fit RandomizedSearchCV
    search.fit(X_train, y_train)

    # Get the best model
    best_model = search.best_estimator_

    # Calculate time taken
    time_taken = time.time() - start_time
    print(f"XGBoost tuning completed in {time_taken:.2f} seconds")
    print(f"Best {scoring} score: {search.best_score_:.4f}")
    print("Best parameters for xgb:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    return best_model, search


xgb_tuned = tune_xgboost(X_train, y_train)[0]

# Make predictions
y1_pred = rf_model.predict(X_test)
y2_pred = catboost_model.predict(X_test)
y3_pred = xgb_baseline.predict(X_test)
y4_pred = xgb_tuned.predict(X_test)

# Function to evaluate models
def evaluate_model(y_true, preds, model_name):
    accuracy = accuracy_score(y_true, preds)
    precision = precision_score(y_true, preds, average="weighted")
    recall = recall_score(y_true, preds, average="weighted")
    f1 = f1_score(y_true, preds, average="weighted")

    print(f"Metrics for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("-" * 30)

# Evaluate each model
evaluate_model(y_test, y1_pred, "Random Forest")
evaluate_model(y_test, y2_pred, "CatBoost")
evaluate_model(y_test, y3_pred, "XGBoost Baseline")
evaluate_model(y_test, y4_pred, "XGBoost Tuned")

print(classification_report(y_test, y1_pred))
print(classification_report(y_test, y2_pred))
print(classification_report(y_test, y3_pred))
print(classification_report(y_test, y4_pred))

# Save the trained models
with open("xgb_baseline.pkl", "wb") as file:
    pickle.dump(xgb_baseline, file)
with open("xgb_tuned.pkl", "wb") as file:
    pickle.dump(xgb_tuned, file)
with open("rf_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)
with open("catboost_model.pkl", "wb") as file:
    pickle.dump(catboost_model, file)