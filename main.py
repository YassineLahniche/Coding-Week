import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import pickle
import plotly.graph_objects as go
import matplotlib.cm as cm
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, auc, confusion_matrix, 
    precision_recall_curve
)
from scipy.stats import randint, uniform
import xgboost as xgb
from catboost import CatBoostClassifier


# 1. Load the dataset
def load_data():
    """Load the UCI obesity dataset"""
    print("Loading UCI obesity dataset...")
    dataset = fetch_ucirepo(id=544)
    df = dataset.data.original.copy()
    return df


# 2. Preprocess the data
def preprocess_data(df):
    """
    Preprocess the dataset:
    - Remove missing values
    - Encode categorical variables
    - Remove outliers
    """
    print("Preprocessing data...")
    
    # Remove missing values
    df.dropna(inplace=True)
    
    # Encode categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {col: LabelEncoder().fit(df[col]) for col in categorical_columns}
    for col in categorical_columns:
        df[col] = label_encoders[col].transform(df[col])
    
    # Remove outliers
    df = remove_outliers(df, df.select_dtypes(include=['float64', 'int64']).columns)
    
    
    return df


def remove_outliers(data, columns, lower=0.01, upper=0.99):
    """Remove extreme values using percentiles."""
    print("Removing outliers...")
    for col in columns:
        q1, q99 = data[col].quantile([lower, upper])
        data = data[(data[col] >= q1) & (data[col] <= q99)]
    return data


# 3. Parameter distribution for XGBoost tuning
def get_parameter_distribution():
    """Define parameter distribution for XGBoost tuning"""
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


# 4. Fine-tune XGBoost with RandomizedSearchCV
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
    print("Best parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    return best_model, search


# 5. Model evaluation function
def evaluate_model(y_true, preds, model_name):
    """
    Evaluate model performance using multiple metrics.
    
    Args:
        y_true: True labels
        preds: Predicted labels
        model_name: Name of the model being evaluated
    """
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
    
    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# 6. Save models to disk
def save_models(models_dict):
    """
    Save trained models to disk.
    
    Args:
        models_dict: Dictionary mapping model names to model objects
    """
    print("Saving models to disk...")
    for name, model in models_dict.items():
        with open(f"{name}.pkl", "wb") as file:
            pickle.dump(model, file)
    print("Models saved successfully!")


# Main execution
def main():
    # Load dataset
    df = load_data()
    
    # Preprocess data
    df = preprocess_data(df)
    print("Data preprocessing complete.")
    
    # Split data into features and target
    X = df.drop("NObeyesdad", axis=1)
    y = df["NObeyesdad"]
    Y = pd.cut(y, bins=6, labels=[0, 1, 2, 3, 4, 5])  # Convert to 6 classes
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")
    
    # Train models
    print("\n" + "="*50)
    print("Training models...")
    print("="*50)
    
    # Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # CatBoost Classifier
    print("\nTraining CatBoost Classifier...")
    catboost_model = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_state=42, verbose=False)
    catboost_model.fit(X_train, y_train)

    # Baseline XGBoost Classifier
    print("\nTraining Baseline XGBoost Classifier...")
    xgb_baseline = xgb.XGBClassifier(random_state=42)
    xgb_baseline.fit(X_train, y_train)
    
    # Fine-tuned XGBoost Classifier
    print("\nTraining and tuning XGBoost Classifier...")
    xgb_tuned, _ = tune_xgboost(X_train, y_train)
    
    # Make predictions
    print("\n" + "="*50)
    print("Evaluating models...")
    print("="*50)
    
    y1_pred = rf_model.predict(X_test)
    y2_pred = catboost_model.predict(X_test)
    y3_pred = xgb_baseline.predict(X_test)
    y4_pred = xgb_tuned.predict(X_test)
    
    # Evaluate models
    results = []
    results.append(evaluate_model(y_test, y1_pred, "Random Forest"))
    results.append(evaluate_model(y_test, y2_pred, "CatBoost"))
    results.append(evaluate_model(y_test, y3_pred, "XGBoost Baseline"))
    results.append(evaluate_model(y_test, y4_pred, "XGBoost Tuned"))
    
    # Save models
    models = {
        "rf_model": rf_model,
        "catboost_model": catboost_model,
        "xgb_baseline": xgb_baseline,
        "xgb_tuned": xgb_tuned
    }
    save_models(models)
    
    # Return results for potential further analysis
    return results


if __name__ == "__main__":
    main()
