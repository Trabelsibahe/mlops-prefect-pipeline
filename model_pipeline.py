"""
model_pipeline.py
-----------------
Modularised ML pipeline for Customer Churn prediction.
MLflow tracking integrated: logs params, metrics, and model artefact
directly into whatever run the caller has opened.
"""

import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ── MLflow configuration (shared with callers) ─────────────
MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI",    "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Customer-Churn")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# ─────────────────────────────────────────────
# 1. prepare_data
# ─────────────────────────────────────────────
def prepare_data(filepath: str, test_size: float = 0.2, random_state: int = 1):
    """
    Load the CSV, clean and encode it, then split into train/test sets.

    Parameters
    ----------
    filepath     : path to Churn_Modelling.csv
    test_size    : fraction of data kept for testing (default 0.2)
    random_state : reproducibility seed (default 1)

    Returns
    -------
    x_train, x_test, y_train, y_test : numpy arrays
    """
    df = pd.read_csv(filepath)

    print(f"[prepare_data] Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"[prepare_data] Missing values:\n{df.isna().sum()[df.isna().sum() > 0]}")
    duplicates = df.duplicated().sum()
    if duplicates:
        print(f"[prepare_data] WARNING: {duplicates} duplicate rows found – dropping them.")
        df.drop_duplicates(inplace=True)

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])
    df.drop(columns=["Surname", "Geography"], axis=1, inplace=True)

    x = df.drop(["Exited"], axis=1)
    y = df["Exited"]
    x.drop(columns=["RowNumber", "CustomerId"], inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    print(f"[prepare_data] Train size: {x_train.shape[0]}  |  Test size: {x_test.shape[0]}")

    # Log data params into the active MLflow run (if one is open)
    if mlflow.active_run():
        mlflow.log_params({
            "data_filepath":  filepath,
            "test_size":      test_size,
            "random_state_split": random_state,
            "train_samples":  x_train.shape[0],
            "test_samples":   x_test.shape[0],
            "n_features":     x_train.shape[1],
        })

    return x_train, x_test, y_train, y_test


# ─────────────────────────────────────────────
# 2. train_model
# ─────────────────────────────────────────────
def train_model(x_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Train a Random Forest classifier and log params + model to the
    active MLflow run.

    Parameters
    ----------
    x_train       : training features
    y_train       : training labels
    n_estimators  : number of trees (default 100)
    random_state  : reproducibility seed (default 42)

    Returns
    -------
    model : fitted RandomForestClassifier
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    print(f"[train_model] Model trained with {n_estimators} estimators.")

    if mlflow.active_run():
        train_pred = model.predict(x_train)
        train_acc = accuracy_score(y_train, train_pred)
        train_precision = precision_score(y_train, train_pred, zero_division=0)
        train_recall = recall_score(y_train, train_pred, zero_division=0)
        train_f1 = f1_score(y_train, train_pred, zero_division=0)

        mlflow.log_params({
            "n_estimators":        n_estimators,
            "random_state_model":  random_state,
            "model_type":          "RandomForestClassifier",
        })
        mlflow.log_metrics({
            "train_accuracy":  train_acc,
            "train_precision":  train_precision,
            "train_recall":     train_recall,
            "train_f1":         train_f1,
        })
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            registered_model_name="CustomerChurnModel",
        )
        print("[mlflow] Params + training metrics + model artefact logged.")

    return model


# ─────────────────────────────────────────────
# 3. evaluate_model
# ─────────────────────────────────────────────
def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model, print key metrics, and log them to the
    active MLflow run.

    Returns
    -------
    metrics : dict with accuracy, precision, recall, f1,
              confusion_matrix, classification_report
    """
    y_pred = model.predict(x_test)

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    matrix    = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred)

    print(f"[evaluate_model] Accuracy : {acc * 100:.2f}%")
    print(f"[evaluate_model] Confusion Matrix:\n{matrix}")
    print(f"[evaluate_model] Classification Report:\n{report}")

    if mlflow.active_run():
        mlflow.log_metrics({
            "accuracy":  acc,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
        })
        print("[mlflow] Metrics logged.")

    return {
        "accuracy":             acc,
        "precision":            precision,
        "recall":               recall,
        "f1":                   f1,
        "confusion_matrix":     matrix,
        "report":               report,
    }


# ─────────────────────────────────────────────
# 4. save_model
# ─────────────────────────────────────────────
def save_model(model, filepath: str = "classifier.joblib"):
    """
    Persist the trained model to disk and log the file as an MLflow artefact.
    """
    joblib.dump(model, filepath)
    print(f"[save_model] Model saved to '{filepath}'.")

    if mlflow.active_run():
        mlflow.log_artifact(filepath, artifact_path="saved_model")
        print(f"[mlflow] Artefact '{filepath}' logged.")


# ─────────────────────────────────────────────
# 5. load_model
# ─────────────────────────────────────────────
def load_model(filepath: str = "classifier.joblib"):
    """Load a previously saved model from disk."""
    model = joblib.load(filepath)
    print(f"[load_model] Model loaded from '{filepath}'.")
    return model


# ─────────────────────────────────────────────
# 6. predict
# ─────────────────────────────────────────────
def predict(model, sample: list):
    """
    Run inference on a single customer record.

    Parameters
    ----------
    sample : list of 9 feature values in order:
             [CreditScore, Gender(0/1), Age, Tenure, Balance,
              NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]

    Returns
    -------
    prediction : 0 (not churned) or 1 (churned)
    """
    arr  = np.array(sample).reshape(1, -1)
    pred = model.predict(arr)[0]
    label = "Churned" if pred == 1 else "Not Churned"
    print(f"[predict] Result: {label} ({pred})")
    return pred