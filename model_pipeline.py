"""
model_pipeline.py
-----------------
Modularised ML pipeline for Customer Churn prediction.
Extracted from customer_churn.ipynb (Random Forest classifier).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


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
    # --- Load ---
    df = pd.read_csv(filepath)

    # --- Basic checks ---
    print(f"[prepare_data] Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    print(f"[prepare_data] Missing values:\n{df.isna().sum()[df.isna().sum() > 0]}")
    duplicates = df.duplicated().sum()
    if duplicates:
        print(
            f"[prepare_data] WARNING: {duplicates} duplicate rows found – dropping them."
        )
        df.drop_duplicates(inplace=True)

    # --- Encode categorical column ---
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # --- Drop irrelevant columns ---
    df.drop(columns=["Surname", "Geography"], axis=1, inplace=True)

    # --- Split features / target ---
    x = df.drop(["Exited"], axis=1)
    y = df["Exited"]

    # --- Drop ID-like columns ---
    x.drop(columns=["RowNumber", "CustomerId"], inplace=True)

    # --- Train / test split ---
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    print(
        f"[prepare_data] Train size: {x_train.shape[0]}  |  Test size: {x_test.shape[0]}"
    )
    return x_train, x_test, y_train, y_test


# ─────────────────────────────────────────────
# 2. train_model
# ─────────────────────────────────────────────
def train_model(x_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Train a Random Forest classifier.

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
    return model


# ─────────────────────────────────────────────
# 3. evaluate_model
# ─────────────────────────────────────────────
def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model and print key metrics.

    Parameters
    ----------
    model  : fitted classifier
    x_test : test features
    y_test : true labels

    Returns
    -------
    metrics : dict with accuracy, confusion_matrix, classification_report
    """
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"[evaluate_model] Accuracy : {acc * 100:.2f}%")
    print(f"[evaluate_model] Confusion Matrix:\n{matrix}")
    print(f"[evaluate_model] Classification Report:\n{report}")

    return {"accuracy": acc, "confusion_matrix": matrix, "report": report}


# ─────────────────────────────────────────────
# 4. save_model
# ─────────────────────────────────────────────
def save_model(model, filepath: str = "classifier.joblib"):
    """
    Persist the trained model to disk using joblib.

    Parameters
    ----------
    model    : fitted classifier
    filepath : destination path (default 'classifier.joblib')
    """
    joblib.dump(model, filepath)
    print(f"[save_model] Model saved to '{filepath}'.")


# ─────────────────────────────────────────────
# 5. load_model
# ─────────────────────────────────────────────
def load_model(filepath: str = "classifier.joblib"):
    """
    Load a previously saved model from disk.

    Parameters
    ----------
    filepath : path to the .joblib file (default 'classifier.joblib')

    Returns
    -------
    model : loaded classifier
    """
    model = joblib.load(filepath)
    print(f"[load_model] Model loaded from '{filepath}'.")
    return model


# ─────────────────────────────────────────────
# 6. predict  (bonus helper)
# ─────────────────────────────────────────────
def predict(model, sample: list):
    """
    Run inference on a single customer record.

    Parameters
    ----------
    model  : fitted classifier
    sample : list of 9 feature values in order:
             [CreditScore, Gender(0/1), Age, Tenure, Balance,
              NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]

    Returns
    -------
    prediction : 0 (not churned) or 1 (churned)
    """
    arr = np.array(sample).reshape(1, -1)
    pred = model.predict(arr)[0]
    label = "Churned" if pred == 1 else "Not Churned"
    print(f"[predict] Result: {label} ({pred})")
    return pred
