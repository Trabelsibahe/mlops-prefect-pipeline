"""
pipeline_prefect.py
-------------------
Prefect pipeline for the Customer Churn ML project.

Usage:
  python pipeline_prefect.py --flow all
  python pipeline_prefect.py --flow train
  python pipeline_prefect.py --flow evaluate
  python pipeline_prefect.py --flow code
  python pipeline_prefect.py --flow install
"""

import argparse
import subprocess
import sys
import os

from prefect import task, flow

# ── Paths ──────────────────────────────────────────────────
DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"
TEST_PATH = "tests/"

# Target files for code quality / security checks
TARGET_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]

# Sample customer for prediction demo
SAMPLE_CUSTOMER = [850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]

# ── Git repo ───────────────────────────────────────────────
# Replace with your actual GitHub repo URL
REPO_URL = "https://github.com/Trabelsibahe/mlops-prefect-pipeline.git"
PROJECT_DIR = os.path.abspath(".")  # current working directory


# ══════════════════════════════════════════════════════════
# ░░  TASKS — GIT
# ══════════════════════════════════════════════════════════


@task(name="git_clone_or_pull", log_prints=True)
def task_git_clone_or_pull():
    """
    First step of the pipeline:
    - If the project folder already exists → git pull (get latest code)
    - If it doesn't exist yet          → git clone (fresh download)
    """
    git_dir = os.path.join(PROJECT_DIR, ".git")

    if os.path.exists(git_dir):
        # Repo already cloned — just pull latest changes
        print(f"[git] Repo found at '{PROJECT_DIR}' — pulling latest changes …")
        result = subprocess.run(
            ["git", "-C", PROJECT_DIR, "pull"], capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"git pull failed:\n{result.stderr}")
        print("[git] Pull complete.")
    else:
        # Fresh machine — clone the repo
        print(f"[git] Cloning repo from {REPO_URL} …")
        result = subprocess.run(
            ["git", "clone", REPO_URL, PROJECT_DIR], capture_output=True, text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed:\n{result.stderr}")
        print(f"[git] Clone complete into '{PROJECT_DIR}'.")


# ══════════════════════════════════════════════════════════
# ░░  TASKS — DATA / MODEL
# ══════════════════════════════════════════════════════════


@task(name="Installer les dépendances", log_prints=True)
def install_dependencies():
    """Install packages listed in requirements.txt."""
    print("[install] Installing dependencies from requirements.txt …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"pip install failed:\n{result.stderr}")
    print("[install] Done.")


@task(name="Préparation des données", log_prints=True)
def task_prepare_data():
    """Load, clean and split the dataset."""
    from model_pipeline import prepare_data

    x_train, x_test, y_train, y_test = prepare_data(DATA_PATH)
    return x_train, x_test, y_train, y_test


@task(name="Entrainement du modèle", log_prints=True)
def task_train_model(x_train, y_train):
    """Train the Random Forest classifier."""
    from model_pipeline import train_model

    model = train_model(x_train, y_train)
    return model


@task(name="Sauvegarder le modèle", log_prints=True)
def task_save_model(model):
    """Persist the trained model to disk."""
    from model_pipeline import save_model

    save_model(model, MODEL_PATH)


@task(name="Charger le modèle", log_prints=True)
def task_load_model():
    """Load a previously saved model from disk."""
    from model_pipeline import load_model

    model = load_model(MODEL_PATH)
    return model


@task(name=" Evaluer le modèle", log_prints=True)
def task_evaluate_model(model, x_test, y_test):
    """Evaluate the model and print metrics."""
    from model_pipeline import evaluate_model

    metrics = evaluate_model(model, x_test, y_test)
    return metrics


@task(name="Prédire", log_prints=True)
def task_predict(model):
    """Run inference on a sample customer."""
    from model_pipeline import predict

    result = predict(model, SAMPLE_CUSTOMER)
    return result

@task(name="start_api", log_prints=True)
def task_start_api():
    """Launch the FastAPI server with Uvicorn."""
    print("[api] Starting FastAPI server on http://0.0.0.0:8000 …")
    print("[api] Interactive docs will be available at http://127.0.0.1:8000/docs")
    import subprocess
    result = subprocess.run(
        ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
    )
    if result.returncode not in (0, 130):  # 130 = Ctrl+C (normal exit)
        raise RuntimeError("[api] Uvicorn exited with an error.")

# ══════════════════════════════════════════════════════════
# ░░  TASKS — CODE QUALITY
# ══════════════════════════════════════════════════════════


@task(name="Formatage du code", log_prints=True)
def task_format_code():
    """Auto-format code with Black."""
    print("[format] Running Black formatter …")
    result = subprocess.run(["black"] + TARGET_FILES, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[format] Warning:\n{result.stderr}")
    print("[format] Done.")


@task(name="Qualité du code", log_prints=True)
def task_lint_code():
    """Check code quality with Flake8."""
    print("[lint] Running Flake8 …")
    result = subprocess.run(
        ["flake8", "--max-line-length=120"] + TARGET_FILES,
        capture_output=True,
        text=True,
    )
    print(result.stdout if result.stdout else "No linting issues found.")
    if result.returncode != 0:
        print(f"[lint] Issues detected (non-blocking):\n{result.stderr}")
    print("[lint] Done.")


@task(name="Sécurité du code", log_prints=True)
def task_security_check():
    """Scan for security vulnerabilities with Bandit."""
    print("[security] Running Bandit …")
    result = subprocess.run(
        ["bandit", "-r"] + TARGET_FILES, capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode not in (0, 1):  # 1 = issues found (non-fatal)
        print(f"[security] Warning:\n{result.stderr}")
    print("[security] Done.")


@task(name="tests unitaires", log_prints=True)
def task_run_unit_tests():
    """Generate a sample unit test file (if missing) and run all tests."""
    os.makedirs(TEST_PATH, exist_ok=True)
    test_file = os.path.join(TEST_PATH, "test_model_pipeline.py")

    # Create a sample test file if it does not exist yet
    if not os.path.exists(test_file):
        print(f"[tests] Creating sample test file at {test_file} …")
        with open(test_file, "w") as f:
            f.write('''"""
test_model_pipeline.py
----------------------
Sample unit tests for model_pipeline.py functions.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_pipeline import train_model, evaluate_model, save_model, load_model, predict


@pytest.fixture
def dummy_data():
    """Generate small fake train/test arrays."""
    np.random.seed(42)
    x_train = np.random.rand(100, 9)
    y_train = np.random.randint(0, 2, 100)
    x_test  = np.random.rand(20, 9)
    y_test  = np.random.randint(0, 2, 20)
    return x_train, x_test, y_train, y_test


def test_train_model_returns_classifier(dummy_data):
    x_train, _, y_train, _ = dummy_data
    model = train_model(x_train, y_train, n_estimators=10)
    assert hasattr(model, "predict"), "Model must have a predict method"


def test_evaluate_model_returns_dict(dummy_data):
    x_train, x_test, y_train, y_test = dummy_data
    model = train_model(x_train, y_train, n_estimators=10)
    metrics = evaluate_model(model, x_test, y_test)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_save_and_load_model(dummy_data, tmp_path):
    x_train, _, y_train, _ = dummy_data
    model = train_model(x_train, y_train, n_estimators=10)
    path  = str(tmp_path / "test_model.joblib")
    save_model(model, path)
    loaded = load_model(path)
    assert hasattr(loaded, "predict")


def test_predict_output(dummy_data):
    x_train, _, y_train, _ = dummy_data
    model  = train_model(x_train, y_train, n_estimators=10)
    sample = [0.5] * 9
    result = predict(model, sample)
    assert result in (0, 1), "Prediction must be 0 or 1"
''')
        print(f"[tests] Test file created.")

    print("[tests] Running pytest …")
    result = subprocess.run(["pytest", TEST_PATH, "-v"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Tests failed:\n{result.stderr}")
    print("[tests] All tests passed.")


# ══════════════════════════════════════════════════════════
# ░░  FLOWS
# ══════════════════════════════════════════════════════════


@flow(name="install", log_prints=True)
def flow_install():
    """Flow: install project dependencies."""
    install_dependencies()


@flow(name="code", log_prints=True)
def flow_code():
    """Flow: format → lint → security → unit tests."""
    task_format_code()
    task_lint_code()
    task_security_check()
    task_run_unit_tests()


@flow(name="train", log_prints=True)
def flow_train():
    """Flow: prepare data + train model."""
    x_train, x_test, y_train, y_test = task_prepare_data()
    task_train_model(x_train, y_train)


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    """Flow: load saved model + evaluate."""
    model = task_load_model()
    x_train, x_test, y_train, y_test = task_prepare_data()
    task_evaluate_model(model, x_test, y_test)


@flow(name="all", log_prints=True)
def flow_all():
    """
    Full pipeline:
      git pull -> install -> code quality -> prepare -> train -> save -> evaluate -> predict
    """
    # 0. Fetch latest code from remote Git repo
    task_git_clone_or_pull()

    # 1. Dependencies
    install_dependencies()

    # 2. Code quality checks
    task_format_code()
    task_lint_code()
    task_security_check()
    task_run_unit_tests()

    # 3. Data / model
    x_train, x_test, y_train, y_test = task_prepare_data()
    model = task_train_model(x_train, y_train)
    task_save_model(model)
    task_evaluate_model(model, x_test, y_test)
    task_predict(model)

@flow(name="api", log_prints=True)
def flow_api():
    """Flow: start the FastAPI prediction service."""
    task_start_api()

# ══════════════════════════════════════════════════════════
# ░░  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════

FLOWS = {
    "all": flow_all,
    "train": flow_train,
    "evaluate": flow_evaluate,
    "code":     flow_code,
    "install":  flow_install,
    "api":      flow_api,
}

#   ARGUMENTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefect ML Pipeline")
    parser.add_argument(
        "--flow",
        choices=list(FLOWS.keys()),
        required=True,
        help="Which flow to run: all | train | evaluate | code | install | api"
    )
    args = parser.parse_args()
    FLOWS[args.flow]()
