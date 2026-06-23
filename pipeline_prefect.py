"""
pipeline_prefect.py
-------------------
Prefect pipeline for the Customer Churn ML project.
MLflow tracking: each flow opens ONE parent run; model_pipeline.py
functions log directly into it (no nested runs, no empty parents).

Usage:
  python pipeline_prefect.py --flow all
  python pipeline_prefect.py --flow train
  python pipeline_prefect.py --flow evaluate
  python pipeline_prefect.py --flow code
  python pipeline_prefect.py --flow install
  python pipeline_prefect.py --flow api
  python pipeline_prefect.py --flow mlflow_ui
"""

import argparse
import subprocess
import sys
import os

import mlflow
from prefect import task, flow

from model_pipeline import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

# ── Paths ──────────────────────────────────────────────────
DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"
TEST_PATH = "tests/"

TARGET_FILES    = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
SAMPLE_CUSTOMER = [850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]

# ── Git repo ───────────────────────────────────────────────
REPO_URL = "https://github.com/Trabelsibahe/mlops-prefect-pipeline.git"
PROJECT_DIR = os.path.abspath(".")
DOCKER_IMAGE_NAME = os.getenv("DOCKER_IMAGE_NAME", "prenom_nom_classe_mlops")
DOCKER_IMAGE_TAG = os.getenv("DOCKER_IMAGE_TAG", "latest")
DOCKER_CONTAINER_NAME = os.getenv("DOCKER_CONTAINER_NAME", "fastapi-mlflow-app")
DOCKER_LOCAL_PORT = os.getenv("DOCKER_LOCAL_PORT", "8000")

# ── MLflow ────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Customer-Churn")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# ══════════════════════════════════════════════════════════
# ░░  TASKS — GIT
# ══════════════════════════════════════════════════════════

@task(name="git_clone_or_pull", log_prints=True)
def task_git_clone_or_pull():
    git_dir = os.path.join(PROJECT_DIR, ".git")
    if os.path.exists(git_dir):
        print(f"[git] Repo found at '{PROJECT_DIR}' — pulling latest changes …")
        result = subprocess.run(["git", "-C", PROJECT_DIR, "pull"], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"git pull failed:\n{result.stderr}")
        print("[git] Pull complete.")
    else:
        print(f"[git] Cloning repo from {REPO_URL} …")
        result = subprocess.run(["git", "clone", REPO_URL, PROJECT_DIR], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"git clone failed:\n{result.stderr}")
        print(f"[git] Clone complete into '{PROJECT_DIR}'.")


# ══════════════════════════════════════════════════════════
# ░░  TASKS — DEPENDENCIES
# ══════════════════════════════════════════════════════════

@task(name="Installer les dépendances", log_prints=True)
def install_dependencies():
    print("[install] Installing dependencies from requirements.txt …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"pip install failed:\n{result.stderr}")
    print("[install] Done.")


# ══════════════════════════════════════════════════════════
# ░░  TASKS — DATA / MODEL
# ══════════════════════════════════════════════════════════

@task(name="Préparation des données", log_prints=True)
def task_prepare_data():
    from model_pipeline import prepare_data
    return prepare_data(DATA_PATH)


@task(name="Entrainement du modèle", log_prints=True)
def task_train_model(x_train, y_train):
    from model_pipeline import train_model
    return train_model(x_train, y_train)


@task(name="Sauvegarder le modèle", log_prints=True)
def task_save_model(model):
    from model_pipeline import save_model
    save_model(model, MODEL_PATH)


@task(name="Charger le modèle", log_prints=True)
def task_load_model():
    from model_pipeline import load_model
    return load_model(MODEL_PATH)


@task(name="Evaluer le modèle", log_prints=True)
def task_evaluate_model(model, x_test, y_test):
    from model_pipeline import evaluate_model
    return evaluate_model(model, x_test, y_test)


@task(name="Prédire", log_prints=True)
def task_predict(model):
    from model_pipeline import predict
    return predict(model, SAMPLE_CUSTOMER)


@task(name="start_api", log_prints=True)
def task_start_api():
    print("[api] Starting FastAPI server on http://0.0.0.0:8000 …")
    result = subprocess.run(
        ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
    )
    if result.returncode not in (0, 130):
        raise RuntimeError("[api] Uvicorn exited with an error.")


@task(name="start_mlflow_ui", log_prints=True)
def task_start_mlflow_ui():
    print("[mlflow] Starting MLflow UI on http://127.0.0.1:5000 …")
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            MLFLOW_TRACKING_URI,
            "--host",
            "0.0.0.0",
            "--port",
            "5000",
        ]
    )
    print("[mlflow] UI started in background.")


# ══════════════════════════════════════════════════════════
# ░░  TASKS — DOCKER/CD
# ══════════════════════════════════════════════════════════

@task(name="docker_build_image", log_prints=True)
def task_docker_build_image():
    local_image = f"{DOCKER_IMAGE_NAME}:{DOCKER_IMAGE_TAG}"
    print(f"[docker] Building image '{local_image}' …")
    result = subprocess.run(
        ["docker", "build", "-t", local_image, "."],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"docker build failed:\n{result.stderr}")
    print("[docker] Build complete.")
    return local_image


@task(name="docker_run_container", log_prints=True)
def task_docker_run_container(local_image: str):
    print(f"[docker] Running container '{DOCKER_CONTAINER_NAME}' from '{local_image}' …")

    subprocess.run(
        ["docker", "rm", "-f", DOCKER_CONTAINER_NAME],
        capture_output=True,
        text=True,
    )

    command = [
        "docker",
        "run",
        "-d",
        "--name",
        DOCKER_CONTAINER_NAME,
        "-p",
        f"{DOCKER_LOCAL_PORT}:8000",
    ]

    env_passthrough = {
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
        "MLFLOW_MODEL_URI": os.getenv("MLFLOW_MODEL_URI"),
        "MLFLOW_RUN_ID": os.getenv("MLFLOW_RUN_ID"),
        "MLFLOW_ARTIFACT_PATH": os.getenv("MLFLOW_ARTIFACT_PATH"),
        "MODEL_PATH": os.getenv("MODEL_PATH"),
    }
    for key, value in env_passthrough.items():
        if value:
            command.extend(["-e", f"{key}={value}"])

    command.append(local_image)

    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"docker run failed:\n{result.stderr}")
    print(f"[docker] Container is running on http://127.0.0.1:{DOCKER_LOCAL_PORT}")


@task(name="docker_login", log_prints=True)
def task_docker_login():
    docker_username = os.getenv("DOCKERHUB_USERNAME")
    docker_token = os.getenv("DOCKERHUB_TOKEN")

    if not docker_username or not docker_token:
        raise RuntimeError(
            "Missing DOCKERHUB_USERNAME or DOCKERHUB_TOKEN environment variables."
        )

    print(f"[docker] Logging in to Docker Hub as '{docker_username}' …")
    result = subprocess.run(
        ["docker", "login", "-u", docker_username, "--password-stdin"],
        input=docker_token,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker login failed:\n{result.stderr}")
    print("[docker] Login successful.")
    return docker_username


@task(name="docker_tag_image", log_prints=True)
def task_docker_tag_image(local_image: str, docker_username: str):
    remote_image = f"{docker_username}/{DOCKER_IMAGE_NAME}:{DOCKER_IMAGE_TAG}"
    print(f"[docker] Tagging '{local_image}' as '{remote_image}' …")
    result = subprocess.run(
        ["docker", "tag", local_image, remote_image],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker tag failed:\n{result.stderr}")
    print("[docker] Tag complete.")
    return remote_image


@task(name="docker_push_image", log_prints=True)
def task_docker_push_image(remote_image: str):
    print(f"[docker] Pushing '{remote_image}' to Docker Hub …")
    result = subprocess.run(
        ["docker", "push", remote_image],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"docker push failed:\n{result.stderr}")
    print("[docker] Push complete.")


# ══════════════════════════════════════════════════════════
# ░░  TASKS — CODE QUALITY
# ══════════════════════════════════════════════════════════

@task(name="Formatage du code", log_prints=True)
def task_format_code():
    print("[format] Running Black formatter …")
    result = subprocess.run(["black"] + TARGET_FILES, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[format] Warning:\n{result.stderr}")
    print("[format] Done.")


@task(name="Qualité du code", log_prints=True)
def task_lint_code():
    print("[lint] Running Flake8 …")
    result = subprocess.run(
        ["flake8", "--max-line-length=120"] + TARGET_FILES, capture_output=True, text=True
    )
    print(result.stdout if result.stdout else "No linting issues found.")
    if result.returncode != 0:
        print(f"[lint] Issues detected (non-blocking):\n{result.stderr}")
    print("[lint] Done.")


@task(name="Sécurité du code", log_prints=True)
def task_security_check():
    print("[security] Running Bandit …")
    result = subprocess.run(["bandit", "-r"] + TARGET_FILES, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode not in (0, 1):
        print(f"[security] Warning:\n{result.stderr}")
    print("[security] Done.")


@task(name="tests unitaires", log_prints=True)
def task_run_unit_tests():
    os.makedirs(TEST_PATH, exist_ok=True)
    test_file = os.path.join(TEST_PATH, "test_model_pipeline.py")

    if not os.path.exists(test_file):
        print(f"[tests] Creating sample test file at {test_file} …")
        with open(test_file, "w") as f:
            f.write('''"""
test_model_pipeline.py
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_pipeline import train_model, evaluate_model, save_model, load_model, predict

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    x_train = np.random.rand(100, 9)
    y_train = np.random.randint(0, 2, 100)
    x_test  = np.random.rand(20, 9)
    y_test  = np.random.randint(0, 2, 20)
    return x_train, x_test, y_train, y_test

def test_train_model_returns_classifier(dummy_data):
    x_train, _, y_train, _ = dummy_data
    model = train_model(x_train, y_train, n_estimators=10)
    assert hasattr(model, "predict")

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
    result = predict(model, [0.5] * 9)
    assert result in (0, 1)
''')
        print("[tests] Test file created.")

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
    """Flow: prepare data + train model. One MLflow run holds everything."""
    with mlflow.start_run(run_name="flow_train"):
        mlflow.set_tag("prefect_flow", "train")
        x_train, x_test, y_train, y_test = task_prepare_data()
        task_train_model(x_train, y_train)
        # ↑ params + model artefact logged directly into this run


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    """Flow: load saved model + evaluate. One MLflow run holds everything."""
    with mlflow.start_run(run_name="flow_evaluate"):
        mlflow.set_tag("prefect_flow", "evaluate")
        model = task_load_model()
        x_train, x_test, y_train, y_test = task_prepare_data()
        task_evaluate_model(model, x_test, y_test)
        # ↑ metrics logged directly into this run


@flow(name="all", log_prints=True)
def flow_all():
    """
    Full pipeline — ONE MLflow run tracks everything:
      git pull → install → code quality → prepare → train → save → evaluate → predict
    """
    with mlflow.start_run(run_name="flow_all"):
        mlflow.set_tag("prefect_flow", "all")

        task_git_clone_or_pull()
        install_dependencies()

        task_format_code()
        task_lint_code()
        task_security_check()
        task_run_unit_tests()

        x_train, x_test, y_train, y_test = task_prepare_data()
        # ↑ data params logged here
        model = task_train_model(x_train, y_train)
        # ↑ model params + artefact logged here
        task_save_model(model)
        # ↑ .joblib artefact logged here
        task_evaluate_model(model, x_test, y_test)
        # ↑ accuracy / precision / recall / f1 logged here
        task_predict(model)


@flow(name="api", log_prints=True)
def flow_api():
    """Flow: start the FastAPI prediction service."""
    task_start_api()


@flow(name="mlflow_ui", log_prints=True)
def flow_mlflow_ui():
    """Flow: start the MLflow tracking UI (background process)."""
    task_start_mlflow_ui()


@flow(name="cd", log_prints=True)
def flow_cd():
    """
    CD flow for Docker:
      build image -> run container -> login -> tag -> push
    """
    with mlflow.start_run(run_name="flow_cd"):
        mlflow.set_tag("prefect_flow", "cd")
        local_image = task_docker_build_image()
        task_docker_run_container(local_image)
        docker_username = task_docker_login()
        remote_image = task_docker_tag_image(local_image, docker_username)
        task_docker_push_image(remote_image)
        mlflow.log_params(
            {
                "docker_image_name": DOCKER_IMAGE_NAME,
                "docker_image_tag": DOCKER_IMAGE_TAG,
                "docker_container_name": DOCKER_CONTAINER_NAME,
                "docker_local_port": DOCKER_LOCAL_PORT,
            }
        )
        mlflow.log_param("docker_remote_image", remote_image)


# ══════════════════════════════════════════════════════════
# ░░  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════

FLOWS = {
    "all": flow_all,
    "train": flow_train,
    "evaluate": flow_evaluate,
    "code": flow_code,
    "install": flow_install,
    "api": flow_api,
    "mlflow_ui": flow_mlflow_ui,
    "cd": flow_cd,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefect ML Pipeline")
    parser.add_argument(
        "--flow",
        choices=list(FLOWS.keys()),
        required=True,
        help="Which flow to run: all | train | evaluate | code | install | api | mlflow_ui | cd",
    )
    args = parser.parse_args()
    FLOWS[args.flow]()
