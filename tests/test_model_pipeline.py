"""
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
