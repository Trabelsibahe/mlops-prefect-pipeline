"""
app.py
------
FastAPI REST service exposing the predict() function from model_pipeline.py.

Start the server:
  uvicorn app:app --reload --host 0.0.0.0 --port 8000

Or in background:
  nohup uvicorn app:app --reload --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &

Interactive docs:
  http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import os
import tempfile
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import mlflow

from model_pipeline import load_model, predict

# ── Config ─────────────────────────────────────────────────
MODEL_PATH = "classifier.joblib"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")
MODEL_SOURCE = "local"


def resolve_model_path() -> str:
    """
    Resolve model path with this priority:
    1) local file (MODEL_PATH)
    2) MLflow URI (MLFLOW_MODEL_URI)
    3) MLflow run artifact (MLFLOW_RUN_ID + MLFLOW_ARTIFACT_PATH)
    """
    local_model_path = os.getenv("MODEL_PATH", MODEL_PATH)
    if os.path.exists(local_model_path):
        return local_model_path

    download_dir = os.getenv(
        "MLFLOW_DOWNLOAD_DIR",
        os.path.join(tempfile.gettempdir(), "mlflow_downloaded_model"),
    )
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    mlflow_model_uri = os.getenv("MLFLOW_MODEL_URI")
    if mlflow_model_uri:
        downloaded_path = mlflow.artifacts.download_artifacts(
            artifact_uri=mlflow_model_uri,
            dst_path=download_dir,
        )
        if os.path.isdir(downloaded_path):
            candidate = os.path.join(downloaded_path, "classifier.joblib")
            if os.path.exists(candidate):
                return candidate
        if os.path.exists(downloaded_path):
            return downloaded_path

    mlflow_run_id = os.getenv("MLFLOW_RUN_ID")
    mlflow_artifact_path = os.getenv("MLFLOW_ARTIFACT_PATH", "saved_model/classifier.joblib")
    if mlflow_run_id:
        downloaded_path = mlflow.artifacts.download_artifacts(
            run_id=mlflow_run_id,
            artifact_path=mlflow_artifact_path,
            dst_path=download_dir,
        )
        if os.path.exists(downloaded_path):
            return downloaded_path

    raise RuntimeError(
        "No model found. Set MODEL_PATH, or provide MLflow artifact settings: "
        "MLFLOW_MODEL_URI or (MLFLOW_RUN_ID + MLFLOW_ARTIFACT_PATH)."
    )

# ── Load model at startup ──────────────────────────────────
resolved_model_path = resolve_model_path()
if os.path.abspath(resolved_model_path) != os.path.abspath(os.getenv("MODEL_PATH", MODEL_PATH)):
    MODEL_SOURCE = "mlflow_artifact"
model = load_model(resolved_model_path)

# ── FastAPI app ────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="REST API exposing the predict() function of the Churn ML model.",
    version="1.0.0",
)

# Allow requests from the HTML UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────
class CustomerFeatures(BaseModel):
    """
    Input features for a single customer.
    Feature order matches the model training:
    CreditScore, Gender, Age, Tenure, Balance,
    NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    """
    CreditScore:      float = Field(..., example=650,       description="Credit score (300–850)")
    Gender:           int   = Field(..., example=1,         description="0 = Female, 1 = Male")
    Age:              int   = Field(..., example=35,         description="Customer age")
    Tenure:           int   = Field(..., example=5,          description="Years as customer (0–10)")
    Balance:          float = Field(..., example=75000.0,   description="Account balance")
    NumOfProducts:    int   = Field(..., example=2,          description="Number of bank products (1–4)")
    HasCrCard:        int   = Field(..., example=1,          description="Has credit card? 0 or 1")
    IsActiveMember:   int   = Field(..., example=1,          description="Is active member? 0 or 1")
    EstimatedSalary:  float = Field(..., example=50000.0,   description="Estimated annual salary")


class PredictionResponse(BaseModel):
    prediction:  int  = Field(..., description="0 = Not Churned, 1 = Churned")
    label:       str  = Field(..., description="Human-readable result")
    probability: float = Field(..., description="Churn probability (0.0 – 1.0)")


# ── Routes ─────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    """Serve the frontend dashboard."""
    return FileResponse(INDEX_HTML_PATH, media_type="text/html")


@app.get("/health", tags=["Health"])
def health():
    """Health check — confirms the API is running."""
    return {
        "status": "ok",
        "message": "Churn Prediction API is running.",
        "model_source": MODEL_SOURCE,
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerFeatures):
    """
    Predict whether a customer will churn.

    Send a JSON body with the 9 customer features and receive:
    - prediction : 0 (stays) or 1 (churns)
    - label      : 'Not Churned' or 'Churned'
    - probability: churn probability score
    """
    try:
        sample = [
            customer.CreditScore,
            customer.Gender,
            customer.Age,
            customer.Tenure,
            customer.Balance,
            customer.NumOfProducts,
            customer.HasCrCard,
            customer.IsActiveMember,
            customer.EstimatedSalary,
        ]

        prediction = predict(model, sample)
        probability = float(model.predict_proba([sample])[0][1])
        label = "Churned" if prediction == 1 else "Not Churned"

        return PredictionResponse(
            prediction=int(prediction),
            label=label,
            probability=round(probability, 4),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/retrain", tags=["Retrain"])
def retrain(n_estimators: int = 100, random_state: int = 42):
    """
    (Perspective) Retrain the model with new hyperparameters.
    Requires Churn_Modelling.csv to be present.
    """
    try:
        from model_pipeline import prepare_data, train_model, save_model
        global model
        x_train, x_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
        model = train_model(x_train, y_train, n_estimators=n_estimators, random_state=random_state)
        save_model(model, MODEL_PATH)
        return {"status": "success", "message": f"Model retrained with {n_estimators} estimators."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain error: {str(e)}")