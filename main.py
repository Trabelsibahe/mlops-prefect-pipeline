"""
main.py
-------
Entry point for the Churn prediction pipeline.
Supports CLI arguments to run individual steps or the full pipeline.
MLflow tracking is enabled: each command opens a parent run.

Usage examples:
  python main.py --all                        # run everything
  python main.py --prepare                    # only load & split data
  python main.py --train                      # prepare + train
  python main.py --evaluate                   # prepare + train + evaluate
  python main.py --save                       # prepare + train + save model
  python main.py --load-predict               # load saved model + sample prediction
  python main.py --data path/to/file.csv      # custom data path
"""

import argparse
import mlflow
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)

# ── MLflow setup ───────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# ── Default paths ──────────────────────────────────────────
DEFAULT_DATA  = "Churn_Modelling.csv"
DEFAULT_MODEL = "classifier.joblib"

# ── Sample customer for inference demo ─────────────────────
# [CreditScore, Gender(0=Female/1=Male), Age, Tenure, Balance,
#  NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
SAMPLE_CUSTOMER = [850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]


def main():
    parser = argparse.ArgumentParser(description="Customer Churn ML Pipeline")
    parser.add_argument("--data",  default=DEFAULT_DATA,  help="Path to CSV data file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to save/load model")
    parser.add_argument("--all",          action="store_true", help="Run the full pipeline")
    parser.add_argument("--prepare",      action="store_true", help="Prepare data only")
    parser.add_argument("--train",        action="store_true", help="Prepare + train")
    parser.add_argument("--evaluate",     action="store_true", help="Prepare + train + evaluate")
    parser.add_argument("--save",         action="store_true", help="Prepare + train + save")
    parser.add_argument("--load-predict", action="store_true", help="Load saved model + predict sample")
    args = parser.parse_args()

    # ── Full pipeline ──────────────────────────────────────
    if args.all:
        print("\n=== FULL PIPELINE ===")
        with mlflow.start_run(run_name="main_all"):
            mlflow.set_tag("entry_point", "main.py --all")
            x_train, x_test, y_train, y_test = prepare_data(args.data)
            model = train_model(x_train, y_train)
            evaluate_model(model, x_test, y_test)
            save_model(model, args.model)
            loaded = load_model(args.model)
            predict(loaded, SAMPLE_CUSTOMER)
        return

    # ── Individual steps ───────────────────────────────────
    if args.prepare:
        print("\n=== STEP: prepare_data ===")
        # No model/metrics to log — just note the run for traceability
        with mlflow.start_run(run_name="main_prepare"):
            mlflow.set_tag("entry_point", "main.py --prepare")
            mlflow.log_param("data_path", args.data)
            prepare_data(args.data)

    elif args.train:
        print("\n=== STEPS: prepare_data + train_model ===")
        with mlflow.start_run(run_name="main_train"):
            mlflow.set_tag("entry_point", "main.py --train")
            mlflow.log_param("data_path", args.data)
            x_train, x_test, y_train, y_test = prepare_data(args.data)
            train_model(x_train, y_train)

    elif args.evaluate:
        print("\n=== STEPS: prepare_data + train_model + evaluate_model ===")
        with mlflow.start_run(run_name="main_evaluate"):
            mlflow.set_tag("entry_point", "main.py --evaluate")
            mlflow.log_param("data_path", args.data)
            x_train, x_test, y_train, y_test = prepare_data(args.data)
            model = train_model(x_train, y_train)
            evaluate_model(model, x_test, y_test)

    elif args.save:
        print("\n=== STEPS: prepare_data + train_model + save_model ===")
        with mlflow.start_run(run_name="main_save"):
            mlflow.set_tag("entry_point", "main.py --save")
            mlflow.log_param("data_path", args.data)
            mlflow.log_param("model_path", args.model)
            x_train, x_test, y_train, y_test = prepare_data(args.data)
            model = train_model(x_train, y_train)
            save_model(model, args.model)

    elif args.load_predict:
        print("\n=== STEPS: load_model + predict ===")
        with mlflow.start_run(run_name="main_load_predict"):
            mlflow.set_tag("entry_point", "main.py --load-predict")
            mlflow.log_param("model_path", args.model)
            model = load_model(args.model)
            predict(model, SAMPLE_CUSTOMER)

    else:
        print("No action specified. Use --help to see available options.")
        print("Quick start: python main.py --all")


if __name__ == "__main__":
    main()