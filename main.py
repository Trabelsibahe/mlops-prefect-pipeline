"""
main.py
-------
Entry point for the Churn prediction pipeline.
MLflow tracking: each command opens ONE run; model_pipeline.py
functions log directly into it.

Usage examples:
  python main.py --all
  python main.py --prepare
  python main.py --train
  python main.py --evaluate
  python main.py --save
  python main.py --load-predict
  python main.py --data path/to/file.csv
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

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

DEFAULT_DATA    = "Churn_Modelling.csv"
DEFAULT_MODEL   = "classifier.joblib"
SAMPLE_CUSTOMER = [850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]


def main():
    parser = argparse.ArgumentParser(description="Customer Churn ML Pipeline")
    parser.add_argument("--data",         default=DEFAULT_DATA,  help="Path to CSV data file")
    parser.add_argument("--model",        default=DEFAULT_MODEL, help="Path to save/load model")
    parser.add_argument("--all",          action="store_true", help="Run the full pipeline")
    parser.add_argument("--prepare",      action="store_true", help="Prepare data only")
    parser.add_argument("--train",        action="store_true", help="Prepare + train")
    parser.add_argument("--evaluate",     action="store_true", help="Prepare + train + evaluate")
    parser.add_argument("--save",         action="store_true", help="Prepare + train + save")
    parser.add_argument("--load-predict", action="store_true", help="Load saved model + predict sample")
    args = parser.parse_args()

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

    if args.prepare:
        print("\n=== STEP: prepare_data ===")
        with mlflow.start_run(run_name="main_prepare"):
            mlflow.set_tag("entry_point", "main.py --prepare")
            prepare_data(args.data)

    elif args.train:
        print("\n=== STEPS: prepare_data + train_model ===")
        with mlflow.start_run(run_name="main_train"):
            mlflow.set_tag("entry_point", "main.py --train")
            x_train, x_test, y_train, y_test = prepare_data(args.data)
            train_model(x_train, y_train)

    elif args.evaluate:
        print("\n=== STEPS: prepare_data + train_model + evaluate_model ===")
        with mlflow.start_run(run_name="main_evaluate"):
            mlflow.set_tag("entry_point", "main.py --evaluate")
            x_train, x_test, y_train, y_test = prepare_data(args.data)
            model = train_model(x_train, y_train)
            evaluate_model(model, x_test, y_test)

    elif args.save:
        print("\n=== STEPS: prepare_data + train_model + save_model ===")
        with mlflow.start_run(run_name="main_save"):
            mlflow.set_tag("entry_point", "main.py --save")
            x_train, x_test, y_train, y_test = prepare_data(args.data)
            model = train_model(x_train, y_train)
            save_model(model, args.model)

    elif args.load_predict:
        print("\n=== STEPS: load_model + predict ===")
        with mlflow.start_run(run_name="main_load_predict"):
            mlflow.set_tag("entry_point", "main.py --load-predict")
            model = load_model(args.model)
            predict(model, SAMPLE_CUSTOMER)

    else:
        print("No action specified. Use --help to see available options.")
        print("Quick start: python main.py --all")


if __name__ == "__main__":
    main()