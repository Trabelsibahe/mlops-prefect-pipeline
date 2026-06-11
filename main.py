"""
main.py
-------
Entry point for the Churn prediction pipeline.
Supports CLI arguments to run individual steps or the full pipeline.

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
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict,
)

# ── Default paths ──────────────────────────────────────────
DEFAULT_DATA = "Churn_Modelling.csv"
DEFAULT_MODEL = "classifier.joblib"

# ── Sample customer for inference demo ─────────────────────
# [CreditScore, Gender(0=Female/1=Male), Age, Tenure, Balance,
#  NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]
SAMPLE_CUSTOMER = [850, 0, 43, 2, 125510.82, 1, 1, 1, 79084.10]


def main():
    parser = argparse.ArgumentParser(description="Customer Churn ML Pipeline")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to CSV data file")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Path to save/load model"
    )
    parser.add_argument("--all", action="store_true", help="Run the full pipeline")
    parser.add_argument("--prepare", action="store_true", help="Prepare data only")
    parser.add_argument("--train", action="store_true", help="Prepare + train")
    parser.add_argument(
        "--evaluate", action="store_true", help="Prepare + train + evaluate"
    )
    parser.add_argument("--save", action="store_true", help="Prepare + train + save")
    parser.add_argument(
        "--load-predict", action="store_true", help="Load saved model + predict sample"
    )
    args = parser.parse_args()

    # ── Full pipeline ──────────────────────────────────────
    if args.all:
        print("\n=== FULL PIPELINE ===")
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
        prepare_data(args.data)

    elif args.train:
        print("\n=== STEPS: prepare_data + train_model ===")
        x_train, x_test, y_train, y_test = prepare_data(args.data)
        train_model(x_train, y_train)

    elif args.evaluate:
        print("\n=== STEPS: prepare_data + train_model + evaluate_model ===")
        x_train, x_test, y_train, y_test = prepare_data(args.data)
        model = train_model(x_train, y_train)
        evaluate_model(model, x_test, y_test)

    elif args.save:
        print("\n=== STEPS: prepare_data + train_model + save_model ===")
        x_train, x_test, y_train, y_test = prepare_data(args.data)
        model = train_model(x_train, y_train)
        save_model(model, args.model)

    elif args.load_predict:
        print("\n=== STEPS: load_model + predict ===")
        model = load_model(args.model)
        predict(model, SAMPLE_CUSTOMER)

    else:
        print("No action specified. Use --help to see available options.")
        print("Quick start: python main.py --all")


if __name__ == "__main__":
    main()
