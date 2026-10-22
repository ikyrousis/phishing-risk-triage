from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved phishing model.")
    parser.add_argument(
        "--dataset",
        default="challenge_dataset.csv",
        help="CSV filename in data/ or an explicit path.",
    )
    parser.add_argument(
        "--model",
        default="phishing_pipeline.joblib",
        help="Model filename in model/ or an explicit path.",
    )
    parser.add_argument(
        "--output",
        default="evaluation_predictions.csv",
        help="Prediction CSV filename written under results/predictions/.",
    )
    return parser.parse_args()


def resolve_path(value: str, default_directory: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else default_directory / path


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.dataset, ROOT / "data")
    model_path = resolve_path(args.model, ROOT / "model")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    df = pd.read_csv(data_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain text and label columns")
    y = (df["label"] == "phishing").astype(int)
    pipeline = joblib.load(model_path)
    predictions = pipeline.predict(df[["text"]])
    probabilities = pipeline.predict_proba(df[["text"]])[:, 1]

    print("Accuracy:", round(accuracy_score(y, predictions), 4))
    print("Confusion matrix:")
    print(confusion_matrix(y, predictions))
    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            target_names=["legitimate", "phishing"],
            zero_division=0,
        )
    )

    output = pd.DataFrame(
        {
            "text": df["text"],
            "true_label": df["label"],
            "predicted_label": [
                "phishing" if prediction == 1 else "legitimate"
                for prediction in predictions
            ],
            "phishing_probability": probabilities,
        }
    )
    output["error_type"] = ""
    output.loc[
        (output["true_label"] == "legitimate")
        & (output["predicted_label"] == "phishing"),
        "error_type",
    ] = "false_positive"
    output.loc[
        (output["true_label"] == "phishing")
        & (output["predicted_label"] == "legitimate"),
        "error_type",
    ] = "false_negative"

    output_path = PREDICTIONS_DIR / Path(args.output).name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(
        json.dumps(
            {
                "dataset": str(data_path),
                "model": str(model_path),
                "rows": len(output),
                "errors": int((output["error_type"] != "").sum()),
                "saved_to": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
