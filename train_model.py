from __future__ import annotations

import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.svm import LinearSVC

from feature_extraction import (
    HIGH_RISK_THRESHOLD,
    REVIEW_RISK_THRESHOLD,
    build_baseline_numeric_feature_frame,
    build_numeric_feature_frame,
    build_urlfix_numeric_feature_frame,
    calculate_risk_breakdown,
    compute_handcrafted_features,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
ERROR_ANALYSIS_DIR = RESULTS_DIR / "error_analysis"
DEFAULT_DATASET = "dataset_homework.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.25
EVALUATION_DATASETS = ("challenge_dataset.csv", "adversarial_dataset.csv")
SOURCE_DATASET = "dataset_classroom.csv"
GENERALISATION_DATASET = "generalisation_corpus.csv"
GENERALISATION_SPEC = "scripts/build_generalisation_corpus.py"
CALIBRATION_GRID = tuple(round(0.05 * step, 2) for step in range(1, 20))
REVIEW_FALSE_ALARM_BUDGET = 0.20
HIGH_FALSE_ALARM_BUDGET = 0.05
REVIEW_PRECISION_FLOOR = 0.90
HIGH_PRECISION_FLOOR = 0.95

MODEL_CONFIGS = {
    "baseline_logistic_regression": {
        "feature_set": "baseline",
        "classifier": "Logistic Regression",
    },
    # Isolates the URL-extraction repair from the eight new features, which the
    # enhanced arms bundle together.
    "urlfix_logistic_regression": {
        "feature_set": "urlfix",
        "classifier": "Logistic Regression",
    },
    "enhanced_logistic_regression": {
        "feature_set": "enhanced",
        "classifier": "Logistic Regression",
    },
    "baseline_linear_svm": {
        "feature_set": "baseline",
        "classifier": "Linear-kernel SVM with probability calibration",
    },
    "enhanced_linear_svm": {
        "feature_set": "enhanced",
        "classifier": "Linear-kernel SVM with probability calibration",
    },
}


def build_pipeline(model_name: str) -> Pipeline:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model configuration: {model_name}")

    feature_builder = {
        "baseline": build_baseline_numeric_feature_frame,
        "urlfix": build_urlfix_numeric_feature_frame,
        "enhanced": build_numeric_feature_frame,
    }[MODEL_CONFIGS[model_name]["feature_set"]]
    text_features = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_features, "text"),
            (
                "numeric",
                FunctionTransformer(feature_builder, validate=False),
                "text",
            ),
        ]
    )

    if "SVM" in MODEL_CONFIGS[model_name]["classifier"]:
        classifier = CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=10000,
            ),
            method="sigmoid",
            cv=3,
        )
    else:
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    steps = [("preprocessor", preprocessor)]
    # Scaling covers both branches. An experiment that scaled only the text branch
    # and passed the counts through unscaled collapsed every arm, because char_count
    # reaches 278 and then dominates the linear model.
    steps.append(("scaler", MaxAbsScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def validate_dataset(df: pd.DataFrame, dataset_name: str) -> None:
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{dataset_name} is missing columns: {sorted(missing)}")
    if df["text"].isna().any() or df["label"].isna().any():
        raise ValueError(f"{dataset_name} contains missing text or labels")
    labels = set(df["label"].unique())
    if labels != {"phishing", "legitimate"}:
        raise ValueError(
            f"{dataset_name} must contain phishing and legitimate labels; found {labels}"
        )


def _normalise(text: str) -> str:
    return " ".join(str(text).split())


def load_evaluation_dataset(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {DATA_DIR / name}")
    frame = pd.read_csv(path).reset_index(drop=True)
    validate_dataset(frame, name)
    return frame


def exclude_evaluation_overlap(
    corpus: pd.DataFrame, evaluations: list[tuple[str, pd.DataFrame]], dataset_name: str
) -> tuple[pd.DataFrame, dict]:
    """Drop corpus rows reproduced verbatim in an evaluation set, keeping it unseen.

    Exact text matching only. Shared-sentence template overlap is reported by
    `template_overlap_audit` and is not excluded.
    """
    normalised_corpus = corpus["text"].map(_normalise)
    evaluation_texts: set[str] = set()
    per_dataset = {}
    for name, frame in evaluations:
        texts = {_normalise(text) for text in frame["text"]}
        per_dataset[name] = {
            "evaluation_rows": int(len(frame)),
            "corpus_rows_matching": int(normalised_corpus.isin(texts).sum()),
        }
        evaluation_texts |= texts

    matched = normalised_corpus.isin(evaluation_texts)
    retained = corpus[~matched].reset_index(drop=True)
    audit = {
        "corpus_dataset": dataset_name,
        "corpus_rows_before_exclusion": int(len(corpus)),
        "excluded_rows": int(matched.sum()),
        "excluded_source_indices": [int(i) for i in corpus.index[matched]],
        "corpus_rows_after_exclusion": int(len(retained)),
        "per_evaluation_dataset": per_dataset,
    }
    return retained, audit


def template_overlap_audit(train: pd.DataFrame, evaluation: pd.DataFrame) -> dict:
    """Count evaluation messages that reuse a sentence seen during training."""
    def sentences(frame: pd.DataFrame) -> set[str]:
        found = set()
        for text in frame["text"]:
            for sentence in re.split(r"(?<=[.!?])\s+", _normalise(text)):
                if len(sentence) > 20:
                    found.add(sentence)
        return found

    train_sentences = sentences(train)
    eval_sentences = sentences(evaluation)
    reused = sorted(s for s in eval_sentences if s in train_sentences)
    return {
        "training_sentences": len(train_sentences),
        "evaluation_sentences": len(eval_sentences),
        "shared_sentences": len(reused),
        "shared_sentence_examples": reused[:5],
    }


def _sentences(text: str) -> set[str]:
    return {s for s in re.split(r"(?<=[.!?])\s+", _normalise(text)) if len(s) > 20}


def sentence_connectivity_audit(frame: pd.DataFrame) -> dict:
    """Join messages that share a sentence and report the resulting groups.

    Template confounding shows up as few components holding all rows of one
    class each: a split of such a corpus measures template recall, not
    generalisation. Sentences shorter than 20 characters are ignored.
    """
    parent = list(range(len(frame)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    sentence_rows: dict[str, list[int]] = {}
    for index, text in enumerate(frame["text"]):
        for sentence in _sentences(str(text)):
            sentence_rows.setdefault(sentence, []).append(index)
    for rows in sentence_rows.values():
        for other in rows[1:]:
            parent[find(rows[0])] = find(other)

    components: dict[int, list[int]] = {}
    for index in range(len(frame)):
        components.setdefault(find(index), []).append(index)
    grouped = sorted(components.values(), key=len, reverse=True)
    return {
        "distinct_sentences": len(sentence_rows),
        "components": [
            {
                "rows": len(rows),
                "labels": {
                    key: int(value)
                    for key, value in frame.loc[rows, "label"].value_counts().items()
                },
            }
            for rows in grouped
        ],
    }


def load_corpus(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path).reset_index(drop=True)
    validate_dataset(frame, name)
    return frame


def deduplicate(frame: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    """Keep one copy of each message so a repeat cannot act as an extra sample."""
    normalised = frame["text"].map(_normalise)
    kept = frame[~normalised.duplicated()].reset_index(drop=True)
    return kept, {
        "dataset": name,
        "rows_before_deduplication": int(len(frame)),
        "duplicate_rows_removed": int(len(frame) - len(kept)),
        "rows_after_deduplication": int(len(kept)),
    }


def load_generalisation_corpus(name: str) -> tuple[pd.DataFrame, dict]:
    """Load the selection and calibration corpus and audit it against the supplied data."""
    frame = load_corpus(name)
    texts = {_normalise(value) for value in frame["text"]}
    corpus_sentences = {
        sentence
        for text in frame["text"]
        for sentence in _sentences(text)
    }
    supplied_sentences: set[str] = set()
    for corpus_name in (DEFAULT_DATASET, SOURCE_DATASET):
        for text in load_corpus(corpus_name)["text"]:
            supplied_sentences |= _sentences(text)
    evaluation_rows = int(
        frame["text"].map(_normalise).isin(_evaluation_texts()).sum()
    )
    audit = {
        "specification": GENERALISATION_SPEC,
        "rows": int(len(frame)),
        "class_counts": {
            key: int(value) for key, value in frame["label"].value_counts().items()
        },
        "cells": int(frame["cell"].nunique()),
        "rows_reproducing_an_evaluated_message": evaluation_rows,
        "sentences_shared_with_the_supplied_corpora": len(
            corpus_sentences & supplied_sentences
        ),
    }
    return frame, audit


def _evaluation_texts() -> set[str]:
    texts: set[str] = set()
    for name in EVALUATION_DATASETS:
        texts |= {_normalise(value) for value in load_corpus(name)["text"]}
    return texts


def generalisation_evaluation(
    pipeline: Pipeline, corpus: pd.DataFrame
) -> tuple[dict, list[float], list[int]]:
    """Score the frozen selection corpus with an arm fitted on the primary corpus."""
    labels = (corpus["label"] == "phishing").astype(int)
    probabilities = [float(v) for v in pipeline.predict_proba(corpus[["text"]])[:, 1]]
    predictions = pipeline.predict(corpus[["text"]])
    return metric_summary(labels, predictions), probabilities, labels.tolist()


def calibrate_thresholds(scores: list[float], labels) -> tuple[list[dict], dict, dict]:
    """Derive band cuts as false-alarm budgets on the calibration distribution.

    The review cut accepts up to 20 percent of legitimate calibration messages and
    the high cut up to 5 percent, so each band has a stated and inspectable cost
    rather than a decreed number. Cuts are properties of the recorded distribution
    and never read an evaluation dataset.
    """
    legitimate = sorted(
        score for score, label in zip(scores, labels) if int(label) == 0
    )
    positive = sorted(score for score, label in zip(scores, labels) if int(label) == 1)
    table = []
    for threshold in CALIBRATION_GRID:
        flagged = [score >= threshold for score in scores]
        true_positive = sum(
            hit and int(label) == 1 for hit, label in zip(flagged, labels)
        )
        false_positive = sum(
            hit and int(label) == 0 for hit, label in zip(flagged, labels)
        )
        table.append(
            {
                "threshold": threshold,
                "phishing_recall": round(true_positive / max(len(positive), 1), 4),
                "flagged_precision": round(
                    true_positive / max(true_positive + false_positive, 1), 4
                )
                if true_positive + false_positive
                else None,
                "flagged_rows": true_positive + false_positive,
                "false_positives": false_positive,
            }
        )

    def budget_cut(false_alarm_budget: float) -> float:
        """Score exceeded by only the given share of legitimate messages."""
        if not legitimate:
            return REVIEW_RISK_THRESHOLD
        rank = int(round((1.0 - false_alarm_budget) * (len(legitimate) - 1)))
        return round(min(legitimate[rank], 0.99), 4)

    review = budget_cut(REVIEW_FALSE_ALARM_BUDGET)
    high = max(review, budget_cut(HIGH_FALSE_ALARM_BUDGET))
    return (
        table,
        {
            "threshold": review,
            "rule": f"{REVIEW_FALSE_ALARM_BUDGET:.0%} false-alarm budget on legitimate calibration scores",
        },
        {
            "threshold": high,
            "rule": f"{HIGH_FALSE_ALARM_BUDGET:.0%} false-alarm budget on legitimate calibration scores",
        },
    )


def metric_summary(y_true, predictions) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", pos_label=1, zero_division=0
    )
    report = classification_report(
        y_true,
        predictions,
        target_names=["legitimate", "phishing"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 6),
        "phishing_precision": round(float(precision), 6),
        "phishing_recall": round(float(recall), 6),
        "phishing_f1": round(float(f1), 6),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 6),
        "confusion_matrix": confusion_matrix(y_true, predictions).astype(int).tolist(),
        "classification_report": report,
    }


def prediction_frame(
    model_name: str,
    split_name: str,
    frame: pd.DataFrame,
    labels,
    predictions,
    probabilities,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "model": model_name,
            "split": split_name,
            "source_row": frame.index.astype(int),
            "text": frame["text"].values,
            "true_label": [
                "phishing" if int(label) == 1 else "legitimate" for label in labels
            ],
            "predicted_label": [
                "phishing" if int(prediction) == 1 else "legitimate"
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
    return output


def choose_deployment_model(results: dict[str, dict]) -> str:
    """Rank arms on held-out F1, then off-template transfer F1, then recall, precision.

    The transfer term is what breaks the tie a saturated in-distribution split
    cannot resolve. Evaluation datasets never enter ranking or tie-breaking.
    """
    return max(
        results,
        key=lambda name: (
            results[name]["test"]["phishing_f1"],
            results[name]["generalisation"]["phishing_f1"],
            results[name]["test"]["phishing_recall"],
            results[name]["test"]["phishing_precision"],
            name == "enhanced_logistic_regression",
        ),
    )


def main() -> None:
    dataset_name = os.environ.get("PHISHING_DATASET", DEFAULT_DATASET)
    data_path = DATA_DIR / dataset_name
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path).reset_index(drop=True)
    validate_dataset(df, dataset_name)
    df, corpus_dedup = deduplicate(df, dataset_name)

    evaluations = [(name, load_evaluation_dataset(name)) for name in EVALUATION_DATASETS]
    df, overlap_audit = exclude_evaluation_overlap(df, evaluations, dataset_name)
    challenge = dict(evaluations)["challenge_dataset.csv"]

    train_indices, test_indices = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    train = df.loc[train_indices]
    test = df.loc[test_indices]
    y_train = (train["label"] == "phishing").astype(int)
    y_test = (test["label"] == "phishing").astype(int)
    y_challenge = (challenge["label"] == "phishing").astype(int)

    MODEL_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    forbidden = {value for value in df["text"].map(_normalise)}
    for _, frame in evaluations:
        forbidden |= {value for value in frame["text"].map(_normalise)}
    generalisation, generalisation_audit = load_generalisation_corpus(
        GENERALISATION_DATASET
    )
    if generalisation["text"].map(_normalise).isin(forbidden).any():
        raise ValueError(
            "The generalisation corpus reproduces a training or evaluation message"
        )

    results: dict[str, dict] = {}
    prediction_frames: list[pd.DataFrame] = []
    cross_risk_scores: dict[str, list[float]] = {}
    gen_risk_scores: dict[str, list[float]] = {}

    for model_name, config in MODEL_CONFIGS.items():
        pipeline = build_pipeline(model_name)
        pipeline.fit(train[["text"]], y_train)

        test_predictions = pipeline.predict(test[["text"]])
        test_probabilities = pipeline.predict_proba(test[["text"]])[:, 1]
        challenge_predictions = pipeline.predict(challenge[["text"]])
        challenge_probabilities = pipeline.predict_proba(challenge[["text"]])[:, 1]
        gen_metrics, gen_probabilities, gen_labels = generalisation_evaluation(
            pipeline, generalisation
        )
        gen_risk_scores[model_name] = [
            calculate_risk_breakdown(
                float(probability), compute_handcrafted_features(str(text))
            )["final_score"]
            for probability, text in zip(gen_probabilities, generalisation["text"])
        ]

        results[model_name] = {
            **config,
            "test": metric_summary(y_test, test_predictions),
            "generalisation": gen_metrics,
            "challenge": metric_summary(y_challenge, challenge_predictions),
        }
        prediction_frames.extend(
            [
                prediction_frame(
                    model_name,
                    "held_out_test",
                    test,
                    y_test,
                    test_predictions,
                    test_probabilities,
                ),
                prediction_frame(
                    model_name,
                    "generalisation",
                    generalisation,
                    gen_labels,
                    pipeline.predict(generalisation[["text"]]),
                    gen_probabilities,
                ),
                prediction_frame(
                    model_name,
                    "challenge",
                    challenge,
                    y_challenge,
                    challenge_predictions,
                    challenge_probabilities,
                ),
            ]
        )
        joblib.dump(pipeline, MODEL_DIR / f"{model_name}.joblib")

    selected_model_name = choose_deployment_model(results)
    deployment_pipeline = build_pipeline(selected_model_name)
    deployment_pipeline.fit(df[["text"]], (df["label"] == "phishing").astype(int))
    joblib.dump(deployment_pipeline, MODEL_DIR / "phishing_pipeline.joblib")

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_predictions.to_csv(PREDICTIONS_DIR / "comparative_predictions.csv", index=False)
    selected_errors = all_predictions[
        (all_predictions["model"] == selected_model_name)
        & (all_predictions["error_type"] != "")
    ].copy()
    selected_errors.to_csv(ERROR_ANALYSIS_DIR / "selected_model_errors.csv", index=False)

    calibration_table, review_point, high_point = calibrate_thresholds(
        gen_risk_scores[selected_model_name],
        (generalisation["label"] == "phishing").astype(int).tolist(),
    )
    for label, derived, shipped in (
        ("review", review_point["threshold"], REVIEW_RISK_THRESHOLD),
        ("high", high_point["threshold"], HIGH_RISK_THRESHOLD),
    ):
        if abs(derived - shipped) > 0.005:
            raise ValueError(
                f"The derived {label} cut {derived} disagrees with the shipped constant "
                f"{shipped} in feature_extraction.py. Update the constant from "
                "model/threshold_calibration.csv before shipping."
            )
    pd.DataFrame(calibration_table).to_csv(
        MODEL_DIR / "threshold_calibration.csv", index=False
    )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "rows": int(len(df)),
        "corpus_deduplication": corpus_dedup,
        "evaluation_overlap_exclusion": overlap_audit,
        "sentence_connectivity": sentence_connectivity_audit(df),
        "generalisation_corpus": generalisation_audit,
        "class_counts": {
            key: int(value) for key, value in df["label"].value_counts().items()
        },
        "split": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "training_rows": int(len(train)),
            "test_rows": int(len(test)),
            "index_reference": (
                "train_indices and test_indices address the retained corpus, "
                "which excludes rows overlapping the evaluation datasets"
            ),
            "train_indices": sorted(int(index) for index in train_indices),
            "test_indices": sorted(int(index) for index in test_indices),
        },
        "challenge_rows": int(len(challenge)),
        "challenge_template_overlap": template_overlap_audit(train, challenge),
        "threshold_calibration": {
            "calibration_set": (
                "the generalisation corpus scored by the selected arm, which never "
                "trains on it; evaluation datasets are not used for calibration"
            ),
            "review_point": review_point,
            "high_point": high_point,
            "grid": calibration_table,
            "shipped_thresholds": {
                "review": REVIEW_RISK_THRESHOLD,
                "high": HIGH_RISK_THRESHOLD,
            },
            "table": (
                MODEL_DIR / "threshold_calibration.csv"
            ).relative_to(ROOT).as_posix(),
        },
        "models": results,
        "selected_deployment_model": selected_model_name,
        "selection_rule": (
            "Corpus rows are deduplicated and every row reproduced in an evaluation "
            "dataset is removed, so evaluation messages are unseen. Arms are ranked by "
            "held-out phishing F1, then by phishing F1 on the frozen selection "
            "corpus, then by recall and "
            "precision; an exact tie favors native probabilities. Evaluation datasets "
            "enter neither ranking nor threshold choice."
        ),
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Dataset: {data_path}")
    print(
        f"Evaluation overlap excluded: {overlap_audit['excluded_rows']} of "
        f"{overlap_audit['corpus_rows_before_exclusion']} corpus rows"
    )
    print(f"Training rows: {len(train)} | Test rows: {len(test)}")
    print(
        f"Selection corpus: {generalisation_audit['rows']} rows "
        f"{generalisation_audit['class_counts']}, sharing "
        f"{generalisation_audit['sentences_shared_with_the_supplied_corpora']} sentences "
        f"with the supplied corpora"
    )
    for model_name, model_results in results.items():
        test_metrics = model_results["test"]
        gen_metrics = model_results["generalisation"]
        challenge_metrics = model_results["challenge"]
        print(
            f"{model_name}: test F1={test_metrics['phishing_f1']:.3f}, "
            f"selection-corpus F1={gen_metrics['phishing_f1']:.3f}, "
            f"challenge F1={challenge_metrics['phishing_f1']:.3f}"
        )
    print(f"Deployment model: {selected_model_name}")
    print(
        "Calibrated bands: review>=%s (%s), high>=%s (%s)"
        % (
            review_point["threshold"], review_point["rule"],
            high_point["threshold"], high_point["rule"],
        )
    )
    print(f"Saved models to: {MODEL_DIR}")
    print(f"Saved predictions to: {PREDICTIONS_DIR / 'comparative_predictions.csv'}")
    print(f"Saved error rows to: {ERROR_ANALYSIS_DIR / 'selected_model_errors.csv'}")
    print(f"Saved calibration to: {MODEL_DIR / 'threshold_calibration.csv'}")


if __name__ == "__main__":
    main()
