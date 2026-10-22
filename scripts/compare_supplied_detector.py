"""Reconstruct the supplied lab detector and score it end to end.

The supplied system trained one Logistic Regression on `dataset_large.csv` with
its own TF-IDF settings and a 20 percent split, scored URLs only in the
application layer, and cut risk at 0.45 (review) and 0.72 (high). This script
reproduces that detector from the supplied feature definitions and measures it
on the same evaluation sets as the submitted detector, so the before/after
comparison is an executed artifact rather than a hand-computed claim.

Fidelity anchor: the supplied lab's own `model/metrics.json` records 0.998
accuracy on its held-out split. The reconstruction must land within two rows of
that record, or the run stops rather than reporting a comparison built on a
different detector.

Run after the submitted detector's band analysis:

    python scripts/compare_supplied_detector.py

Writes results/comparison/supplied_detector.csv (band summary) and
results/predictions/supplied_detector_predictions.csv (row-level scores), and joins
the existing band summaries into results/comparison/before_after_comparison.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Keep the lecturer's core modules importable when this file is run directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from feature_extraction import (
    ACTION_PATTERNS,
    EMOTIONAL_PATTERNS,
    IP_URL_REGEX,
    SENSITIVE_PATTERNS,
    SUPPLIED_GENERIC_GREETING_PATTERNS,
    SUPPLIED_SUSPICIOUS_DOMAIN_REGEX,
    SUPPLIED_THREAT_PATTERNS,
    URL_REGEX,
    URGENCY_PATTERNS,
    _count_matches,
    compute_supplied_features,
)

RESULTS_DIR = ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
SUPPLIED_DATASET = "dataset_large.csv"
SUPPLIED_TEST_SIZE = 0.2
SUPPLIED_RANDOM_STATE = 42
SUPPLIED_REVIEW_CUT = 0.45
SUPPLIED_HIGH_CUT = 0.72
# Recorded in the supplied lab's model/metrics.json for its held-out split.
SUPPLIED_RECORDED_ACCURACY = 0.998
# Tolerance of two rows on the 500-message held-out split, for library drift.
ACCURACY_TOLERANCE = 0.004

EVALUATION_SETS = ("challenge_dataset.csv", "adversarial_dataset.csv")

SUPPLIED_TEXT_CATEGORIES = (
    URGENCY_PATTERNS,
    SENSITIVE_PATTERNS,
    ACTION_PATTERNS,
    SUPPLIED_THREAT_PATTERNS,
    SUPPLIED_GENERIC_GREETING_PATTERNS,
    EMOTIONAL_PATTERNS,
)


def supplied_feature_frame(series) -> pd.DataFrame:
    """Supplied training behaviour: URL and header inputs were always empty."""
    return pd.DataFrame(
        [compute_supplied_features(str(text), urls=[], headers=None) for text in series]
    )


def build_supplied_pipeline() -> Pipeline:
    text_features = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, stop_words="english"
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_features, "text"),
            ("numeric", FunctionTransformer(supplied_feature_frame, validate=False), "text"),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1200, class_weight="balanced"),
            ),
        ]
    )


def supplied_cue_count(text: str, urls: list[str]) -> int:
    """The supplied cue list: matched categories plus two URL flags."""
    count = sum(
        1 for patterns in SUPPLIED_TEXT_CATEGORIES if _count_matches(text, patterns)
    )
    if any(IP_URL_REGEX.search(url) for url in urls):
        count += 1
    if any(SUPPLIED_SUSPICIOUS_DOMAIN_REGEX.search(url) for url in urls):
        count += 1
    return count


def supplied_risk(probability: float, text: str, urls: list[str]) -> float:
    """The supplied application-layer risk formula, including its cue term."""
    cues = supplied_cue_count(text, urls)
    features = compute_supplied_features(text, urls=urls, headers=None)
    heuristic = min(
        1.0,
        (cues * 0.07)
        + (features.get("suspicious_domain_count", 0) * 0.12)
        + (features.get("ip_url_count", 0) * 0.15)
        + (features.get("urgency_count", 0) * 0.04)
        + (features.get("sensitive_count", 0) * 0.05)
        + (features.get("url_shortener_count", 0) * 0.08),
    )
    return round(min(1.0, 0.68 * float(probability) + 0.32 * heuristic), 4)


def before_after_table(supplied: pd.DataFrame, submitted: pd.DataFrame) -> pd.DataFrame:
    """Join recorded whole-detector results without rerunning or tuning either arm.

    The thresholds are included because this is a comparison of complete systems,
    not an isolation of feature effects. Selection-corpus rows are not external
    checks and are excluded from this view.
    """
    split_to_dataset = {
        Path(filename).stem.removesuffix("_dataset"): filename
        for filename in EVALUATION_SETS
    }
    after = submitted[submitted["split"].isin(split_to_dataset)].copy()
    after["dataset"] = after["split"].map(split_to_dataset)
    keys = ["dataset", "band"]
    measures = [
        "cut",
        "true_positives",
        "false_positives",
        "phishing_recall",
        "flagged_precision",
    ]
    before = supplied[keys + ["phishing_rows"] + measures].rename(
        columns={column: f"before_{column}" for column in measures}
    )
    after = after[keys + ["phishing_rows"] + measures].rename(
        columns={column: f"after_{column}" for column in measures}
    )
    joined = before.merge(
        after,
        on=keys,
        how="outer",
        suffixes=("_before", "_after"),
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Before and after must cover the same datasets and bands.")
    if not joined["phishing_rows_before"].eq(joined["phishing_rows_after"]).all():
        raise ValueError("Before and after must contain the same phishing-row counts.")
    joined["phishing_rows"] = joined["phishing_rows_before"]
    columns = keys + ["phishing_rows"]
    for measure in measures:
        columns.extend([f"before_{measure}", f"after_{measure}"])
    return joined[columns]


def main() -> None:
    submitted_path = COMPARISON_DIR / "end_to_end_bands.csv"
    if not submitted_path.exists():
        raise SystemExit(
            "Run `python scripts/analyze_results.py` first to create "
            "results/comparison/end_to_end_bands.csv."
        )
    submitted = pd.read_csv(submitted_path)
    corpus = pd.read_csv(ROOT / "data" / SUPPLIED_DATASET)
    train, test = train_test_split(
        corpus,
        test_size=SUPPLIED_TEST_SIZE,
        random_state=SUPPLIED_RANDOM_STATE,
        stratify=corpus["label"],
    )
    pipeline = build_supplied_pipeline()
    pipeline.fit(train[["text"]], (train["label"] == "phishing").astype(int))

    held_out_accuracy = accuracy_score(
        (test["label"] == "phishing").astype(int), pipeline.predict(test[["text"]])
    )
    if abs(held_out_accuracy - SUPPLIED_RECORDED_ACCURACY) > ACCURACY_TOLERANCE:
        raise SystemExit(
            f"Reconstruction fidelity check failed: held-out accuracy "
            f"{held_out_accuracy:.4f} differs from the supplied record "
            f"{SUPPLIED_RECORDED_ACCURACY} by more than {ACCURACY_TOLERANCE}"
        )

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = []
    for dataset_name in EVALUATION_SETS:
        frame = pd.read_csv(ROOT / "data" / dataset_name)
        for _, row in frame.iterrows():
            text = str(row["text"])
            urls = URL_REGEX.findall(text)
            probability = float(
                pipeline.predict_proba(pd.DataFrame({"text": [text]}))[0][1]
            )
            risk = supplied_risk(probability, text, urls)
            predictions.append(
                {
                    "dataset": dataset_name,
                    "text": text,
                    "true_label": row["label"],
                    "phishing_probability": round(probability, 6),
                    "risk_score": risk,
                    "band": (
                        "high" if risk >= SUPPLIED_HIGH_CUT
                        else "review" if risk >= SUPPLIED_REVIEW_CUT
                        else "low"
                    ),
                }
            )
    prediction_frame = pd.DataFrame(predictions)
    prediction_frame.to_csv(
        PREDICTIONS_DIR / "supplied_detector_predictions.csv", index=False
    )

    summary_rows = []
    for dataset_name in EVALUATION_SETS:
        subset = prediction_frame[prediction_frame["dataset"] == dataset_name]
        positives = int((subset["true_label"] == "phishing").sum())
        for band, cut in (("high", SUPPLIED_HIGH_CUT), ("review_or_above", SUPPLIED_REVIEW_CUT)):
            flagged = subset["risk_score"] >= cut
            true_positive = int(
                (flagged & (subset["true_label"] == "phishing")).sum()
            )
            false_positive = int(
                (flagged & (subset["true_label"] == "legitimate")).sum()
            )
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "band": band,
                    "cut": cut,
                    "phishing_rows": positives,
                    "true_positives": true_positive,
                    "false_positives": false_positive,
                    "phishing_recall": round(true_positive / max(positives, 1), 4),
                    "flagged_precision": round(
                        true_positive / max(true_positive + false_positive, 1), 4
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(COMPARISON_DIR / "supplied_detector.csv", index=False)
    comparison = before_after_table(summary, submitted)
    comparison.to_csv(COMPARISON_DIR / "before_after_comparison.csv", index=False)

    print(f"Supplied detector reconstruction on {SUPPLIED_DATASET}")
    print(
        f"  held-out accuracy {held_out_accuracy:.4f} "
        f"(supplied record {SUPPLIED_RECORDED_ACCURACY}, tolerance {ACCURACY_TOLERANCE})"
    )
    print(summary.to_string(index=False))
    print(f"Row-level scores in {PREDICTIONS_DIR / 'supplied_detector_predictions.csv'}")
    print(f"Side-by-side before/after in {COMPARISON_DIR / 'before_after_comparison.csv'}")


if __name__ == "__main__":
    main()
