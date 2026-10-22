from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Support direct invocation from scripts/ without an installed project package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from feature_extraction import (
    HIGH_RISK_THRESHOLD,
    REVIEW_RISK_THRESHOLD,
    calculate_risk_breakdown,
    compute_handcrafted_features,
    extract_content,
    phishing_cues,
)


MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
ERROR_ANALYSIS_DIR = RESULTS_DIR / "error_analysis"
# Ordered so ablation output is stable across runs and interpreters.
NEW_FEATURES = (
    "suspicious_tld_count",
    "domain_similarity_count",
    "complex_url_count",
    "credential_request_count",
    "financial_pressure_count",
    "impersonation_signal_count",
    "polite_action_count",
    "reassurance_count",
)

ERROR_INTERPRETATIONS = {
    0: {
        "cause": (
            "The classifier treats quoted urgency, password, and click language as "
            "direct instructions and does not model the educational negation."
        ),
        "missing_control": (
            "Add quotation/negation scope features or a contextual language model "
            "that distinguishes warnings about phishing from phishing requests."
        ),
    },
    1: {
        "cause": (
            "Threat phrases dominate even though 'case study' identifies an academic "
            "discussion rather than an operational account notice."
        ),
        "missing_control": (
            "Add discourse-purpose features for training, simulation, and case-study "
            "contexts, supported by more benign security-awareness examples."
        ),
    },
    6: {
        "cause": (
            "The lure uses familiar collaboration language, no URL, no urgency, and "
            "no credential term, so content-only signals resemble routine work."
        ),
        "missing_control": (
            "Inspect attachment type and behavior, sender relationship, message-thread "
            "history, and attachment sandbox results."
        ),
    },
    7: {
        "cause": (
            "The calm wording and neutral-looking domain avoid the lexical URL rules; "
            "the model has no domain-age, reputation, redirect, or page-content data."
        ),
        "missing_control": (
            "Add live domain reputation and age, redirect-chain inspection, TLS and "
            "registration signals, and landing-page credential-form detection."
        ),
    },
    8: {
        "cause": (
            "The message carries no destination at all, so every URL and header signal "
            "is absent and the wording matches routine document sharing."
        ),
        "missing_control": (
            "Inspect attachment type and behaviour, sandbox results, and the sender's "
            "history with this recipient, since the content alone does not expose the lure."
        ),
    },
    11: {
        "cause": (
            "A voicemail notification on a neutral domain shares no vocabulary with the "
            "corpus phishing family, and no reputation or call-audio evidence exists."
        ),
        "missing_control": (
            "Add destination reputation, media-content inspection, and cross-message "
            "frequency checks for newly seen link patterns."
        ),
    },
}

# Interpretations for false negatives on the supplied challenge set, keyed by the
# row index in data/challenge_dataset.csv. The analysis reads the error rows off
# the deployed model's predictions and fails if a selected row has no entry.
CHALLENGE_INTERPRETATIONS = {
    2: {
        "cause": (
            "Only the action and financial features fire. 'Staff details' is not in "
            "the sensitive-information list, and 'the secure link' is prose rather "
            "than a URL, so every destination feature stays at zero."
        ),
        "missing_control": (
            "Extend the sensitive-information vocabulary to personal and staff "
            "details, and treat an unlinked reference to a 'secure link' or "
            "'validation page' as a destination cue."
        ),
    },
    3: {
        "cause": (
            "The credential-request feature fires twice, but the TF-IDF vocabulary "
            "('research repository', 'shared file') sits in the legitimate template "
            "family, and the classifier weights that vocabulary above the counts."
        ),
        "missing_control": (
            "Raise the weight of credential-request evidence relative to topic "
            "vocabulary, for example by training on messages where the same topic "
            "words appear with both labels."
        ),
    },
    4: {
        "cause": (
            "No handcrafted feature and no cue fires. 'Authentication mismatch', "
            "'validation page' and 'service disruption' appear in no pattern list, "
            "and the destination is referred to but never linked."
        ),
        "missing_control": (
            "Add an account-status pretext category (mismatch, validation, "
            "disruption, synchronisation) and an unlinked-destination cue, so a "
            "message that points at a page without a URL still raises evidence."
        ),
    },
}


def comparison_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for model_name, model in metrics["models"].items():
        for split in ("test", "generalisation", "challenge"):
            result = model[split]
            rows.append(
                {
                    "model": model_name,
                    "feature_set": model["feature_set"],
                    "classifier": model["classifier"],
                    "split": split,
                    "accuracy": result["accuracy"],
                    "phishing_precision": result["phishing_precision"],
                    "phishing_recall": result["phishing_recall"],
                    "phishing_f1": result["phishing_f1"],
                    "macro_f1": result["macro_f1"],
                    "tn": result["confusion_matrix"][0][0],
                    "fp": result["confusion_matrix"][0][1],
                    "fn": result["confusion_matrix"][1][0],
                    "tp": result["confusion_matrix"][1][1],
                }
            )
    return pd.DataFrame(rows)


def adversarial_predictions(pipeline) -> pd.DataFrame:
    """Full row-level predictions on the adversarial set, with error counts."""
    data = pd.read_csv(ROOT / "data" / "adversarial_dataset.csv")
    predictions = pipeline.predict(data[["text"]])
    probabilities = pipeline.predict_proba(data[["text"]])[:, 1]
    frame = pd.DataFrame(
        {
            "source_row": data.index.astype(int),
            "scenario": data["scenario"],
            "text": data["text"],
            "true_label": data["label"],
            "predicted_label": [
                "phishing" if int(value) == 1 else "legitimate"
                for value in predictions
            ],
            "phishing_probability": [round(float(value), 6) for value in probabilities],
        }
    )
    frame["error_type"] = ""
    false_positive = (frame["true_label"] == "legitimate") & (
        frame["predicted_label"] == "phishing"
    )
    false_negative = (frame["true_label"] == "phishing") & (
        frame["predicted_label"] == "legitimate"
    )
    frame.loc[false_positive, "error_type"] = "false_positive"
    frame.loc[false_negative, "error_type"] = "false_negative"
    return frame


def build_error_analysis(pipeline) -> list[dict]:
    """Derive the two most confident false positives and false negatives from output.

    Errors are read off the predictions, never assumed from row positions, and the
    run fails rather than silently reporting a correct classification as an error.
    """
    frame = adversarial_predictions(pipeline)
    frame = frame[frame["error_type"] != ""]
    data = pd.read_csv(ROOT / "data" / "adversarial_dataset.csv")
    cases = []
    for error_type, order in (("false_positive", "descending"), ("false_negative", "ascending")):
        subset = frame[frame["error_type"] == error_type]
        if len(subset) < 2:
            raise ValueError(
                f"Only {len(subset)} {error_type} cases found on the adversarial set; "
                "the error analysis requires two of each type"
            )
        subset = subset.sort_values(
            "phishing_probability",
            ascending=order == "ascending",
            kind="stable",
        ).head(2)
        for _, row in subset.iterrows():
            if int(row["source_row"]) not in ERROR_INTERPRETATIONS:
                raise ValueError(
                    f"Adversarial case {row['source_row']} has no recorded interpretation"
                )
            features = compute_handcrafted_features(row["text"])
            cases.append(
                {
                    "dataset": "adversarial_dataset.csv",
                    "source_row": int(row["source_row"]),
                    "error_type": error_type,
                    "scenario": data.loc[int(row["source_row"]), "scenario"],
                    "text": row["text"],
                    "true_label": row["true_label"],
                    "predicted_label": row["predicted_label"],
                    "phishing_probability": row["phishing_probability"],
                    "nonzero_features": {
                        key: value for key, value in features.items() if value > 0
                    },
                    "triggered_cues": phishing_cues(row["text"]),
                    **ERROR_INTERPRETATIONS[int(row["source_row"])],
                }
            )
    return cases


def challenge_error_cases(pipeline) -> list[dict]:
    """Every misclassified row of the supplied challenge set, read off predictions.

    The challenge set is evaluation data: these cases are analysed and proposed
    improvements are recorded, but no rule or weight is tuned on them.
    """
    data = pd.read_csv(ROOT / "data" / "challenge_dataset.csv")
    predictions = pipeline.predict(data[["text"]])
    probabilities = pipeline.predict_proba(data[["text"]])[:, 1]
    cases = []
    for index, row in data.iterrows():
        predicted = "phishing" if int(predictions[index]) == 1 else "legitimate"
        if predicted == row["label"]:
            continue
        error_type = (
            "false_negative" if row["label"] == "phishing" else "false_positive"
        )
        if error_type == "false_negative" and int(index) not in CHALLENGE_INTERPRETATIONS:
            raise ValueError(
                f"Challenge case {index} has no recorded interpretation"
            )
        features = compute_handcrafted_features(row["text"])
        cases.append(
            {
                "dataset": "challenge_dataset.csv",
                "source_row": int(index),
                "error_type": error_type,
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": predicted,
                "phishing_probability": round(float(probabilities[index]), 6),
                "nonzero_features": {
                    key: value for key, value in features.items() if value > 0
                },
                "triggered_cues": phishing_cues(row["text"]),
                **CHALLENGE_INTERPRETATIONS.get(int(index), {}),
            }
        )
    return cases


def risk_impact_table(pipeline) -> pd.DataFrame:
    samples = [
        ROOT / "sample_inputs" / "hard_phishing_polite.html",
        ROOT / "sample_inputs" / "phishing_payroll.docx",
        ROOT / "sample_inputs" / "legitimate_internal_notice.eml",
    ]
    rows = []
    for path in samples:
        parsed = extract_content(str(path))
        probability = float(
            pipeline.predict_proba(pd.DataFrame({"text": [parsed["text"]]}))[0][1]
        )
        features = compute_handcrafted_features(
            parsed["text"], parsed["urls"], parsed["headers"]
        )
        ablated = features.copy()
        for feature in NEW_FEATURES:
            ablated[feature] = 0.0
        without_new = calculate_risk_breakdown(probability, ablated)["final_score"]
        with_new = calculate_risk_breakdown(probability, features)["final_score"]
        rows.append(
            {
                "sample": path.name,
                "model_probability": round(probability, 4),
                "risk_without_new_features": without_new,
                "risk_with_new_features": with_new,
                "risk_change": round(with_new - without_new, 4),
                "new_features_triggered": ", ".join(
                    feature for feature in NEW_FEATURES if features.get(feature, 0) > 0
                )
                or "none",
            }
        )
    return pd.DataFrame(rows)


def end_to_end_band_table(pipeline) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the deployed detector end to end at the calibrated band cuts.

    Uses the deployment pipeline, the arm refit on the full retained corpus.
    The challenge and adversarial sets are external checks; the selection corpus
    supplies the calibration distribution itself, so its band figures are
    in-sample for the cuts and are labelled as such. Returns the band summary
    and the per-row risk scores it was computed from.
    """
    rows = []
    row_level = []
    for split, filename in (
        ("challenge", "challenge_dataset.csv"),
        ("adversarial", "adversarial_dataset.csv"),
        ("selection_corpus", "generalisation_corpus.csv"),
    ):
        frame = pd.read_csv(ROOT / "data" / filename)
        labels = (frame["label"] == "phishing").astype(int)
        probabilities = pipeline.predict_proba(frame[["text"]])[:, 1]
        risks = [
            calculate_risk_breakdown(
                float(probability), compute_handcrafted_features(str(text))
            )["final_score"]
            for probability, text in zip(probabilities, frame["text"])
        ]
        for index, (probability, risk) in enumerate(zip(probabilities, risks)):
            row_level.append(
                {
                    "split": split,
                    "source_row": int(index),
                    "true_label": frame.loc[index, "label"],
                    "phishing_probability": round(float(probability), 6),
                    "risk_score": risk,
                    "band": (
                        "high" if risk >= HIGH_RISK_THRESHOLD
                        else "review" if risk >= REVIEW_RISK_THRESHOLD
                        else "low"
                    ),
                }
            )
        for band, cut in (
            ("high", HIGH_RISK_THRESHOLD),
            ("review_or_above", REVIEW_RISK_THRESHOLD),
        ):
            flagged = [risk >= cut for risk in risks]
            true_positive = sum(
                hit and int(label) == 1 for hit, label in zip(flagged, labels)
            )
            false_positive = sum(
                hit and int(label) == 0 for hit, label in zip(flagged, labels)
            )
            positives = int(labels.sum())
            rows.append(
                {
                    "split": split,
                    "band": band,
                    "cut": cut,
                    "phishing_rows": positives,
                    "flagged": true_positive + false_positive,
                    "true_positives": true_positive,
                    "false_positives": false_positive,
                    "phishing_recall": round(
                        true_positive / max(positives, 1), 4
                    ),
                    "flagged_precision": round(
                        true_positive / max(true_positive + false_positive, 1), 4
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(row_level)


def main() -> None:
    with open(MODEL_DIR / "metrics.json", encoding="utf-8") as file:
        metrics = json.load(file)
    pipeline = joblib.load(MODEL_DIR / "phishing_pipeline.joblib")
    for directory in (COMPARISON_DIR, PREDICTIONS_DIR, ERROR_ANALYSIS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    comparison = comparison_table(metrics)
    comparison.to_csv(COMPARISON_DIR / "model_comparison.csv", index=False)
    risk_impact = risk_impact_table(pipeline)
    risk_impact.to_csv(COMPARISON_DIR / "risk_scoring_impact.csv", index=False)
    end_to_end, end_to_end_rows = end_to_end_band_table(pipeline)
    end_to_end.to_csv(COMPARISON_DIR / "end_to_end_bands.csv", index=False)
    end_to_end_rows.to_csv(PREDICTIONS_DIR / "end_to_end_rows.csv", index=False)
    adversarial = adversarial_predictions(pipeline)
    adversarial.to_csv(PREDICTIONS_DIR / "adversarial_predictions.csv", index=False)
    errors = challenge_error_cases(pipeline) + build_error_analysis(pipeline)
    with open(ERROR_ANALYSIS_DIR / "error_analysis.json", "w", encoding="utf-8") as file:
        json.dump(errors, file, indent=2)

    print("Model comparison")
    print(comparison.to_string(index=False))
    print("\nRisk-scoring impact")
    print(risk_impact.to_string(index=False))
    print("\nEnd-to-end band performance (deployed arm at the calibrated cuts)")
    print(end_to_end.to_string(index=False))
    challenge_rows = end_to_end_rows[end_to_end_rows["split"] == "challenge"]
    for label in ("phishing", "legitimate"):
        scores = challenge_rows.loc[challenge_rows["true_label"] == label, "risk_score"]
        print(f"challenge risk range, {label}: {scores.min():.4f} to {scores.max():.4f}")
    print("\nSelected error cases")
    for case in errors:
        print(
            f"{case['dataset']} {case['error_type']}: row {case['source_row']} "
            f"p={case['phishing_probability']:.3f} | "
            f"{case.get('scenario', case['text'][:60])}"
        )


if __name__ == "__main__":
    main()
