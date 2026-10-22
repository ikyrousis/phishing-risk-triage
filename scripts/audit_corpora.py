"""Produce the corpus audits and the preprocessing control that the report cites.

Every number in the report's data-quality discussion and in its fixed-preprocessing
section is written by this script, so each can be regenerated and checked:

* duplicate rows, distinct sentences and sentence-sharing components for the
  supplied classroom and large corpora
* cue co-occurrence over every supplied and generated message, to show that the
  action cue and the polite cue are never raised by the same phrase
* the rejected preprocessing variant (scaling the text branch only) scored on the
  same split, selection corpus and challenge set as the shipped arms

Run after the data files are in place; no trained artifacts are required:

    python scripts/audit_corpora.py

Writes logs/corpus_audit.json and results/comparison/preprocessing_control.csv.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Resolve the core modules independently of the terminal's working directory.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

import train_model as tm
from feature_extraction import (
    CUE_ACTION_PATTERNS,
    _count_matches,
    build_baseline_numeric_feature_frame,
    build_numeric_feature_frame,
    build_urlfix_numeric_feature_frame,
    phishing_cues,
)

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "comparison"
LOGS_DIR = ROOT / "logs"
CUE_CORPORA = (
    "dataset_homework.csv",
    "challenge_dataset.csv",
    "generalisation_corpus.csv",
    "adversarial_dataset.csv",
)
FEATURE_BUILDERS = {
    "baseline": build_baseline_numeric_feature_frame,
    "urlfix": build_urlfix_numeric_feature_frame,
    "enhanced": build_numeric_feature_frame,
}


def corpus_audit(name: str) -> dict:
    frame = tm.load_corpus(name)
    deduplicated, dedup = tm.deduplicate(frame, name)
    connectivity = tm.sentence_connectivity_audit(deduplicated)
    return {
        "dataset": name,
        "rows": int(len(frame)),
        "duplicate_rows": dedup["duplicate_rows_removed"],
        "rows_after_deduplication": dedup["rows_after_deduplication"],
        "distinct_sentences": connectivity["distinct_sentences"],
        "components": connectivity["components"],
    }


def cue_cooccurrence() -> dict:
    """Count how the action, credential and polite cues co-occur.

    The action cue masks polite phrases before matching, so a verb inside a
    softened phrase raises the polite cue only. `prevented_double_counts` is the
    number of rows where the unmasked action patterns would have fired but the
    masked rule did not, which is exactly the set of rows where one phrasing
    would otherwise have raised two cues.
    """
    texts = pd.concat(
        [pd.read_csv(DATA_DIR / name)["text"] for name in CUE_CORPORA],
        ignore_index=True,
    )
    counts = {
        "rows": int(len(texts)),
        "credential_rows": 0,
        "credential_rows_with_action_cue": 0,
        "polite_rows": 0,
        "polite_rows_with_action_cue": 0,
        "prevented_double_counts": 0,
    }
    for text in texts:
        cues = phishing_cues(str(text))
        action = any(cue.startswith("Requested action") for cue in cues)
        credential = any(cue.startswith("Credential request") for cue in cues)
        polite = any(cue.startswith("Polite social engineering") for cue in cues)
        unmasked_action = _count_matches(str(text), CUE_ACTION_PATTERNS) > 0
        counts["credential_rows"] += credential
        counts["credential_rows_with_action_cue"] += action and credential
        counts["polite_rows"] += polite
        counts["polite_rows_with_action_cue"] += action and polite
        counts["prevented_double_counts"] += polite and unmasked_action and not action
    return counts


def build_text_only_scaling_pipeline(model_name: str) -> Pipeline:
    """The rejected variant: scale TF-IDF only, pass the counts through unscaled."""
    shipped = tm.build_pipeline(model_name)
    builder = FEATURE_BUILDERS[tm.MODEL_CONFIGS[model_name]["feature_set"]]
    text_branch = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True
                ),
            ),
            ("scaler", MaxAbsScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_branch, "text"),
            ("numeric", FunctionTransformer(builder, validate=False), "text"),
        ]
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("classifier", shipped.named_steps["classifier"])]
    )


def preprocessing_control() -> pd.DataFrame:
    """Score the rejected scaling variant on the shipped split and evaluation sets."""
    corpus = tm.load_corpus(tm.DEFAULT_DATASET)
    corpus, _ = tm.deduplicate(corpus, tm.DEFAULT_DATASET)
    evaluations = [(name, tm.load_evaluation_dataset(name)) for name in tm.EVALUATION_DATASETS]
    corpus, _ = tm.exclude_evaluation_overlap(corpus, evaluations, tm.DEFAULT_DATASET)
    train_indices, _ = train_test_split(
        corpus.index,
        test_size=tm.TEST_SIZE,
        random_state=tm.RANDOM_STATE,
        stratify=corpus["label"],
    )
    train = corpus.loc[train_indices]
    y_train = (train["label"] == "phishing").astype(int)
    selection = tm.load_corpus(tm.GENERALISATION_DATASET)
    challenge = dict(evaluations)["challenge_dataset.csv"]
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for model_name in tm.MODEL_CONFIGS:
            pipeline = build_text_only_scaling_pipeline(model_name)
            pipeline.fit(train[["text"]], y_train)
            selection_metrics = tm.metric_summary(
                (selection["label"] == "phishing").astype(int),
                pipeline.predict(selection[["text"]]),
            )
            challenge_metrics = tm.metric_summary(
                (challenge["label"] == "phishing").astype(int),
                pipeline.predict(challenge[["text"]]),
            )
            rows.append(
                {
                    "model": model_name,
                    "variant": "text-branch scaling only",
                    "selection_corpus_f1": round(selection_metrics["phishing_f1"], 3),
                    "selection_corpus_recall": round(selection_metrics["phishing_recall"], 3),
                    "challenge_f1": round(challenge_metrics["phishing_f1"], 3),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    audit = {
        "corpora": [corpus_audit("dataset_classroom.csv"), corpus_audit("dataset_large.csv")],
        "cue_cooccurrence": cue_cooccurrence(),
    }
    with open(LOGS_DIR / "corpus_audit.json", "w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)
    control = preprocessing_control()
    control.to_csv(COMPARISON_DIR / "preprocessing_control.csv", index=False)

    for item in audit["corpora"]:
        components = ", ".join(
            f"{component['rows']} rows {component['labels']}"
            for component in item["components"][:3]
        )
        print(
            f"{item['dataset']}: {item['rows']} rows, {item['duplicate_rows']} duplicates, "
            f"{item['rows_after_deduplication']} after deduplication, "
            f"{item['distinct_sentences']} distinct sentences, components: {components}"
        )
    print("Cue co-occurrence:", json.dumps(audit["cue_cooccurrence"]))
    print("\nPreprocessing control (text-branch scaling only)")
    print(control.to_string(index=False))
    print(f"Saved to {LOGS_DIR / 'corpus_audit.json'} and {COMPARISON_DIR / 'preprocessing_control.csv'}")


if __name__ == "__main__":
    main()
