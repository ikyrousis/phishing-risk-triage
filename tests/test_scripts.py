from __future__ import annotations

import runpy
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

import app as application
import evaluate_model
import train_model
from scripts import analyze_results, audit_corpora, compare_supplied_detector
from scripts.compare_supplied_detector import before_after_table
from train_model import GENERALISATION_SPEC

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "build_generalisation_corpus.py",
    "analyze_results.py",
    "compare_supplied_detector.py",
    "audit_corpora.py",
    "capture_screenshots.py",
)


def test_supporting_scripts_resolve_the_project_from_an_unrelated_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    for filename in SCRIPT_NAMES:
        script = ROOT / "scripts" / filename
        namespace = runpy.run_path(str(script), run_name="script_import_check")
        assert namespace["ROOT"] == ROOT
        assert callable(namespace["main"])
        assert not (ROOT / filename).exists()
        if filename == "capture_screenshots.py":
            assert namespace["OUTPUT_DIR"] == ROOT / "results"
            assert namespace["OUTPUTS"] == (
                "screenshot-high-risk.png",
                "screenshot-medium-risk.png",
                "screenshot-low-risk.png",
            )
    assert (ROOT / GENERALISATION_SPEC).is_file()


def test_output_directories_separate_results_from_consolidated_logs():
    for module in (
        train_model,
        evaluate_model,
        analyze_results,
        audit_corpora,
        compare_supplied_detector,
    ):
        assert module.RESULTS_DIR == ROOT / "results"
    for module in (analyze_results, audit_corpora, compare_supplied_detector):
        assert module.COMPARISON_DIR == ROOT / "results" / "comparison"
    for module in (train_model, evaluate_model, analyze_results, compare_supplied_detector):
        assert module.PREDICTIONS_DIR == ROOT / "results" / "predictions"
    for module in (train_model, analyze_results):
        assert module.ERROR_ANALYSIS_DIR == ROOT / "results" / "error_analysis"
    assert audit_corpora.LOGS_DIR == ROOT / "logs"
    assert train_model.MODEL_DIR == ROOT / "model"
    assert application.LOG_PATH == ROOT / "logs" / "analysis_log.csv"
    assert application.LOG_PATH.parent == audit_corpora.LOGS_DIR
    assert application.LOG_PATH.parent not in (
        train_model.RESULTS_DIR,
        train_model.PREDICTIONS_DIR,
        train_model.ERROR_ANALYSIS_DIR,
        train_model.MODEL_DIR,
    )


@pytest.mark.parametrize(
    "filename", ["challenge_predictions.csv", "evaluation_predictions.csv"]
)
def test_evaluator_creates_prediction_subfolder_from_an_unrelated_directory(
    tmp_path, monkeypatch, filename
):
    destination = tmp_path / "results" / "predictions"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluate_model, "PREDICTIONS_DIR", destination)
    monkeypatch.setattr(
        evaluate_model,
        "parse_args",
        lambda: Namespace(
            dataset="challenge_dataset.csv",
            model="phishing_pipeline.joblib",
            output=filename,
        ),
    )
    evaluate_model.main()
    predictions = pd.read_csv(destination / filename)
    assert len(predictions) == len(pd.read_csv(ROOT / "data" / "challenge_dataset.csv"))
    assert {"true_label", "predicted_label", "error_type"}.issubset(predictions.columns)
    assert not (tmp_path / "results" / filename).exists()
    assert not (tmp_path / "logs").exists()


def test_audit_writer_uses_logs_and_preserves_upload_history(tmp_path, monkeypatch):
    log_directory = tmp_path / "logs"
    comparison_directory = tmp_path / "results" / "comparison"
    log_directory.mkdir()
    upload_log = log_directory / "analysis_log.csv"
    existing_history = b"timestamp,filename\r\n2026-01-01T00:00:00,sample.eml\r\n"
    upload_log.write_bytes(existing_history)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit_corpora, "LOGS_DIR", log_directory)
    monkeypatch.setattr(audit_corpora, "COMPARISON_DIR", comparison_directory)
    monkeypatch.setattr(
        audit_corpora,
        "corpus_audit",
        lambda name: {
            "dataset": name,
            "rows": 2,
            "duplicate_rows": 0,
            "rows_after_deduplication": 2,
            "distinct_sentences": 2,
            "components": [],
        },
    )
    monkeypatch.setattr(audit_corpora, "cue_cooccurrence", lambda: {"rows": 2})
    monkeypatch.setattr(
        audit_corpora,
        "preprocessing_control",
        lambda: pd.DataFrame([{"model": "control", "challenge_f1": 0.0}]),
    )
    audit_corpora.main()
    assert (log_directory / "corpus_audit.json").is_file()
    assert (comparison_directory / "preprocessing_control.csv").is_file()
    assert not (tmp_path / "results" / "corpus_audit.json").exists()
    assert not (tmp_path / "audits").exists()
    assert upload_log.read_bytes() == existing_history


def test_before_after_export_pairs_the_same_datasets_and_bands():
    before = pd.DataFrame(
        [{
            "dataset": "challenge_dataset.csv",
            "band": "review_or_above",
            "phishing_rows": 5,
            "cut": 0.45,
            "true_positives": 0,
            "false_positives": 0,
            "phishing_recall": 0.0,
            "flagged_precision": 0.0,
        }]
    )
    after = pd.DataFrame(
        [{
            "split": "challenge",
            "band": "review_or_above",
            "phishing_rows": 5,
            "cut": 0.3162,
            "true_positives": 3,
            "false_positives": 0,
            "phishing_recall": 0.6,
            "flagged_precision": 1.0,
        }]
    )
    comparison = before_after_table(before, after)
    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert row["before_phishing_recall"] == 0.0
    assert row["after_phishing_recall"] == 0.6
    assert row["before_cut"] == 0.45
    assert row["after_cut"] == 0.3162
    with pytest.raises(ValueError, match="same datasets and bands"):
        before_after_table(before, after.assign(band="high"))
    with pytest.raises(ValueError, match="phishing-row counts"):
        before_after_table(before, after.assign(phishing_rows=6))
