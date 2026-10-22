from __future__ import annotations

import pandas as pd

from feature_extraction import (
    ParsedEmail,
    build_baseline_numeric_feature_frame,
    build_numeric_feature_frame,
    compute_handcrafted_features,
    extract_urls_from_text,
    get_header_anomalies,
    phishing_cues,
)
from train_model import MODEL_CONFIGS, build_pipeline


def test_url_extraction_strips_terminal_punctuation():
    text = "Review https://example.com/login, then visit www.example.org."
    assert extract_urls_from_text(text) == [
        "https://example.com/login",
        "www.example.org",
    ]


def test_baseline_preserves_original_empty_url_training_behavior():
    series = pd.Series(["Open http://203.0.113.10/login immediately."])
    baseline = build_baseline_numeric_feature_frame(series)
    enhanced = build_numeric_feature_frame(series)
    assert baseline.loc[0, "url_count"] == 0
    assert baseline.loc[0, "ip_url_count"] == 0
    assert enhanced.loc[0, "url_count"] == 1
    assert enhanced.loc[0, "ip_url_count"] == 1


def test_enhanced_url_features_detect_spoof_tld_and_complexity():
    text = (
        "Sign in at "
        "https://micr0soft-verify-account-secure.sub.accounts.example.top/verify."
    )
    features = compute_handcrafted_features(text)
    assert features["suspicious_tld_count"] == 1
    assert features["domain_similarity_count"] == 1
    assert features["complex_url_count"] == 1
    assert features["suspicious_domain_count"] == 1
    neutral = compute_handcrafted_features(
        "View it at https://files.example.com/view/4832 when you have a moment."
    )
    assert neutral["suspicious_domain_count"] == 0


def test_enhanced_linguistic_features_cover_required_categories():
    text = (
        "IT Security asks you to re-enter your credentials. Please review the "
        "payroll reimbursement today."
    )
    features = compute_handcrafted_features(text)
    assert features["credential_request_count"] >= 1
    assert features["financial_pressure_count"] >= 1
    assert features["impersonation_signal_count"] >= 1
    assert features["polite_action_count"] >= 1


def test_reassurance_feature_recognizes_legitimate_language():
    features = compute_handcrafted_features(
        "The issue is being handled by IT. No user action is required."
    )
    assert features["reassurance_count"] >= 1


def test_cues_are_detailed_unique_and_include_new_categories():
    text = (
        "IT Security: please review the payroll page and re-enter credentials at "
        "https://micr0soft-login-check.top now."
    )
    cues = phishing_cues(text)
    assert len(cues) == len(set(cues))
    assert any(cue.startswith("Financial pressure:") for cue in cues)
    assert any(cue.startswith("Authority impersonation:") for cue in cues)
    assert any(cue.startswith("Credential request:") for cue in cues)
    assert any(cue.startswith("Possible brand lookalike:") for cue in cues)


def test_header_anomalies_detect_free_sender_and_domain_mismatches():
    headers = ParsedEmail(
        from_addr="Security <alert@gmail.com>",
        reply_to="help@secure-check.example",
        return_path="bounce@mailer.example",
    )
    anomalies = get_header_anomalies(headers)
    assert len(anomalies) == 3
    assert any("free email provider" in item for item in anomalies)
    assert any("Reply-To" in item for item in anomalies)
    assert any("Return-Path" in item for item in anomalies)


def test_all_model_configurations_build_and_fit():
    frame = pd.DataFrame(
        {
                "text": [
                    "Urgent verify your account password now",
                    "Meeting agenda and course notes for tomorrow",
                    "Sign in to your account at http://203.0.113.10/login",
                    "No user action is required for the meeting",
                    "Payroll account identity check today",
                    "Course timetable and course reading attached",
                ]
        }
    )
    labels = [1, 0, 1, 0, 1, 0]
    for name in MODEL_CONFIGS:
        pipeline = build_pipeline(name)
        pipeline.fit(frame, labels)
        predictions = pipeline.predict(frame)
        assert len(predictions) == len(frame)
        assert pipeline.predict_proba(frame).shape == (len(frame), 2)
