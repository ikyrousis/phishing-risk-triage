from __future__ import annotations

import io

import pytest

from feature_extraction import HIGH_RISK_THRESHOLD, REVIEW_RISK_THRESHOLD

from app import (
    app,
    calculate_risk,
    calculate_risk_breakdown,
    label_from_risk,
    recommendation_for_band,
)


def test_risk_breakdown_is_bounded_and_explained():
    features = {
        "ip_url_count": 1,
        "credential_request_count": 1,
        "header_anomaly_count": 2,
        "reassurance_count": 0,
    }
    breakdown = calculate_risk_breakdown(0.8, features)
    assert 0 <= breakdown["final_score"] <= 1
    assert breakdown["model_component"] == pytest.approx(0.56)
    assert breakdown["heuristic_component"] > 0
    assert len(breakdown["contributions"]) == 3


def test_compatibility_risk_wrapper_does_not_double_count_cue_lists():
    features = {"header_anomaly_count": 1, "credential_request_count": 1}
    first = calculate_risk(0.5, ["one cue"], ["one anomaly"], features)
    second = calculate_risk(
        0.5,
        ["one cue", "duplicated cue", "another cue"],
        ["one anomaly", "duplicated anomaly"],
        features,
    )
    assert first == second


@pytest.mark.parametrize(
    ("risk", "expected"),
    [
        (REVIEW_RISK_THRESHOLD / 2, "Likely legitimate"),
        (REVIEW_RISK_THRESHOLD + 0.01, "Suspicious / needs review"),
        (HIGH_RISK_THRESHOLD + 0.01, "Likely phishing"),
    ],
)
def test_risk_labels(risk, expected):
    assert label_from_risk(risk) == expected


def test_high_band_is_reachable_from_model_confidence_alone():
    """A maximally confident prediction must not be capped below the high band."""
    breakdown = calculate_risk_breakdown(1.0, {})
    assert breakdown["final_score"] >= HIGH_RISK_THRESHOLD
    assert label_from_risk(breakdown["final_score"]) == "Likely phishing"


def test_recommendations_are_band_specific():
    assert "Do not follow links" in recommendation_for_band("high")
    assert "verify the sender" in recommendation_for_band("medium")
    assert "does not prove" in recommendation_for_band("low")


def test_get_route_renders_upload_form():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Analyze message" in response.data


def test_invalid_extension_returns_interpretable_error():
    client = app.test_client()
    response = client.post(
        "/",
        data={"file": (io.BytesIO(b"plain text"), "message.txt")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert b"Unsupported file type" in response.data


def test_phishing_html_upload_returns_breakdown_and_removes_temporary_file():
    client = app.test_client()
    payload = b"""
    <html><body><p>IT Security: verify your password immediately.</p>
    <a href="http://203.0.113.10/login">Sign in</a></body></html>
    """
    response = client.post(
        "/",
        data={"file": (io.BytesIO(payload), "test-phishing.html")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"Recommended response" in response.data
    assert b"Model contribution" in response.data
    assert b"Credential request" in response.data
    assert not (app.root_path and (__import__("pathlib").Path(app.root_path) / "uploads" / "test-phishing.html").exists())
