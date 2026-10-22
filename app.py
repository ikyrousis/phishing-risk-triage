from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from feature_extraction import (
    HIGH_RISK_THRESHOLD,
    REVIEW_RISK_THRESHOLD,
    calculate_risk_breakdown,
    compute_handcrafted_features,
    extract_content,
    get_header_anomalies,
    log_result,
    phishing_cues,
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model" / "phishing_pipeline.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
LOG_PATH = BASE_DIR / "logs" / "analysis_log.csv"
ALLOWED_EXTENSIONS = {"html", "docx", "eml"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
UPLOAD_FOLDER.mkdir(exist_ok=True)

_PIPELINE = None
_METADATA: dict | None = None


def get_pipeline():
    """Load the trained pipeline on first use, so importing the app needs no artifacts."""
    global _PIPELINE
    if _PIPELINE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Run `python train_model.py` first."
            )
        _PIPELINE = joblib.load(MODEL_PATH)
    return _PIPELINE


def get_metadata() -> dict:
    """Read the training metadata once, tolerating an untrained checkout."""
    global _METADATA
    if _METADATA is None:
        _METADATA = (
            json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            if METRICS_PATH.exists()
            else {}
        )
    return _METADATA


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_risk(
    model_probability: float,
    cues: list[str],
    header_anomalies: list[str],
    feature_map: dict[str, float],
) -> float:
    """Compatibility wrapper; scoring uses features to avoid duplicate cue counting."""
    del cues, header_anomalies
    return calculate_risk_breakdown(model_probability, feature_map)["final_score"]


def label_from_risk(risk: float) -> str:
    if risk >= HIGH_RISK_THRESHOLD:
        return "Likely phishing"
    if risk >= REVIEW_RISK_THRESHOLD:
        return "Suspicious / needs review"
    return "Likely legitimate"


def risk_band(risk: float) -> str:
    if risk >= HIGH_RISK_THRESHOLD:
        return "high"
    if risk >= REVIEW_RISK_THRESHOLD:
        return "medium"
    return "low"


def recommendation_for_band(band: str) -> str:
    if band == "high":
        return (
            "Do not follow links or provide information. Verify the request through "
            "a known contact method and report the message to the security team."
        )
    if band == "medium":
        return (
            "Pause and verify the sender, destination domain, and requested action "
            "before interacting with the message."
        )
    return (
        "No strong phishing evidence was found. This does not prove the message is "
        "safe; unexpected requests should still be verified through an official channel."
    )


def base_context() -> dict:
    return {
        "result": None,
        "error_message": None,
        "risk_score": None,
        "model_probability": None,
        "cues": [],
        "cue_count": 0,
        "header_anomalies": [],
        "header_anomaly_count": 0,
        "features": None,
        "filename": None,
        "risk_percent": None,
        "risk_band": None,
        "recommendation": None,
        "risk_breakdown": None,
        "model_name": get_metadata().get(
            "selected_deployment_model", "enhanced phishing classifier"
        ).replace("_", " "),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    context = base_context()
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            context["error_message"] = "Select an HTML, DOCX, or EML file to analyze."
            return render_template("upload.html", **context), 400
        if not allowed_file(file.filename):
            context["error_message"] = (
                "Unsupported file type. Select an .html, .docx, or .eml file."
            )
            return render_template("upload.html", **context), 400

        try:
            estimator = get_pipeline()
        except FileNotFoundError:
            context["error_message"] = (
                "No trained model is available. Run `python train_model.py` before "
                "analyzing files."
            )
            return render_template("upload.html", **context), 503

        filename = secure_filename(file.filename)
        path = UPLOAD_FOLDER / filename
        file.save(path)

        try:
            parsed = extract_content(str(path))
            text = parsed["text"]
            urls = parsed["urls"]
            headers = parsed["headers"]

            model_probability = float(
                estimator.predict_proba(pd.DataFrame({"text": [text]}))[0][1]
            )
            feature_map = compute_handcrafted_features(text, urls, headers)
            cues = phishing_cues(text, urls, headers)
            header_anomalies = get_header_anomalies(headers)
            breakdown = calculate_risk_breakdown(model_probability, feature_map)
            risk_score = breakdown["final_score"]
            band = risk_band(risk_score)
            result = label_from_risk(risk_score)

            log_result(
                str(LOG_PATH),
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "filename": filename,
                    "prediction": result,
                    "risk_score": risk_score,
                    "model_probability": round(model_probability, 4),
                    "cue_count": len(cues),
                    "header_anomaly_count": len(header_anomalies),
                    "triggered_cues": "; ".join(cues),
                },
            )
            context.update(
                {
                    "filename": filename,
                    "result": result,
                    "risk_score": risk_score,
                    "risk_percent": int(round(risk_score * 100)),
                    "risk_band": band,
                    "model_probability": round(model_probability, 4),
                    "cues": cues,
                    "cue_count": len(cues),
                    "header_anomalies": header_anomalies,
                    "header_anomaly_count": len(header_anomalies),
                    "features": feature_map,
                    "recommendation": recommendation_for_band(band),
                    "risk_breakdown": breakdown,
                }
            )
        except Exception:
            app.logger.exception("Analysis failed for %s", filename)
            context["error_message"] = (
                "The file could not be analyzed. Confirm that it is a valid, "
                "non-corrupted HTML, DOCX, or EML document."
            )
            return render_template("upload.html", **context), 422
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                app.logger.warning("Could not remove temporary upload %s", path)

    return render_template("upload.html", **context)


@app.errorhandler(413)
def file_too_large(_error):
    context = base_context()
    context["error_message"] = "The file is too large. The maximum upload size is 5 MB."
    return render_template("upload.html", **context), 413


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
