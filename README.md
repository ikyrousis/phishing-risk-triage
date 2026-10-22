# Explainable Phishing Detection with Calibrated Risk Triage

An explainable phishing detection system for HTML pages, Word documents and
email files. Each upload passes through three stages: a feature extractor and
rule-based cue layer that surface observable evidence in the file, a trained
linear SVM selected from five candidate classifiers against a frozen selection
corpus, and calibrated thresholds that sort the verdict into low, medium and
high risk bands. The Flask interface reports the band, the score and the
specific cues that fired, so every decision traces back to concrete evidence
rather than an opaque model output.

## Prerequisites

- Python 3.11 or later
- A Chromium-based browser (Edge, Chrome or Chromium) for the screenshot step
- PowerShell on Windows, or a POSIX terminal on macOS and Linux

## Setup

Create and activate a virtual environment, upgrade pip and install the project
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Serialized models are version-sensitive, so record the active versions:

```bash
python -c "import sklearn, pandas; print(sklearn.__version__, pandas.__version__)"
```

## Commands in run order

Run every command from the project directory, with the environment active. Training
precedes the checks so the upload test also works when `model/` starts empty.

```bash
python scripts/build_generalisation_corpus.py
python train_model.py
python -m pytest -q -W error
python scripts/analyze_results.py
python scripts/compare_supplied_detector.py
python scripts/audit_corpora.py
python evaluate_model.py --dataset challenge_dataset.csv --model phishing_pipeline.joblib --output challenge_predictions.csv
python scripts/capture_screenshots.py
python app.py
```

Operational notes:

- `scripts/build_generalisation_corpus.py` regenerates and gates the selection
  corpus; writes `data/generalisation_corpus.csv`.
- `train_model.py` trains five arms, selects and refits the deployment arm, and
  derives band cuts; writes six model binaries under `model/`,
  `model/metrics.json`, `model/threshold_calibration.csv`,
  `results/predictions/comparative_predictions.csv` and
  `results/error_analysis/selected_model_errors.csv`. It uses
  `data/dataset_homework.csv` by default; set `PHISHING_DATASET` to another
  filename under `data/` to change it. It overwrites these outputs on each run
  and stops if the derived band cuts drift from the constants in
  `feature_extraction.py` or if the selection corpus reproduces a training or
  evaluation message.
- `python -m pytest -q -W error` runs 24 checks with warnings treated as errors;
  outputs terminal test results.
- `scripts/analyze_results.py` compares arms and derives error cases; writes
  `results/comparison/{model_comparison,end_to_end_bands,risk_scoring_impact}.csv`,
  `results/predictions/{end_to_end_rows,adversarial_predictions}.csv` and
  `results/error_analysis/error_analysis.json`. It reads errors off the deployed
  model's predictions and stops if a selected error row has no recorded
  interpretation or if the adversarial set yields fewer than two errors of
  either type.
- `scripts/compare_supplied_detector.py` reconstructs the supplied detector and
  joins before/after results; writes
  `results/comparison/{supplied_detector,before_after_comparison}.csv` and
  `results/predictions/supplied_detector_predictions.csv`. It requires
  `results/comparison/end_to_end_bands.csv` from the analysis stage and stops if
  its reconstruction misses the supplied lab's recorded held-out accuracy by
  more than two rows. Its before/after CSV is a join of the existing
  whole-detector measurements, with each detector's cut included.
- `scripts/audit_corpora.py` audits corpora, cue co-occurrence and
  preprocessing; writes `logs/corpus_audit.json` and
  `results/comparison/preprocessing_control.csv`. The JSON records the corpus
  and cue-overlap checks and is regenerated on each run.
- `evaluate_model.py` evaluates one saved model on one dataset; the command
  above writes `results/predictions/challenge_predictions.csv`. It accepts any
  dataset under `data/` and any model under `model/`, by filename or absolute
  path. Its `--output` accepts a filename and writes directly into
  `results/predictions/`; the default filename is `evaluation_predictions.csv`.
- `scripts/capture_screenshots.py` uploads one sample per risk band through the
  app and captures each page at 1440 px; overwrites
  `results/screenshot-high-risk.png`, `results/screenshot-medium-risk.png` and
  `results/screenshot-low-risk.png`. It takes the browser from
  `PHISHING_BROWSER` if set, otherwise from the PATH (`msedge`, `chrome`,
  `google-chrome`, `chromium`), otherwise from the standard Windows or macOS
  install locations. It uses the Flask test client to render genuine upload
  results and then headless Chromium to capture them; a separately running
  `app.py` server is not required. Manual capture is equivalent: upload
  `phishing_example.eml`, `phishing_1.html` and
  `legitimate_internal_notice.eml` at `http://127.0.0.1:5000` and capture each
  result page at a viewport of at least 1280 px.
- `python app.py` starts optional manual inspection on `http://127.0.0.1:5000`
  and appends `logs/analysis_log.csv` on each upload. It keeps the server
  running until Ctrl+C.
- `logs/analysis_log.csv` retains the verification uploads and is included with the
  generated artifacts. Tests, screenshots and subsequent manual uploads append
  entries rather than clearing the existing history.
- The scripts create their output directories automatically and overwrite only
  their named analytical outputs. Copy the project before a run if an earlier
  snapshot is needed. The numbered folders in the separate submission package
  contain copies, not additional runtime destinations.
- A repeated `train_model.py` run under unchanged Python and dependency versions
  reproduces every metric, split index, prediction and probability in
  `model/metrics.json`; only `generated_at` changes. After copying
  `model/metrics.json` to `metrics-first.json` before the second run, compare
  the two files with the command below; it prints `True` when every metric and
  split index matches.

  ```bash
  python -c "import json; a=json.load(open('metrics-first.json')); b=json.load(open('model/metrics.json')); print('identical:', a['models']==b['models'] and a['split']==b['split'])"
  ```

  The calibration-table reference is project-relative. Different dependency
  or browser versions can change probabilities or screenshot pixels; the numerical
  verification used scikit-learn 1.9.1 and pandas 3.0.6.

## Repository map

- `feature_extraction.py`: parsing, supplied and enhanced features, cue rules, risk scoring
- `train_model.py`: corpus hygiene, five arms, model selection, calibration
- `evaluate_model.py`: evaluates one saved model on one dataset
- `app.py`, `templates/upload.html`: the application
- `scripts/build_generalisation_corpus.py`: generates and gates the selection corpus
- `scripts/analyze_results.py`: comparison, band, risk-ablation and error analysis
- `scripts/compare_supplied_detector.py`: supplied-lab reconstruction and before/after view
- `scripts/audit_corpora.py`: corpus audits, cue co-occurrence and preprocessing control
- `scripts/capture_screenshots.py`: regenerates screenshots from application responses
- `tests/`: the automated checks
- `data/`: the supplied datasets, unchanged, plus the generated selection corpus and the authored adversarial set
- `sample_inputs/`: supplied sample files, unchanged
- `model/`: trained model binaries, training metadata and the threshold-calibration table
- `results/`: generated comparisons, predictions, error analysis and screenshots
- `logs/`: operational upload history and structured corpus checks