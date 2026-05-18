#!/usr/bin/env python3
"""
local_scoring_service.py - Flask HTTP server wrapping the ReCiter scoring Lambda handler.

This service mimics the AWS Lambda Runtime Interface Emulator (RIE) endpoint,
allowing ReCiter Java to invoke local scoring without any AWS dependency.

Environment Variables:
    SCORING_APP_DIR:  Path to ReCiter---Scoring/app directory (default: ~/Dropbox/GitHub/ReCiter---Scoring/app)
    SCORING_DATA_DIR: Path where Java writes scoring input JSON files (default: /tmp/reciter-scoring-data)
    SCORING_PORT:     Port to listen on (default: 9000)

Endpoints:
    GET  /health                                          -> Health check
    POST /2015-03-31/functions/function/invocations       -> Score articles (Lambda-compatible)
"""
import json
import logging
import os
import sys
import time

# Force single-threaded joblib before any imports that might use it
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

# Configuration from environment
SCORING_APP_DIR = os.environ.get(
    "SCORING_APP_DIR",
    # This harness lives inside the ReCiter---Scoring repo: app/ is two levels up.
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "app")
)
SCORING_DATA_DIR = os.environ.get(
    "SCORING_DATA_DIR",
    "/tmp/reciter-scoring-data"
)
SCORING_PORT = int(os.environ.get("SCORING_PORT", "9000"))

# Add scoring app directory to path so we can import main, verify_setup, etc.
if SCORING_APP_DIR not in sys.path:
    sys.path.insert(0, SCORING_APP_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("local_scoring_service")

# Import the Lambda handler and monkey-patch the file read path
try:
    import verify_setup

    _original_read = verify_setup.read_json_file

    def _patched_read(scoring_data_file, timeout=60):
        """Read scoring data from SCORING_DATA_DIR instead of /var/task/data."""
        file_path = os.path.join(SCORING_DATA_DIR, scoring_data_file)
        logger.info("Reading scoring data from: %s", file_path)
        start_time = time.time()
        while time.time() - start_time < timeout:
            if os.path.exists(file_path):
                with open(file_path) as f:
                    return json.load(f)
            time.sleep(0.5)
        raise FileNotFoundError(f"File not found after {timeout}s: {file_path}")

    verify_setup.read_json_file = _patched_read
    logger.info("Patched verify_setup.read_json_file to use SCORING_DATA_DIR=%s", SCORING_DATA_DIR)

    from main import lambda_handler as _real_handler, cached_models
    _lambda_handler = _real_handler
    _models_loaded = cached_models is not None and len(cached_models) > 0
    logger.info("Lambda handler imported successfully. Models loaded: %s", _models_loaded)

except Exception as e:
    logger.warning("Could not import scoring modules: %s", e)
    logger.warning("Service will start but scoring will fail until modules are available")
    _lambda_handler = None
    _models_loaded = False
    cached_models = None

# Flask app
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": _models_loaded,
        "scoring_data_dir": SCORING_DATA_DIR,
        "scoring_app_dir": SCORING_APP_DIR,
    })


@app.route("/2015-03-31/functions/function/invocations", methods=["POST"])
def invoke():
    """
    Lambda-compatible invocation endpoint.

    Accepts the same JSON payload as the AWS Lambda Runtime Interface:
    {
        "modelName": "feedback" | "identity",
        "scoringDataFile": "<uid>-feedbackIdentityScoringInput.json",
        "useS3Bucket": "true" | "false",
        "bucket_name": "feedbackscoring"
    }

    Forces useS3Bucket to "false" since we read from local disk.
    """
    if _lambda_handler is None:
        return jsonify({
            "predictionScores": "",
            "returnCode": 1,
            "error": "Lambda handler not loaded"
        }), 500

    try:
        event = request.get_json(force=True)
    except Exception as e:
        return jsonify({
            "predictionScores": "",
            "returnCode": 1,
            "error": f"Invalid JSON payload: {e}"
        }), 400

    # Force local file reads (never use S3 in standalone mode)
    event["useS3Bucket"] = "false"

    logger.info(
        "Invoking lambda_handler: modelName=%s, scoringDataFile=%s",
        event.get("modelName", "?"),
        event.get("scoringDataFile", "?"),
    )

    try:
        result = _lambda_handler(event, None)
        return jsonify(result)
    except Exception as e:
        logger.exception("Lambda handler failed: %s", e)
        return jsonify({
            "predictionScores": "",
            "returnCode": 1,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    logger.info("Starting local scoring service on port %d", SCORING_PORT)
    logger.info("SCORING_APP_DIR=%s", SCORING_APP_DIR)
    logger.info("SCORING_DATA_DIR=%s", SCORING_DATA_DIR)
    app.run(host="0.0.0.0", port=SCORING_PORT, debug=False)
