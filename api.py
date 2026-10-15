"""Optional Flask REST API (/health, /predict, /batch-score).

Run (development):  python api.py
Run (production):   gunicorn -w 2 -b 127.0.0.1:5000 api:app     (pip install gunicorn)
Every request except /health needs the header  X-API-Key: <API_KEY from your .env>.
"""
import hmac

import pandas as pd
from flask import Flask, jsonify, request

from src import config
from src.m2_features import FEATURE_COLUMNS
from src.m4_scoring import ChurnScoringService

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
_service: ChurnScoringService | None = None


def service() -> ChurnScoringService:
    global _service
    if _service is None:
        _service = ChurnScoringService()
    return _service


def _authorised() -> bool:
    key = config.API_KEY
    supplied = request.headers.get("X-API-Key", "")
    return bool(key) and hmac.compare_digest(key, supplied)


@app.get("/health")
def health():
    try:
        return jsonify(status="ok", model=service().version)
    except Exception as exc:
        return jsonify(status="error", detail=str(exc)), 503


def _score(records: list[dict]):
    df = pd.DataFrame(records)
    for i, rec in enumerate(records):
        if "customer_id" not in rec:
            df.loc[i, "customer_id"] = f"api-{i}"
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        return jsonify(error=f"missing feature(s): {missing}"), 400
    scored = service().score_customers(df, with_drivers=True)
    return jsonify(results=scored.drop(columns=["model_version"]).to_dict(orient="records"),
                   model_version=service().version)


@app.post("/predict")
def predict():
    if not _authorised():
        return jsonify(error="unauthorised"), 401
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="JSON object with the feature values expected"), 400
    return _score([body])


@app.post("/batch-score")
def batch_score():
    if not _authorised():
        return jsonify(error="unauthorised"), 401
    body = request.get_json(silent=True)
    if not isinstance(body, list) or not body:
        return jsonify(error="non-empty JSON list of customer objects expected"), 400
    return _score(body)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
