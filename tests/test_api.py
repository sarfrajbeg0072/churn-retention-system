import pytest

import api
from src import config
from src.m2_features import FEATURE_COLUMNS
from src.m4_scoring import ChurnScoringService


@pytest.fixture()
def client(trained_bundle, monkeypatch):
    path, _, feats = trained_bundle
    monkeypatch.setattr(config, "API_KEY", "secret-key")
    monkeypatch.setattr(api, "_service", ChurnScoringService(path))
    return api.app.test_client(), feats


def test_health_open(client):
    c, _ = client
    assert c.get("/health").get_json()["status"] == "ok"


def test_predict_requires_key(client):
    c, feats = client
    body = feats.iloc[0][FEATURE_COLUMNS].astype(float).to_dict()
    assert c.post("/predict", json=body).status_code == 401
    r = c.post("/predict", json=body, headers={"X-API-Key": "secret-key"})
    assert r.status_code == 200
    assert 0 <= r.get_json()["results"][0]["churn_probability"] <= 1


def test_predict_missing_features(client):
    c, _ = client
    assert c.post("/predict", json={"recency_days": 1}, headers={"X-API-Key": "secret-key"}).status_code == 400
