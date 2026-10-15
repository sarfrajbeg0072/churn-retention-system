import joblib
import numpy as np
import pytest

from src.m2_features import FEATURE_COLUMNS
from src.m4_scoring import ChurnScoringService


def test_tc05_risk_band_boundaries(trained_bundle):
    path, _, _ = trained_bundle
    s = ChurnScoringService(path)
    assert s.classify_risk(0.70)[0] == "High"
    assert s.classify_risk(0.6999)[0] == "Medium"
    assert s.classify_risk(0.35)[0] == "Medium"
    assert s.classify_risk(0.3499)[0] == "Low"


def test_tc08_model_reload_gives_identical_predictions(trained_bundle):
    path, _, feats = trained_bundle
    in_memory = joblib.load(path)["model"].predict_proba(feats[FEATURE_COLUMNS])[:, 1]
    reloaded = ChurnScoringService(path).predict_proba(feats)
    assert np.allclose(in_memory, reloaded)


def test_score_output_shape_and_drivers(trained_bundle):
    path, _, feats = trained_bundle
    out = ChurnScoringService(path).score_customers(feats.head(25))
    assert len(out) == 25
    assert out.churn_probability.between(0, 1).all()
    assert set(out.risk_category) <= {"High", "Medium", "Low"}
    assert out.primary_churn_driver.notna().all()


def test_missing_feature_columns_raise(trained_bundle):
    path, _, feats = trained_bundle
    with pytest.raises(ValueError):
        ChurnScoringService(path).score_customers(feats.drop(columns=["recency_days"]))


def test_trained_model_beats_random_and_saves_metrics(trained_bundle):
    _, metrics, _ = trained_bundle
    assert metrics["models"][metrics["best_model"]]["test"]["roc_auc"] > 0.7
    assert "baseline_rule_30d" in metrics and metrics["confusion_matrix"]
