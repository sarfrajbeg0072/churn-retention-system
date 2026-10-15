"""Module 4 - Prediction & scoring service (risk bands + retention actions)."""
from __future__ import annotations

import joblib
import pandas as pd
import sklearn

from src import config, explain
from src.m2_features import FEATURE_COLUMNS


class ChurnScoringService:
    """Loads the persisted pipeline and converts probabilities into risk bands.

    SECURITY: ``joblib`` files are pickles - only load files from the trusted
    local ``models/`` directory, never from user uploads.
    """

    HIGH = config.HIGH_THRESHOLD
    MEDIUM = config.MEDIUM_THRESHOLD

    ACTIONS = {
        "High": "Offer 20% discount coupon and priority support call",
        "Medium": "Send automated personalised product-recommendation email",
        "Low": "Maintain standard marketing communication",
    }

    def __init__(self, model_path=None, bundle: dict | None = None):
        self.bundle = bundle if bundle is not None else joblib.load(model_path or config.MODEL_PATH)
        self.model = self.bundle["model"]
        self.feature_names = self.bundle["features"]
        self.model_name = self.bundle["name"]
        self.version = self.bundle.get("version", self.model_name)

    @staticmethod
    def load_if_compatible(model_path=None):
        """Return a service, or None if the file is missing / built with another scikit-learn."""
        try:
            bundle = joblib.load(model_path or config.MODEL_PATH)
        except Exception:
            return None
        if bundle.get("sklearn_version") != sklearn.__version__:
            return None
        return ChurnScoringService(bundle=bundle)

    def classify_risk(self, proba: float) -> tuple[str, str]:
        if proba >= self.HIGH:
            band = "High"
        elif proba >= self.MEDIUM:
            band = "Medium"
        else:
            band = "Low"
        return band, self.ACTIONS[band]

    def predict_proba(self, feature_df: pd.DataFrame):
        X = feature_df[self.feature_names]  # enforce training column order
        return self.model.predict_proba(X)[:, 1]

    def contributions(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        return explain.contributions(self.bundle, feature_df[self.feature_names])

    def score_customers(self, feature_df: pd.DataFrame, with_drivers: bool = True) -> pd.DataFrame:
        missing = [c for c in FEATURE_COLUMNS if c not in feature_df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        feature_df = feature_df.reset_index(drop=True)
        probs = self.predict_proba(feature_df)
        rows = []
        for cid, p in zip(feature_df["customer_id"], probs):
            band, action = self.classify_risk(float(p))
            rows.append({"customer_id": cid, "churn_probability": round(float(p), 4),
                         "risk_category": band, "recommended_action": action})
        out = pd.DataFrame(rows)
        if with_drivers and len(out):
            contrib = self.contributions(feature_df)
            out["primary_churn_driver"] = explain.primary_drivers(contrib).to_numpy()
        else:
            out["primary_churn_driver"] = "n/a"
        out["model_version"] = self.version
        return out
