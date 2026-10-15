"""Explainability: SHAP values (with a safe fallback) and global importance."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.m2_features import FEATURE_LABELS


def _transform(model, X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Apply the fitted preprocessing + scaling steps (not SMOTE / classifier)."""
    Xp = model.named_steps["prep"].transform(X[features])
    Xs = model.named_steps["scale"].transform(Xp)
    return pd.DataFrame(Xs, columns=features, index=X.index)


def _occlusion(model, X: pd.DataFrame, features: list[str], reference: pd.DataFrame) -> pd.DataFrame:
    """Model-agnostic fallback: change in P(churn) when a feature is set to its median."""
    base = model.predict_proba(X[features])[:, 1]
    medians = reference[features].median()
    out = {}
    for f in features:
        X2 = X[features].copy()
        X2[f] = medians[f]
        out[f] = base - model.predict_proba(X2)[:, 1]
    return pd.DataFrame(out, index=X.index)


def contributions(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Per-customer contribution of every feature to the churn score (rows x features).

    Positive value = pushes the customer towards churn.
    """
    model, features = bundle["model"], bundle["features"]
    reference = bundle.get("background")
    if reference is None:
        reference = X
    clf = model.named_steps["clf"]
    kind = type(clf).__name__
    try:
        import shap

        Xt = _transform(model, X, features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind in ("RandomForestClassifier", "DecisionTreeClassifier", "XGBClassifier"):
                values = shap.TreeExplainer(clf).shap_values(Xt, check_additivity=False)
            elif kind == "LogisticRegression":
                bg = _transform(model, reference, features)
                values = shap.LinearExplainer(clf, bg).shap_values(Xt)
            else:
                raise NotImplementedError(kind)
        if isinstance(values, list):  # older shap: list per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:  # newer shap: (rows, features, classes)
            values = values[:, :, 1]
        return pd.DataFrame(values, columns=features, index=X.index)
    except Exception:
        return _occlusion(model, X, features, reference)


def primary_drivers(contrib: pd.DataFrame) -> pd.Series:
    """Human-readable name of the feature pushing each customer most towards churn."""
    top = contrib.idxmax(axis=1)
    return top.map(lambda f: FEATURE_LABELS.get(f, f)).rename("primary_churn_driver")


def global_importance(bundle: dict, X: pd.DataFrame, max_rows: int = 1000) -> dict[str, float]:
    """Mean |contribution| per feature, sorted descending."""
    sample = X.sample(min(len(X), max_rows), random_state=42) if len(X) > max_rows else X
    imp = contributions(bundle, sample).abs().mean().sort_values(ascending=False)
    return {k: float(v) for k, v in imp.items()}
