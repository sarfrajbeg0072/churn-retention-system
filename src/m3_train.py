"""Module 3 - Model training & evaluation subsystem.

Preprocessing, scaling, SMOTE and the classifier live in ONE imblearn Pipeline;
GridSearchCV tunes it with 5-fold stratified CV, so SMOTE only ever touches the
training part of each fold. Five models are compared (Logistic Regression,
Decision Tree, Random Forest, XGBoost, SVM) and the best is saved with joblib.

Run:  python -m src.m3_train [--fast]
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # NOT sklearn.pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config, explain
from src.m1_preprocess import DataIngestionPreprocessor
from src.m2_features import FEATURE_COLUMNS

try:
    from xgboost import XGBClassifier
except ImportError:  # XGBoost is optional
    XGBClassifier = None


def build_candidates(random_state: int = 42, fast: bool = False) -> dict:
    """Return {name: (pipeline, param_grid)} for every candidate model."""

    def pipe(clf):
        return Pipeline([
            ("prep", DataIngestionPreprocessor()),
            ("scale", StandardScaler()),
            ("smote", SMOTE(random_state=random_state)),  # training folds only
            ("clf", clf),
        ])

    cands = {
        "logistic_regression": (
            pipe(LogisticRegression(max_iter=2000)),
            {"clf__C": [0.1, 1.0, 10.0]}),
        "decision_tree": (
            pipe(DecisionTreeClassifier(random_state=random_state)),
            {"clf__max_depth": [4, 6, 8], "clf__min_samples_leaf": [10, 30]}),
        "random_forest": (
            pipe(RandomForestClassifier(random_state=random_state, n_jobs=-1)),
            {"clf__n_estimators": [100] if fast else [100, 200],
             "clf__max_depth": [8] if fast else [8, 14],
             "clf__min_samples_split": [2, 10]}),
        "svm": (
            # CalibratedClassifierCV gives SVM probabilities (SVC(probability=True) is deprecated)
            pipe(CalibratedClassifierCV(SVC(random_state=random_state), cv=3, ensemble=False)),
            {"clf__estimator__C": [1.0] if fast else [0.5, 2.0]}),
    }
    if XGBClassifier is not None:
        cands["xgboost"] = (
            pipe(XGBClassifier(eval_metric="logloss", random_state=random_state, n_jobs=1)),
            {"clf__n_estimators": [150] if fast else [150, 300],
             "clf__max_depth": [3] if fast else [3, 5],
             "clf__learning_rate": [0.1] if fast else [0.05, 0.1]})
    return cands


def _scores(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
    }


def _roc_points(y_true, proba, n: int = 80) -> dict:
    fpr, tpr, _ = roc_curve(y_true, proba)
    idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return {"fpr": [round(float(v), 4) for v in fpr[idx]], "tpr": [round(float(v), 4) for v in tpr[idx]]}


def train_and_evaluate_churn_model(X: pd.DataFrame, y, model_path: Path | str | None = None,
                                   metrics_path: Path | str | None = None, fast: bool = False,
                                   train_ref_date=None, verbose: bool = True) -> dict:
    """Train, tune and compare all candidates; persist the best pipeline + metrics."""
    model_path = Path(model_path or config.MODEL_PATH)
    metrics_path = Path(metrics_path or model_path.with_name("metrics.json"))
    X = X[FEATURE_COLUMNS]
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=config.RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    results, curves = {}, {}
    best_name, best_cv, best_model = None, -1.0, None
    for name, (pipeline, grid) in build_candidates(config.RANDOM_STATE, fast).items():
        t0 = time.time()
        search = GridSearchCV(pipeline, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        proba = search.best_estimator_.predict_proba(X_test)[:, 1]
        results[name] = {
            "cv_auc": round(float(search.best_score_), 4),
            "test": _scores(y_test, proba),
            "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                            for k, v in search.best_params_.items()},
            "seconds": round(time.time() - t0, 1),
        }
        curves[name] = _roc_points(y_test, proba)
        if verbose:
            print(f"  {name:<20} CV AUC={results[name]['cv_auc']:.4f}  "
                  f"test AUC={results[name]['test']['roc_auc']:.4f}  ({results[name]['seconds']}s)")
        # model selection on cross-validated AUC keeps the hold-out test set untouched
        if search.best_score_ > best_cv:
            best_name, best_cv, best_model = name, float(search.best_score_), search.best_estimator_

    proba_best = best_model.predict_proba(X_test)[:, 1]
    pred_best = (proba_best >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_best, labels=[0, 1]).ravel()

    # baseline: the 30-day-inactivity rule used by many shops today
    rule_pred = (X_test["recency_days"].to_numpy() > 30).astype(int)
    baseline = {
        "accuracy": round(float(accuracy_score(y_test, rule_pred)), 4),
        "precision": round(float(precision_score(y_test, rule_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, rule_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, rule_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, X_test["recency_days"])), 4),
    }

    trained_at = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
    version = f"{best_name}-{trained_at:%Y%m%d%H%M}"
    bundle = {
        "model": best_model,
        "features": list(FEATURE_COLUMNS),
        "name": best_name,
        "version": version,
        "sklearn_version": sklearn.__version__,
        "trained_at": trained_at.isoformat(),
        "train_ref_date": str(train_ref_date) if train_ref_date is not None else None,
        "background": X_train.sample(min(200, len(X_train)), random_state=0).reset_index(drop=True),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    importance = explain.global_importance(bundle, X_test)
    metrics = {
        "best_model": best_name,
        "model_version": version,
        "trained_at": trained_at.isoformat(),
        "train_ref_date": bundle["train_ref_date"],
        "n_customers": int(len(X)),
        "churn_rate": round(float(np.mean(y)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "models": results,
        "roc_curves": curves,
        "baseline_rule_30d": baseline,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "feature_importance": importance,
        "acceptance_targets": {"roc_auc": 0.85, "recall": 0.75},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    if verbose:
        print(f"Best model: {best_name} -> saved to {model_path}")
    return metrics


def main() -> None:
    from src import pipeline  # local import avoids a circular dependency

    parser = argparse.ArgumentParser(description="Train the churn models from the database")
    parser.add_argument("--fast", action="store_true", help="small hyper-parameter grids (quick demo)")
    args = parser.parse_args()
    engine = pipeline.init_and_seed()
    pipeline.train_model(engine, fast=args.fast)


if __name__ == "__main__":
    main()
