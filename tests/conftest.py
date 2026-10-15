import pandas as pd
import pytest

from src import db, m3_train, pipeline
from src.data_generator import generate_synthetic_data
from src.m2_features import (FEATURE_COLUMNS, generate_customer_feature_matrix,
                             make_churn_labels, training_reference_date)


@pytest.fixture(scope="session")
def small_data():
    return generate_synthetic_data(n_customers=500, seed=7, end_date="2026-06-30")


@pytest.fixture(scope="session")
def trained_bundle(small_data, tmp_path_factory):
    """Train a small, fast model once for all tests (LR + RF only)."""
    customers, orders, tickets = small_data
    ref = training_reference_date(orders)
    feats = generate_customer_feature_matrix(orders, tickets, customers, ref)
    y = make_churn_labels(orders, feats["customer_id"], ref)
    full = m3_train.build_candidates

    def tiny(random_state=42, fast=True):
        c = full(random_state, True)
        return {k: c[k] for k in ("logistic_regression", "random_forest")}

    m3_train.build_candidates = tiny
    path = tmp_path_factory.mktemp("model") / "m.joblib"
    try:
        metrics = m3_train.train_and_evaluate_churn_model(
            feats[FEATURE_COLUMNS], y.to_numpy(), model_path=path, fast=True, verbose=False)
    finally:
        m3_train.build_candidates = full
    return path, metrics, feats


@pytest.fixture()
def engine(tmp_path):
    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    return db.init_db(db.get_engine(url))
