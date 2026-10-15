import numpy as np
import pandas as pd
import pytest

from src.m1_preprocess import (DataIngestionPreprocessor, ValidationError,
                               validate_customers, validate_orders, validate_tickets)


def test_tc02_median_imputation_uses_training_median():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0, np.nan] * 10})
    p = DataIngestionPreprocessor().fit(train)
    out = p.transform(pd.DataFrame({"x": [np.nan]}))
    assert out["x"].isna().sum() == 0
    assert out["x"].iloc[0] == pytest.approx(train["x"].median())


def test_tc03_outlier_capped_to_upper_bound():
    train = pd.DataFrame({"x": np.arange(1, 101, dtype=float)})
    p = DataIngestionPreprocessor().fit(train)
    out = p.transform(pd.DataFrame({"x": [10_000.0]}))
    assert out["x"].iloc[0] == pytest.approx(p.upper_bounds_["x"]) and out["x"].iloc[0] < 200


def test_zero_inflated_column_is_not_flattened():
    train = pd.DataFrame({"tickets": [0] * 80 + list(range(1, 21))}, dtype=float)
    p = DataIngestionPreprocessor().fit(train)
    assert p.transform(pd.DataFrame({"tickets": [15.0]}))["tickets"].iloc[0] > 0


def test_all_null_column_does_not_crash():
    p = DataIngestionPreprocessor().fit(pd.DataFrame({"x": [np.nan] * 5}))
    assert p.transform(pd.DataFrame({"x": [np.nan]}))["x"].iloc[0] == 0


def test_tc06_missing_mandatory_column_rejected():
    df = pd.DataFrame({"order_id": ["1"], "customer_id": ["A"], "order_timestamp": ["2026-01-01"]})
    with pytest.raises(ValidationError) as e:
        validate_orders(df, {"A"})
    assert "order_amount" in str(e.value)


def test_negative_amount_malformed_date_and_unknown_customer_rejected():
    df = pd.DataFrame({"order_id": ["1", "2", "3"], "customer_id": ["A", "A", "Z"],
                       "order_timestamp": ["2026-01-01", "not-a-date", "2026-01-02"],
                       "order_amount": [-5, 10, 10]})
    with pytest.raises(ValidationError) as e:
        validate_orders(df, {"A"})
    assert len(e.value.errors) == 3


def test_customers_defaults_and_duplicates():
    df = pd.DataFrame({"Customer_ID": ["A", "A", "B"], "signup_timestamp": ["2026-01-01"] * 3})
    clean, warns = validate_customers(df)
    assert len(clean) == 2 and warns and clean.email.is_unique


def test_tickets_status_validation():
    df = pd.DataFrame({"ticket_id": ["1"], "customer_id": ["A"], "created_at": ["2026-01-01"], "status": ["weird"]})
    with pytest.raises(ValidationError):
        validate_tickets(df, {"A"})
