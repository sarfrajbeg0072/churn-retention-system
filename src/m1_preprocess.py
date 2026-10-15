"""Module 1 - Data ingestion, validation and preprocessing.

* ``DataIngestionPreprocessor``: scikit-learn transformer (median imputation +
  IQR outlier capping). All statistics are learned in ``fit`` only, so nothing
  from validation / production data leaks into the model.
* ``validate_*`` functions: schema + value checks for uploaded CSV files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataIngestionPreprocessor(BaseEstimator, TransformerMixin):
    """Median/mean imputation + IQR outlier capping.

    Zero-inflated columns (e.g. ``total_tickets`` where most customers have 0)
    have IQR == 0, and a classic 1.5*IQR rule would flatten the whole column to
    zero. For those columns the bounds fall back to the 1st / 99th percentile.
    """

    def __init__(self, numeric_impute_strategy: str = "median", iqr_factor: float = 1.5):
        self.numeric_impute_strategy = numeric_impute_strategy
        self.iqr_factor = iqr_factor

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        num_cols = X_df.select_dtypes(include=[np.number]).columns
        self.numeric_columns_ = list(num_cols)
        agg = "median" if self.numeric_impute_strategy == "median" else "mean"
        fill = X_df[num_cols].agg(agg)
        self.impute_values_ = fill.fillna(0.0).to_dict()  # all-null column -> 0

        q1 = X_df[num_cols].quantile(0.25)
        q3 = X_df[num_cols].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.iqr_factor * iqr
        upper = q3 + self.iqr_factor * iqr
        p01 = X_df[num_cols].quantile(0.01)
        p99 = X_df[num_cols].quantile(0.99)
        degenerate = iqr == 0
        lower = lower.where(~degenerate, p01)
        upper = upper.where(~degenerate, p99)
        self.lower_bounds_ = lower.fillna(-np.inf).to_dict()
        self.upper_bounds_ = upper.fillna(np.inf).to_dict()
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in self.numeric_columns_:
            if col in X_df.columns:
                X_df[col] = X_df[col].astype(float).fillna(self.impute_values_[col])
                X_df[col] = X_df[col].clip(self.lower_bounds_[col], self.upper_bounds_[col])
        return X_df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.numeric_columns_)


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
class ValidationError(Exception):
    """Raised when an uploaded file is rejected. ``errors`` lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


REQUIRED = {
    "customers": ["customer_id", "signup_timestamp"],
    "orders": ["order_id", "customer_id", "order_timestamp", "order_amount"],
    "tickets": ["ticket_id", "customer_id", "created_at"],
}
CUSTOMER_COLUMNS = ["customer_id", "first_name", "last_name", "email", "signup_timestamp", "geography", "gender"]
ORDER_COLUMNS = ["order_id", "customer_id", "order_timestamp", "order_amount", "item_count", "payment_method"]
TICKET_COLUMNS = ["ticket_id", "customer_id", "created_at", "issue_category", "status", "resolution_hours"]
_STATUS_MAP = {"open": "Open", "closed": "Closed", "resolved": "Closed", "escalated": "Escalated"}


def _prepare(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED[kind] if c not in df.columns]
    if missing:
        raise ValidationError([f"{kind}: missing mandatory column(s): {', '.join(missing)}"])
    return df


def _parse_dates(df: pd.DataFrame, col: str, kind: str, errors: list[str]) -> pd.DataFrame:
    parsed = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
    bad = df.index[parsed.isna()]
    if len(bad):
        errors.append(f"{kind}: {len(bad)} row(s) have a malformed '{col}' (e.g. data rows {[int(i) + 2 for i in bad[:5]]}).")
    df[col] = parsed
    return df


def _dedupe(df: pd.DataFrame, key: str, kind: str, warnings: list[str]) -> pd.DataFrame:
    n = int(df.duplicated(subset=key).sum())
    if n:
        warnings.append(f"{kind}: removed {n} duplicate {key} row(s).")
        df = df.drop_duplicates(subset=key, keep="first")
    return df


def validate_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    df = _prepare(df, "customers")
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df = _parse_dates(df, "signup_timestamp", "customers", errors)
    if errors:
        raise ValidationError(errors)
    df = _dedupe(df, "customer_id", "customers", warnings)
    for col in ("first_name", "last_name"):
        df[col] = df[col].fillna("Masked").astype(str).str[:50] if col in df.columns else "Masked"
    if "email" not in df.columns:
        df["email"] = df["customer_id"] + "@masked.local"
    df["email"] = df["email"].fillna(df["customer_id"] + "@masked.local").astype(str).str[:100]
    if df["email"].duplicated().any():  # e-mail must be unique
        dup = df["email"].duplicated(keep="first")
        df.loc[dup, "email"] = df.loc[dup, "customer_id"] + "@masked.local"
    df["geography"] = df["geography"].fillna("Unknown").astype(str).str[:50] if "geography" in df.columns else "Unknown"
    if "gender" in df.columns:
        g = df["gender"].astype(str).str.strip().str.title()
        df["gender"] = g.where(g.isin(["Male", "Female", "Other"]), None)
    else:
        df["gender"] = None
    return df[CUSTOMER_COLUMNS].reset_index(drop=True), warnings


def validate_orders(df: pd.DataFrame, known_customers: set[str]) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    df = _prepare(df, "orders")
    df["order_id"] = df["order_id"].astype(str).str.strip()
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df = _parse_dates(df, "order_timestamp", "orders", errors)
    amount = pd.to_numeric(df["order_amount"], errors="coerce")
    if amount.isna().any():
        errors.append(f"orders: {int(amount.isna().sum())} row(s) have a non-numeric 'order_amount'.")
    if (amount < 0).any():
        errors.append(f"orders: {int((amount < 0).sum())} row(s) have a negative 'order_amount' "
                      f"(e.g. data rows {[int(i) + 2 for i in df.index[amount < 0][:5]]}).")
    df["order_amount"] = amount
    if "item_count" in df.columns:
        items = pd.to_numeric(df["item_count"], errors="coerce").fillna(1)
        if (items <= 0).any():
            errors.append("orders: 'item_count' must be greater than 0.")
        df["item_count"] = items.astype(int)
    else:
        df["item_count"] = 1
    df["payment_method"] = (df["payment_method"].fillna("Unknown").astype(str).str[:30]
                            if "payment_method" in df.columns else "Unknown")
    orphan = ~df["customer_id"].isin(known_customers)
    if orphan.any():
        errors.append(f"orders: {int(orphan.sum())} row(s) reference an unknown customer_id.")
    if errors:
        raise ValidationError(errors)
    df = _dedupe(df, "order_id", "orders", warnings)
    return df[ORDER_COLUMNS].reset_index(drop=True), warnings


def validate_tickets(df: pd.DataFrame, known_customers: set[str]) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    df = _prepare(df, "tickets")
    df["ticket_id"] = df["ticket_id"].astype(str).str.strip()
    df["customer_id"] = df["customer_id"].astype(str).str.strip()
    df = _parse_dates(df, "created_at", "tickets", errors)
    if "resolution_hours" in df.columns:
        hours = pd.to_numeric(df["resolution_hours"], errors="coerce").fillna(0.0)
        if (hours < 0).any():
            errors.append("tickets: 'resolution_hours' cannot be negative.")
        df["resolution_hours"] = hours
    else:
        df["resolution_hours"] = 0.0
    df["issue_category"] = (df["issue_category"].fillna("Other").astype(str).str[:50]
                            if "issue_category" in df.columns else "Other")
    if "status" in df.columns:
        status = df["status"].astype(str).str.strip().str.lower().map(_STATUS_MAP)
        if status.isna().any():
            errors.append("tickets: 'status' must be one of Open, Closed, Escalated.")
        df["status"] = status
    else:
        df["status"] = "Closed"
    orphan = ~df["customer_id"].isin(known_customers)
    if orphan.any():
        errors.append(f"tickets: {int(orphan.sum())} row(s) reference an unknown customer_id.")
    if errors:
        raise ValidationError(errors)
    df = _dedupe(df, "ticket_id", "tickets", warnings)
    return df[TICKET_COLUMNS].reset_index(drop=True), warnings
