"""Module 2 - Feature engineering & RFM analytics engine.

Leakage rule: features use ONLY records strictly before the reference date
``T_ref``; the churn label is computed from the 90 days that follow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "recency_days",
    "frequency_count",
    "monetary_sum",
    "avg_order_value",
    "total_tickets",
    "avg_resolution_hours",
    "support_ticket_density",
    "tenure_days",
    "orders_per_month",
    "orders_last_90d",
    "avg_gap_days",
]

FEATURE_LABELS = {
    "recency_days": "Recency (days since last order)",
    "frequency_count": "Order frequency",
    "monetary_sum": "Total spend",
    "avg_order_value": "Average order value",
    "total_tickets": "Number of support tickets",
    "avg_resolution_hours": "Slow ticket resolution",
    "support_ticket_density": "Support tickets per order",
    "tenure_days": "Customer tenure",
    "orders_per_month": "Order cadence (orders/month)",
    "orders_last_90d": "Orders in last 90 days",
    "avg_gap_days": "Average gap between orders",
}


def generate_customer_feature_matrix(orders_df: pd.DataFrame, tickets_df: pd.DataFrame,
                                     customers_df: pd.DataFrame, reference_date) -> pd.DataFrame:
    """One feature row per customer using ONLY data before ``reference_date``.

    Customers with no orders before T_ref are dropped (cold-start rule).
    """
    ref = pd.Timestamp(reference_date)
    orders = orders_df[orders_df["order_timestamp"] < ref].copy()
    tickets = tickets_df[tickets_df["created_at"] < ref]  # tickets are time-filtered too

    orders["order_amount"] = orders["order_amount"].astype(float)
    orders["in_last_90"] = orders["order_timestamp"] >= (ref - pd.Timedelta(days=90))

    rfm = orders.groupby("customer_id").agg(
        first_order=("order_timestamp", "min"),
        last_order=("order_timestamp", "max"),
        frequency_count=("order_id", "count"),
        monetary_sum=("order_amount", "sum"),
        avg_order_value=("order_amount", "mean"),
        orders_last_90d=("in_last_90", "sum"),
    ).reset_index()
    rfm["recency_days"] = (ref - rfm["last_order"]).dt.days
    span_days = (rfm["last_order"] - rfm["first_order"]).dt.total_seconds() / 86400.0

    ticket_summary = tickets.groupby("customer_id").agg(
        total_tickets=("ticket_id", "count"),
        avg_resolution_hours=("resolution_hours", "mean"),
    ).reset_index()

    feats = rfm.merge(ticket_summary, on="customer_id", how="left")
    feats = feats.merge(customers_df[["customer_id", "signup_timestamp"]], on="customer_id", how="inner")
    feats["total_tickets"] = feats["total_tickets"].fillna(0)
    feats["avg_resolution_hours"] = feats["avg_resolution_hours"].fillna(0.0)

    feats["tenure_days"] = (ref - feats["signup_timestamp"]).dt.days.clip(lower=0)
    feats["support_ticket_density"] = feats["total_tickets"] / feats["frequency_count"]
    feats["orders_per_month"] = feats["frequency_count"] / (feats["tenure_days"].clip(lower=30) / 30.0)
    # average gap between consecutive orders; single-order customers -> tenure
    gaps = np.where(feats["frequency_count"] > 1,
                    span_days.reindex(feats.index).to_numpy() / (feats["frequency_count"] - 1).clip(lower=1),
                    feats["tenure_days"])
    feats["avg_gap_days"] = gaps

    feats["orders_last_90d"] = feats["orders_last_90d"].astype(int)
    out = feats[["customer_id"] + FEATURE_COLUMNS].copy()
    out["reference_date"] = ref
    return out.reset_index(drop=True)


def make_churn_labels(orders_df: pd.DataFrame, customer_ids, reference_date, window_days: int = 90) -> pd.Series:
    """label = 1 if the customer places NO order in [T_ref, T_ref + window)."""
    ref = pd.Timestamp(reference_date)
    end = ref + pd.Timedelta(days=window_days)
    future = orders_df[(orders_df["order_timestamp"] >= ref) & (orders_df["order_timestamp"] < end)]
    active = set(future["customer_id"])
    return pd.Series([0 if c in active else 1 for c in customer_ids],
                     index=list(customer_ids), name="churn")


def latest_reference_date(orders_df: pd.DataFrame) -> pd.Timestamp:
    """Scoring date = the day after the most recent recorded order."""
    return orders_df["order_timestamp"].max().normalize() + pd.Timedelta(days=1)


def training_reference_date(orders_df: pd.DataFrame, window_days: int = 90) -> pd.Timestamp:
    """Latest T_ref whose full label window is already observed in the data."""
    return latest_reference_date(orders_df) - pd.Timedelta(days=window_days)
