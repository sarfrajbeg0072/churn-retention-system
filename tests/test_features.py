import pandas as pd

from src.m2_features import generate_customer_feature_matrix, make_churn_labels

REF = pd.Timestamp("2026-01-01")


def _frames(order_times, tickets=()):
    customers = pd.DataFrame({"customer_id": ["A", "B"],
                              "signup_timestamp": [pd.Timestamp("2025-01-01")] * 2})
    orders = pd.DataFrame({"order_id": [f"o{i}" for i in range(len(order_times))],
                           "customer_id": ["A"] * len(order_times),
                           "order_timestamp": pd.to_datetime(order_times),
                           "order_amount": [100.0] * len(order_times)})
    tix = pd.DataFrame({"ticket_id": [f"t{i}" for i in range(len(tickets))],
                        "customer_id": ["A"] * len(tickets),
                        "created_at": pd.to_datetime(list(tickets)),
                        "resolution_hours": [10.0] * len(tickets)})
    return customers, orders, tix


def test_tc01_recency_and_reference_day_excluded():
    c, o, t = _frames(["2025-12-31", "2026-01-01"])  # second order is ON T_ref -> excluded
    f = generate_customer_feature_matrix(o, t, c, REF)
    row = f[f.customer_id == "A"].iloc[0]
    assert row.recency_days == 1
    assert row.frequency_count == 1


def test_tc04_zero_orders_customer_excluded_no_division_error():
    c, o, t = _frames(["2025-12-01"])
    f = generate_customer_feature_matrix(o, t, c, REF)
    assert "B" not in set(f.customer_id)
    assert f.isna().sum().sum() == 0


def test_tc07_no_leakage_from_future_orders_or_tickets():
    c, o, t = _frames(["2025-12-01", "2026-02-01"], tickets=["2025-12-02", "2026-03-01"])
    f = generate_customer_feature_matrix(o, t, c, REF).iloc[0]
    assert f.frequency_count == 1
    assert f.total_tickets == 1


def test_churn_labels_window():
    c, o, t = _frames(["2025-12-01", "2026-02-01"])
    y = make_churn_labels(o, ["A", "B"], REF, 90)
    assert y["A"] == 0 and y["B"] == 1
    y2 = make_churn_labels(o, ["A"], REF, 10)  # order on Feb 1 is outside a 10-day window
    assert y2["A"] == 1
