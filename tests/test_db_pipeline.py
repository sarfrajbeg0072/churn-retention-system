import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src import db, pipeline
from src.m1_preprocess import ValidationError


def test_tc09_authentication(engine):
    db.create_user(engine, "alice", "correct-horse", "Admin")
    assert db.verify_user(engine, "alice", "correct-horse") == "Admin"
    assert db.verify_user(engine, "alice", "wrong") is None
    assert db.verify_user(engine, "nobody", "x") is None


def test_password_is_stored_as_bcrypt_hash(engine):
    db.create_user(engine, "bob", "password123")
    h = db.list_users(engine)
    with engine.connect() as c:
        stored = c.execute(text("SELECT password_hash FROM tbl_admin_users")).scalar()
    assert stored.startswith("$2") and "password123" not in stored


def test_short_password_and_duplicate_user_rejected(engine):
    with pytest.raises(ValueError):
        db.create_user(engine, "x", "short")
    db.create_user(engine, "dup", "longenough1")
    with pytest.raises(ValueError):
        db.create_user(engine, "dup", "longenough1")


def test_foreign_key_enforced(engine):
    with pytest.raises(IntegrityError):
        with engine.begin() as c:
            c.execute(text("INSERT INTO tbl_orders (order_id, customer_id, order_timestamp, order_amount, item_count, payment_method) "
                           "VALUES ('o1','ghost','2026-01-01',10,1,'UPI')"))


def test_ingest_and_full_scoring_flow(engine, small_data, trained_bundle, monkeypatch):
    path, _, _ = trained_bundle
    from src.m4_scoring import ChurnScoringService
    customers, orders, tickets = small_data
    res = pipeline.ingest_frames(engine, customers, orders, tickets)
    assert res["inserted"]["customers"] == len(customers)
    again = pipeline.ingest_frames(engine, customers, orders, tickets)  # idempotent
    assert again["inserted"] == {"customers": 0, "orders": 0, "tickets": 0}
    summary = pipeline.refresh_features_and_predictions(engine, ChurnScoringService(path))
    assert summary["scored"] > 400
    assert db.table_count(engine, "tbl_churn_predictions") == summary["scored"]
    pipeline.refresh_features_and_predictions(engine, ChurnScoringService(path))  # history accumulates
    assert db.table_count(engine, "tbl_churn_predictions") == 2 * summary["scored"]
    assert db.table_count(engine, "tbl_feature_store") == summary["scored"]  # 1:1 with customers


def test_bad_upload_writes_nothing(engine, small_data):
    customers, orders, tickets = small_data
    bad_orders = orders.head(10).drop(columns=["order_amount"])
    with pytest.raises(ValidationError):
        pipeline.ingest_frames(engine, customers, bad_orders, None)
    assert db.table_count(engine, "tbl_customers") == 0  # all-or-nothing
