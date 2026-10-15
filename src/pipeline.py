"""Orchestration: seed data, train, extract features, score, store predictions.

CLI examples
    python -m src.pipeline --bootstrap     # DB + sample data + model + predictions
    python -m src.pipeline --rescore       # nightly batch scoring (cron)
    python -m src.pipeline --retrain       # retrain model then rescore
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src import config, db
from src.data_generator import generate_synthetic_data
from src.m1_preprocess import (ValidationError, validate_customers,
                               validate_orders, validate_tickets)
from src.m2_features import (FEATURE_COLUMNS, generate_customer_feature_matrix,
                             latest_reference_date, make_churn_labels,
                             training_reference_date)
from src.m3_train import train_and_evaluate_churn_model
from src.m4_scoring import ChurnScoringService


# --------------------------------------------------------------------------
# Data ingestion
# --------------------------------------------------------------------------
def ingest_frames(engine: Engine, customers: pd.DataFrame | None, orders: pd.DataFrame | None,
                  tickets: pd.DataFrame | None) -> dict:
    """Validate and insert new rows. Raises ValidationError (nothing written) on bad data."""
    errors: list[str] = []
    warnings: list[str] = []
    existing = set(db.read_customers(engine)["customer_id"]) if db.table_count(engine, "tbl_customers") else set()

    clean_c = clean_o = clean_t = None
    try:
        if customers is not None:
            clean_c, w = validate_customers(customers)
            warnings += w
            existing |= set(clean_c["customer_id"])
    except ValidationError as e:
        errors += e.errors
    for name, frame, fn in (("orders", orders, validate_orders), ("tickets", tickets, validate_tickets)):
        if frame is None:
            continue
        try:
            clean, w = fn(frame, existing)
            warnings += w
            if name == "orders":
                clean_o = clean
            else:
                clean_t = clean
        except ValidationError as e:
            errors += e.errors
    if errors:
        raise ValidationError(errors)  # all-or-nothing: nothing written

    inserted = {"customers": 0, "orders": 0, "tickets": 0}
    if clean_c is not None:
        inserted["customers"] = db.insert_new_rows(engine, "tbl_customers", clean_c, "customer_id")
    if clean_o is not None:
        inserted["orders"] = db.insert_new_rows(engine, "tbl_orders", clean_o, "order_id")
    if clean_t is not None:
        inserted["tickets"] = db.insert_new_rows(engine, "tbl_support_tickets", clean_t, "ticket_id")
    return {"inserted": inserted, "warnings": warnings}


def load_sample_data(engine: Engine) -> dict:
    """Load data/*.csv (or generate synthetic data if the files are missing)."""
    c_p, o_p, t_p = (config.DATA_DIR / n for n in ("customers.csv", "orders.csv", "support_tickets.csv"))
    if c_p.exists() and o_p.exists() and t_p.exists():
        customers, orders, tickets = pd.read_csv(c_p), pd.read_csv(o_p), pd.read_csv(t_p)
    else:
        customers, orders, tickets = generate_synthetic_data()
    return ingest_frames(engine, customers, orders, tickets)


# --------------------------------------------------------------------------
# Features, training, scoring
# --------------------------------------------------------------------------
def train_model(engine: Engine, fast: bool = False, verbose: bool = True) -> dict:
    customers, orders, tickets = db.read_customers(engine), db.read_orders(engine), db.read_tickets(engine)
    ref = training_reference_date(orders, config.CHURN_WINDOW_DAYS)
    feats = generate_customer_feature_matrix(orders, tickets, customers, ref)
    y = make_churn_labels(orders, feats["customer_id"], ref, config.CHURN_WINDOW_DAYS)
    if verbose:
        print(f"Training on {len(feats):,} customers (T_ref={ref.date()}, churn rate={y.mean():.1%})")
    return train_and_evaluate_churn_model(feats[FEATURE_COLUMNS], y.to_numpy(), fast=fast,
                                          train_ref_date=ref.date(), verbose=verbose)


def get_service(engine: Engine, allow_train: bool = True, fast: bool = True) -> ChurnScoringService:
    svc = ChurnScoringService.load_if_compatible()
    if svc is None:
        if not allow_train:
            raise RuntimeError("No compatible trained model found. Run: python -m src.m3_train")
        train_model(engine, fast=fast, verbose=False)
        svc = ChurnScoringService()
    return svc


def refresh_features_and_predictions(engine: Engine, service: ChurnScoringService | None = None,
                                     reference_date=None) -> dict:
    """Recompute features at T_ref (default: day after last order), score, store."""
    service = service or get_service(engine)
    customers, orders, tickets = db.read_customers(engine), db.read_orders(engine), db.read_tickets(engine)
    if orders.empty:
        return {"scored": 0, "excluded_no_history": len(customers)}
    ref = pd.Timestamp(reference_date) if reference_date is not None else latest_reference_date(orders)
    feats = generate_customer_feature_matrix(orders, tickets, customers, ref)

    with engine.begin() as conn:  # feature store holds exactly one current row per customer
        conn.execute(text("DELETE FROM tbl_feature_store"))
    feats.to_sql("tbl_feature_store", engine, if_exists="append", index=False, chunksize=2000)

    scored = service.score_customers(feats, with_drivers=True)
    now = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
    with engine.connect() as conn:
        last = conn.execute(text("SELECT MAX(execution_date) FROM tbl_churn_predictions")).scalar()
    if last is not None:
        last = pd.Timestamp(last).to_pydatetime()
        if now <= last:  # keep run timestamps strictly increasing
            now = last + timedelta(seconds=1)
    scored["execution_date"] = now
    scored[["customer_id", "churn_probability", "risk_category", "primary_churn_driver",
            "recommended_action", "model_version", "execution_date"]].to_sql(
        "tbl_churn_predictions", engine, if_exists="append", index=False, chunksize=2000)
    return {"scored": len(scored), "excluded_no_history": len(customers) - len(scored),
            "reference_date": str(ref.date()), "model_version": service.version}


def init_and_seed(engine: Engine | None = None) -> Engine:
    """Create tables, bootstrap admin, and load sample data if the database is empty."""
    engine = db.init_db(engine)
    db.ensure_bootstrap_admin(engine)
    if db.table_count(engine, "tbl_customers") == 0:
        load_sample_data(engine)
    return engine


def ensure_ready(engine: Engine | None = None) -> tuple[Engine, ChurnScoringService]:
    """Everything the dashboard needs: data, a compatible model, and fresh predictions."""
    engine = init_and_seed(engine)
    service = get_service(engine, fast=True)  # quick retrain only if the saved model is unusable
    if db.table_count(engine, "tbl_churn_predictions") == 0:
        refresh_features_and_predictions(engine, service)
    return engine, service


def main() -> None:
    p = argparse.ArgumentParser(description="Churn pipeline tasks")
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--rescore", action="store_true")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--fast", action="store_true", help="quick training grids")
    a = p.parse_args()
    if a.bootstrap:
        engine, svc = ensure_ready()
        print("Ready. Model:", svc.version)
    elif a.retrain:
        engine = init_and_seed()
        train_model(engine, fast=a.fast)
        print(refresh_features_and_predictions(engine, ChurnScoringService()))
    elif a.rescore:
        engine = init_and_seed()
        print(refresh_features_and_predictions(engine, get_service(engine, allow_train=False)))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
