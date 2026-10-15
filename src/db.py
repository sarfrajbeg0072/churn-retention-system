"""Relational layer (SQLAlchemy 2.0): six tables, 3NF, parameterised access.

Run ``python -m src.db --create-admin`` to create the first administrator.
"""
from __future__ import annotations

import argparse
import getpass
from datetime import datetime

import bcrypt
import pandas as pd
from sqlalchemy import (CheckConstraint, DateTime, Float, ForeignKey, Integer,
                        Numeric, String, Text, create_engine, event, func,
                        select, text)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (DeclarativeBase, Mapped, Session, mapped_column,
                            sessionmaker)

from src import config


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "tbl_customers"
    customer_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    first_name: Mapped[str] = mapped_column(String(50), nullable=False)
    last_name: Mapped[str] = mapped_column(String(50), nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    signup_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    geography: Mapped[str] = mapped_column(String(50), default="Unknown")
    gender: Mapped[str | None] = mapped_column(String(10))
    __table_args__ = (CheckConstraint("gender IN ('Male','Female','Other')", name="ck_gender"),)


class Order(Base):
    __tablename__ = "tbl_orders"
    order_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("tbl_customers.customer_id"), nullable=False, index=True)
    order_timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    order_amount: Mapped[float] = mapped_column(Numeric(10, 2))
    item_count: Mapped[int] = mapped_column(Integer)
    payment_method: Mapped[str] = mapped_column(String(30), nullable=False)
    __table_args__ = (
        CheckConstraint("order_amount >= 0.00", name="ck_amount"),
        CheckConstraint("item_count > 0", name="ck_items"),
    )


class SupportTicket(Base):
    __tablename__ = "tbl_support_tickets"
    ticket_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("tbl_customers.customer_id"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    issue_category: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str | None] = mapped_column(String(20))
    resolution_hours: Mapped[float] = mapped_column(Float, default=0.0)
    __table_args__ = (CheckConstraint("status IN ('Open','Closed','Escalated')", name="ck_status"),)


class FeatureStore(Base):
    __tablename__ = "tbl_feature_store"
    feature_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("tbl_customers.customer_id"), unique=True, nullable=False)
    recency_days: Mapped[int] = mapped_column(Integer, nullable=False)
    frequency_count: Mapped[int] = mapped_column(Integer, nullable=False)
    monetary_sum: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    avg_order_value: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    total_tickets: Mapped[int] = mapped_column(Integer, default=0)
    avg_resolution_hours: Mapped[float] = mapped_column(Float, default=0.0)
    support_ticket_density: Mapped[float] = mapped_column(Float, default=0.0)
    tenure_days: Mapped[int] = mapped_column(Integer, default=0)
    orders_per_month: Mapped[float] = mapped_column(Float, default=0.0)
    orders_last_90d: Mapped[int] = mapped_column(Integer, default=0)
    avg_gap_days: Mapped[float] = mapped_column(Float, default=0.0)
    reference_date: Mapped[datetime] = mapped_column(DateTime)
    last_updated: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChurnPrediction(Base):
    __tablename__ = "tbl_churn_predictions"
    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("tbl_customers.customer_id"), nullable=False, index=True)
    churn_probability: Mapped[float] = mapped_column(Float)
    risk_category: Mapped[str] = mapped_column(String(15))
    primary_churn_driver: Mapped[str] = mapped_column(String(100), nullable=False)
    recommended_action: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(String(60), nullable=False)
    execution_date: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    __table_args__ = (
        CheckConstraint("churn_probability BETWEEN 0.0 AND 1.0", name="ck_prob"),
        CheckConstraint("risk_category IN ('Low','Medium','High')", name="ck_risk"),
    )


class AdminUser(Base):
    __tablename__ = "tbl_admin_users"
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[str] = mapped_column(String(20), default="Analyst")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    __table_args__ = (CheckConstraint("role IN ('Admin','Analyst')", name="ck_role"),)


# --------------------------------------------------------------------------
_ENGINES: dict[str, Engine] = {}


def get_engine(url: str | None = None) -> Engine:
    url = url or config.DATABASE_URL
    if url not in _ENGINES:
        kwargs = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(engine, "connect")
            def _fk_pragma(dbapi_conn, _):  # enforce foreign keys in SQLite
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        _ENGINES[url] = engine
    return _ENGINES[url]


def init_db(engine: Engine | None = None) -> Engine:
    engine = engine or get_engine()
    if engine.url.get_backend_name() == "sqlite" and engine.url.database:
        from pathlib import Path
        Path(engine.url.database).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)
    return engine


def session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(engine or get_engine(), expire_on_commit=False)


# --- Users / authentication ------------------------------------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def create_user(engine: Engine, username: str, password: str, role: str = "Analyst") -> None:
    if role not in ("Admin", "Analyst"):
        raise ValueError("role must be 'Admin' or 'Analyst'")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    with session_factory(engine)() as s:
        if s.scalar(select(AdminUser).where(AdminUser.username == username)):
            raise ValueError(f"User '{username}' already exists.")
        s.add(AdminUser(username=username, password_hash=hash_password(password), role=role))
        s.commit()


def verify_user(engine: Engine, username: str, password: str) -> str | None:
    """Return the user's role if credentials are valid, else None."""
    with session_factory(engine)() as s:
        user = s.scalar(select(AdminUser).where(AdminUser.username == username))
        # always run a hash check so timing does not reveal valid usernames
        stored = user.password_hash.encode() if user else bcrypt.hashpw(b"x", bcrypt.gensalt())
        ok = bcrypt.checkpw(password.encode("utf-8"), stored)
        return user.role if (user and ok) else None


def list_users(engine: Engine) -> pd.DataFrame:
    return pd.read_sql_query(
        text("SELECT user_id, username, role, created_at FROM tbl_admin_users ORDER BY user_id"), engine)


def delete_user(engine: Engine, username: str) -> None:
    with session_factory(engine)() as s:
        user = s.scalar(select(AdminUser).where(AdminUser.username == username))
        if user:
            s.delete(user)
            s.commit()


def ensure_bootstrap_admin(engine: Engine) -> bool:
    """Create the first Admin from ADMIN_USERNAME / ADMIN_PASSWORD if no users exist."""
    with session_factory(engine)() as s:
        if s.scalar(select(func.count()).select_from(AdminUser)):
            return True
    pw = config.ADMIN_PASSWORD
    if not pw or pw.lower().startswith("change-me"):  # refuse missing / placeholder passwords
        return False
    create_user(engine, config.ADMIN_USERNAME, config.ADMIN_PASSWORD, "Admin")
    return True


# --- Bulk data helpers -----------------------------------------------------
def table_count(engine: Engine, table: str) -> int:
    allowed = {t.name for t in Base.metadata.sorted_tables}
    if table not in allowed:
        raise ValueError("unknown table")
    with engine.connect() as c:
        return int(c.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0)


def insert_new_rows(engine: Engine, table: str, df: pd.DataFrame, key: str) -> int:
    """Insert rows whose primary key is not yet present. Returns rows inserted."""
    if df.empty:
        return 0
    existing = pd.read_sql_query(text(f"SELECT {key} FROM {table}"), engine)[key]
    new = df[~df[key].isin(set(existing))]
    if new.empty:
        return 0
    new.to_sql(table, engine, if_exists="append", index=False, chunksize=2000)
    return len(new)


def read_customers(engine: Engine) -> pd.DataFrame:
    return pd.read_sql_query(text("SELECT * FROM tbl_customers"), engine, parse_dates=["signup_timestamp"])


def read_orders(engine: Engine) -> pd.DataFrame:
    return pd.read_sql_query(text("SELECT * FROM tbl_orders"), engine, parse_dates=["order_timestamp"])


def read_tickets(engine: Engine) -> pd.DataFrame:
    return pd.read_sql_query(text("SELECT * FROM tbl_support_tickets"), engine, parse_dates=["created_at"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Database utilities")
    parser.add_argument("--create-admin", action="store_true", help="create an Admin user interactively")
    parser.add_argument("--init", action="store_true", help="create tables only")
    args = parser.parse_args()
    engine = init_db()
    if args.create_admin:
        username = input("Admin username [admin]: ").strip() or "admin"
        password = getpass.getpass("Password (min 8 chars): ")
        create_user(engine, username, password, "Admin")
        print(f"Admin '{username}' created.")
    elif args.init:
        print("Tables created.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
