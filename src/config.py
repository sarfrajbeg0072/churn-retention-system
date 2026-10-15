"""Central configuration.

Settings are read (in this order) from:
  1. real environment variables,
  2. a local ``.env`` file (development / VPS),
  3. Streamlit secrets (Streamlit Community Cloud).
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"


def get_setting(name: str, default: str | None = None) -> str | None:
    """Return a setting from env, then Streamlit secrets, then ``default``."""
    value = os.getenv(name)
    if value not in (None, ""):
        return value
    try:  # Streamlit secrets are optional
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file / streamlit not running
        pass
    return default


# --- Database --------------------------------------------------------------
def _normalise_db_url(url: str) -> str:
    # Hosted Postgres providers often hand out "postgres://"; SQLAlchemy needs "postgresql://"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


DATABASE_URL = _normalise_db_url(
    get_setting("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'churn.db').as_posix()}")
)

# --- Model / business rules ------------------------------------------------
MODEL_PATH = Path(get_setting("MODEL_PATH", str(MODEL_DIR / "best_churn_model.joblib")))
METRICS_PATH = MODEL_PATH.with_name("metrics.json")
CHURN_WINDOW_DAYS = int(get_setting("CHURN_WINDOW_DAYS", "90"))
HIGH_THRESHOLD = float(get_setting("HIGH_RISK_THRESHOLD", "0.70"))
MEDIUM_THRESHOLD = float(get_setting("MEDIUM_RISK_THRESHOLD", "0.35"))
RANDOM_STATE = 42

# --- Security --------------------------------------------------------------
ADMIN_USERNAME = get_setting("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = get_setting("ADMIN_PASSWORD")  # no default on purpose
API_KEY = get_setting("API_KEY")  # protects the optional Flask API
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 60


def get_audit_logger() -> logging.Logger:
    """Audit log: logins, uploads, exports, retraining (written to logs/audit.log)."""
    logger = logging.getLogger("churn.audit")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        try:
            LOG_DIR.mkdir(exist_ok=True)
            handler: logging.Handler = RotatingFileHandler(
                LOG_DIR / "audit.log", maxBytes=1_000_000, backupCount=3
            )
        except OSError:  # read-only file system
            handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger

CURRENCY_SYMBOL = get_setting("CURRENCY_SYMBOL", "₹")
