"""Streamlit entry point:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="E-Commerce Churn Analytics", page_icon="📉", layout="wide")

from src import config, db, pipeline  # noqa: E402  (after set_page_config)
from src import m5_dashboard  # noqa: E402


@st.cache_resource(show_spinner="Preparing the system - the first start can take a minute...")
def bootstrap():
    engine = db.init_db()
    if not db.ensure_bootstrap_admin(engine):
        return None, None
    return pipeline.ensure_ready(engine)


try:
    engine, service = bootstrap()
except ValueError as exc:  # e.g. ADMIN_PASSWORD shorter than 8 characters
    st.error(f"Configuration problem: {exc}")
    st.stop()
if engine is None:
    st.error("No administrator account exists yet.")
    st.markdown(
        "Set the **ADMIN_PASSWORD** (and optionally ADMIN_USERNAME) as an environment variable, in a "
        "`.env` file, or in Streamlit *Secrets*, then restart the app. See `DEPLOYMENT_GUIDE.md`."
    )
    st.stop()

if m5_dashboard.check_login(engine):
    m5_dashboard.render_app(engine, service)
