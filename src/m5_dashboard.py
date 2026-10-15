"""Module 5 - Interactive executive dashboard & retention interface (Streamlit)."""
from __future__ import annotations

import io
import json
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from src import config, db, pipeline
from src.m1_preprocess import ValidationError
from src.m2_features import FEATURE_COLUMNS, FEATURE_LABELS

AUDIT = config.get_audit_logger()
RISK_COLORS = {"High": "#d62728", "Medium": "#ff9f1c", "Low": "#2ca02c"}
CUR = config.CURRENCY_SYMBOL


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------
def check_login(engine) -> bool:
    """BCrypt-verified login with a simple lock-out after repeated failures."""
    if st.session_state.get("auth"):
        return True
    st.title("🔐 Administrator Login")
    st.caption("E-Commerce Customer Churn Prediction & Retention Analytics")
    locked_until = st.session_state.get("locked_until", 0)
    if time.time() < locked_until:
        st.error(f"Too many failed attempts. Try again in {int(locked_until - time.time())} seconds.")
        return False
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        role = db.verify_user(engine, username.strip(), password)
        if role:
            st.session_state.update(auth=True, user=username.strip(), role=role, failed=0)
            AUDIT.info("LOGIN_OK user=%s role=%s", username, role)
            st.rerun()
        else:
            st.session_state["failed"] = st.session_state.get("failed", 0) + 1
            AUDIT.warning("LOGIN_FAIL user=%s", username)
            if st.session_state["failed"] >= config.MAX_LOGIN_ATTEMPTS:
                st.session_state["locked_until"] = time.time() + config.LOCKOUT_SECONDS
                st.session_state["failed"] = 0
            st.error("Invalid credentials")
    return False


# --------------------------------------------------------------------------
# Data access (cached)
# --------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_scored(_engine, _cache_key: str = "") -> pd.DataFrame:
    sql = text("""
        SELECT p.customer_id, p.churn_probability, p.risk_category, p.primary_churn_driver,
               p.recommended_action, p.model_version, p.execution_date,
               f.recency_days, f.frequency_count, f.monetary_sum, f.avg_order_value, f.total_tickets,
               f.avg_resolution_hours, f.support_ticket_density, f.tenure_days, f.orders_per_month,
               f.orders_last_90d, f.avg_gap_days, c.geography
        FROM tbl_churn_predictions p
        JOIN tbl_feature_store f ON f.customer_id = p.customer_id
        JOIN tbl_customers c ON c.customer_id = p.customer_id
        WHERE p.execution_date = (SELECT MAX(execution_date) FROM tbl_churn_predictions)
    """)
    df = pd.read_sql_query(sql, _engine, parse_dates=["execution_date"])
    df["revenue_at_risk"] = (df["churn_probability"] * df["avg_order_value"] * (df["orders_per_month"] * 12)).round(2)
    return df


@st.cache_data(ttl=600, show_spinner=False)
def load_history(_engine, customer_id: str) -> pd.DataFrame:
    return pd.read_sql_query(
        text("SELECT execution_date, churn_probability, risk_category FROM tbl_churn_predictions "
             "WHERE customer_id = :cid ORDER BY execution_date"),
        _engine, params={"cid": customer_id}, parse_dates=["execution_date"])


def load_metrics() -> dict | None:
    try:
        return json.loads(config.METRICS_PATH.read_text())
    except Exception:
        return None


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _log_export(name: str, rows: int) -> None:
    AUDIT.info("EXPORT user=%s file=%s rows=%s", st.session_state.get("user"), name, rows)


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_overview(df: pd.DataFrame, engine) -> None:
    st.header("📊 Executive Overview")
    total_customers = db.table_count(engine, "tbl_customers")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers evaluated", f"{len(df):,}")
    c2.metric("High-risk customers", f"{int((df.risk_category == 'High').sum()):,}")
    c3.metric("Average churn probability", f"{df.churn_probability.mean():.1%}")
    c4.metric("Revenue at risk (High band)", f"{CUR}{df.loc[df.risk_category == 'High', 'revenue_at_risk'].sum():,.0f}")
    excluded = total_customers - len(df)
    if excluded > 0:
        st.caption(f"{excluded:,} customer(s) have no order history and are flagged *insufficient history* (cold-start rule).")

    left, right = st.columns([3, 2])
    fig = px.histogram(df, x="churn_probability", color="risk_category", nbins=25,
                       color_discrete_map=RISK_COLORS, title="Distribution of predicted churn probabilities",
                       category_orders={"risk_category": ["High", "Medium", "Low"]})
    fig.update_layout(bargap=0.05, xaxis_title="P(churn within 90 days)", yaxis_title="Customers")
    left.plotly_chart(fig, width="stretch")
    counts = df.risk_category.value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
    counts.columns = ["risk_category", "customers"]
    pie = px.pie(counts, names="risk_category", values="customers", hole=0.5, color="risk_category",
                 color_discrete_map=RISK_COLORS, title="Customers by risk band")
    right.plotly_chart(pie, width="stretch")

    st.subheader("Top churn drivers (High-risk customers)")
    drivers = df[df.risk_category == "High"].primary_churn_driver.value_counts().head(8).reset_index()
    drivers.columns = ["driver", "customers"]
    if len(drivers):
        bar = px.bar(drivers.sort_values("customers"), x="customers", y="driver", orientation="h")
        bar.update_layout(yaxis_title="", height=320)
        st.plotly_chart(bar, width="stretch")
    st.caption(f"Model: **{df.model_version.iloc[0]}** · scored on {df.execution_date.iloc[0]:%d %b %Y %H:%M} UTC")


def page_risk_table(df: pd.DataFrame) -> None:
    st.header("👥 Customer Risk Table & Retention Action Plan")
    c1, c2, c3 = st.columns([2, 2, 2])
    levels = c1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High"])
    geos = c2.multiselect("City / geography", sorted(df.geography.unique()))
    search = c3.text_input("Search customer ID")
    min_p = st.slider("Minimum churn probability", 0.0, 1.0, 0.0, 0.05)
    out = df[df.risk_category.isin(levels) & (df.churn_probability >= min_p)]
    if geos:
        out = out[out.geography.isin(geos)]
    if search:
        out = out[out.customer_id.str.contains(search.strip(), case=False, regex=False)]
    out = out.sort_values("revenue_at_risk", ascending=False)
    cols = ["customer_id", "churn_probability", "risk_category", "primary_churn_driver", "recommended_action",
            "revenue_at_risk", "recency_days", "frequency_count", "avg_order_value", "total_tickets", "geography"]
    st.write(f"**{len(out):,}** customers match · sorted by revenue at risk")
    st.dataframe(out[cols], width="stretch", hide_index=True, column_config={
        "churn_probability": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=1, format="%.2f"),
        "revenue_at_risk": st.column_config.NumberColumn(f"Revenue at risk ({CUR})", format="%.0f"),
    })
    st.download_button("⬇️ Download Retention Action Plan (CSV)", _csv_bytes(out[cols]),
                       file_name="retention_action_plan.csv", mime="text/csv",
                       on_click=_log_export, args=("retention_action_plan.csv", len(out)))


def page_customer(df: pd.DataFrame, engine, service) -> None:
    st.header("🔎 Customer Detail & Explanation")
    cid = st.selectbox("Customer", df.sort_values("churn_probability", ascending=False).customer_id.tolist())
    row = df[df.customer_id == cid].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{row.churn_probability:.1%}")
    c2.metric("Risk band", row.risk_category)
    c3.metric("Revenue at risk", f"{CUR}{row.revenue_at_risk:,.0f}")
    st.info(f"**Recommended action:** {row.recommended_action}  \n**Main driver:** {row.primary_churn_driver}")

    contrib = service.contributions(df[df.customer_id == cid][["customer_id"] + FEATURE_COLUMNS]).iloc[0]
    cdf = pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in contrib.index], "contribution": contrib.values})
    cdf = cdf.reindex(cdf.contribution.abs().sort_values().index)
    fig = px.bar(cdf, x="contribution", y="feature", orientation="h", title="What pushes this customer towards (+) or away from (−) churn",
                 color=cdf.contribution > 0, color_discrete_map={True: "#d62728", False: "#2ca02c"})
    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Contribution to churn score", height=420)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Customer profile")
    prof = pd.DataFrame({"Feature": [FEATURE_LABELS[f] for f in FEATURE_COLUMNS],
                         "Value": [round(float(row[f]), 2) for f in FEATURE_COLUMNS]})
    st.dataframe(prof, hide_index=True, width="stretch")
    hist = load_history(engine, cid)
    if len(hist) > 1:
        st.subheader("Prediction history")
        st.plotly_chart(px.line(hist, x="execution_date", y="churn_probability", markers=True), width="stretch")


def page_revenue(df: pd.DataFrame) -> None:
    st.header("💰 Revenue at Risk")
    st.caption("Revenue at risk = P(churn) × average order value × expected orders per year "
               "(a simple customer-value proxy, not a forecast).")
    by_band = df.groupby("risk_category").agg(customers=("customer_id", "count"),
                                              revenue_at_risk=("revenue_at_risk", "sum")).reindex(["High", "Medium", "Low"]).reset_index()
    c1, c2 = st.columns(2)
    c1.metric("Total revenue at risk", f"{CUR}{df.revenue_at_risk.sum():,.0f}")
    top_n = df.nlargest(100, "revenue_at_risk").revenue_at_risk.sum()
    c2.metric("Held by the top 100 customers", f"{CUR}{top_n:,.0f}", f"{top_n / max(df.revenue_at_risk.sum(), 1):.0%} of total")
    st.plotly_chart(px.bar(by_band, x="risk_category", y="revenue_at_risk", color="risk_category",
                           color_discrete_map=RISK_COLORS, title="Revenue at risk by band"), width="stretch")
    st.subheader("Top 20 customers to save first")
    top = df.nlargest(20, "revenue_at_risk")[["customer_id", "churn_probability", "risk_category",
                                              "revenue_at_risk", "primary_churn_driver", "recommended_action"]]
    st.dataframe(top, hide_index=True, width="stretch")
    st.download_button("⬇️ Download top-100 priority list (CSV)",
                       _csv_bytes(df.nlargest(100, "revenue_at_risk")), file_name="priority_customers.csv",
                       mime="text/csv", on_click=_log_export, args=("priority_customers.csv", 100))


def page_model(metrics: dict | None) -> None:
    st.header("🤖 Model Performance")
    if not metrics:
        st.warning("No metrics found. Train the model from the Admin page.")
        return
    st.caption(f"Best model: **{metrics['best_model']}** · trained {metrics['trained_at'][:16]} UTC · "
               f"{metrics['n_customers']:,} customers · churn rate {metrics['churn_rate']:.1%} · "
               f"label window ends {metrics.get('train_ref_date')} + 90 days")
    rows = []
    for name, m in metrics["models"].items():
        rows.append({"model": name, "CV ROC-AUC": m["cv_auc"], "Test ROC-AUC": m["test"]["roc_auc"],
                     "Precision": m["test"]["precision"], "Recall": m["test"]["recall"], "F1": m["test"]["f1"]})
    b = metrics["baseline_rule_30d"]
    rows.append({"model": "baseline: 30-day inactivity rule", "CV ROC-AUC": None, "Test ROC-AUC": b["roc_auc"],
                 "Precision": b["precision"], "Recall": b["recall"], "F1": b["f1"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    best = metrics["models"][metrics["best_model"]]["test"]
    tgt = metrics["acceptance_targets"]
    ok_auc, ok_rec = best["roc_auc"] >= tgt["roc_auc"], best["recall"] >= tgt["recall"]
    st.write(f"Acceptance targets: ROC-AUC ≥ {tgt['roc_auc']} → **{'met' if ok_auc else 'not met'}** "
             f"({best['roc_auc']}); churn recall ≥ {tgt['recall']} → **{'met' if ok_rec else 'not met'}** ({best['recall']}).")
    st.caption("Results are measured on a synthetic dataset; real-world performance must be re-validated.")

    c1, c2 = st.columns(2)
    fig = go.Figure()
    for name, cv in metrics["roc_curves"].items():
        fig.add_trace(go.Scatter(x=cv["fpr"], y=cv["tpr"], mode="lines", name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random", line=dict(dash="dash", color="grey")))
    fig.update_layout(title="ROC curves (hold-out test set)", xaxis_title="False-positive rate", yaxis_title="True-positive rate")
    c1.plotly_chart(fig, width="stretch")
    cm = metrics["confusion_matrix"]
    z = [[cm["tp"], cm["fp"]], [cm["fn"], cm["tn"]]]
    heat = go.Figure(go.Heatmap(z=z, x=["Actual churn", "Actual active"], y=["Predicted churn", "Predicted active"],
                                text=z, texttemplate="%{text}", colorscale="Blues", showscale=False))
    heat.update_layout(title=f"Confusion matrix – {metrics['best_model']}", yaxis_autorange="reversed")
    c2.plotly_chart(heat, width="stretch")

    imp = pd.DataFrame({"feature": [FEATURE_LABELS.get(k, k) for k in metrics["feature_importance"]],
                        "importance": list(metrics["feature_importance"].values())}).head(8)
    st.plotly_chart(px.bar(imp.sort_values("importance"), x="importance", y="feature", orientation="h",
                           title="Top churn drivers (mean |SHAP| on test set)"), width="stretch")


TEMPLATES = {
    "customers_template.csv": "customer_id,first_name,last_name,email,signup_timestamp,geography,gender\n"
                              "CUST-90001,Aarav,Sharma,aarav@example.com,2026-01-15 10:30:00,Delhi,Male\n",
    "orders_template.csv": "order_id,customer_id,order_timestamp,order_amount,item_count,payment_method\n"
                           "ORD-9000001,CUST-90001,2026-02-01 18:20:00,1499.00,2,UPI\n",
    "tickets_template.csv": "ticket_id,customer_id,created_at,issue_category,status,resolution_hours\n"
                            "TKT-900001,CUST-90001,2026-02-03 09:00:00,Delivery,Closed,12.5\n",
}


def page_upload(engine, service) -> None:
    st.header("📤 Upload Data & Score")
    st.write("Upload new customers, orders and/or support tickets. Files are validated first; "
             "if any file has a problem, **nothing is written**. Existing IDs are skipped (no duplicates).")
    with st.expander("Download CSV templates"):
        for name, content in TEMPLATES.items():
            st.download_button(name, content, file_name=name, mime="text/csv", key=name)
    cu = st.file_uploader("Customers CSV", type="csv", key="up_c")
    orf = st.file_uploader("Orders CSV", type="csv", key="up_o")
    tk = st.file_uploader("Support tickets CSV (optional)", type="csv", key="up_t")
    if st.button("Validate, import and re-score", type="primary", disabled=not (cu or orf or tk)):
        try:
            frames = [pd.read_csv(io.BytesIO(f.getvalue())) if f else None for f in (cu, orf, tk)]
            with st.spinner("Validating and importing..."):
                result = pipeline.ingest_frames(engine, *frames)
            for w in result["warnings"]:
                st.warning(w)
            st.success(f"Imported: {result['inserted']}")
            with st.spinner("Re-scoring all customers..."):
                summary = pipeline.refresh_features_and_predictions(engine, service)
            st.cache_data.clear()
            AUDIT.info("UPLOAD user=%s inserted=%s scored=%s", st.session_state.get("user"), result["inserted"], summary.get("scored"))
            st.success(f"Scored {summary['scored']:,} customers (reference date {summary.get('reference_date')}). "
                       "Open the other pages to see updated results.")
        except ValidationError as e:
            st.error("Upload rejected - nothing was written to the database:")
            for msg in e.errors:
                st.write(f"• {msg}")
        except Exception as e:  # unreadable file etc.
            st.error(f"Could not process the files: {e}")


def page_admin(engine, service) -> None:
    st.header("⚙️ Administration")
    st.subheader("Model management")
    c1, c2 = st.columns(2)
    fast = c1.checkbox("Fast mode (smaller hyper-parameter grids)", value=True)
    if c1.button("Retrain model and re-score"):
        with st.spinner("Training 5 models with 5-fold CV + SMOTE... (1-5 minutes)"):
            pipeline.train_model(engine, fast=fast, verbose=False)
            st.cache_resource.clear()
            st.cache_data.clear()
        AUDIT.info("RETRAIN user=%s fast=%s", st.session_state.get("user"), fast)
        st.success("Model retrained. Reloading...")
        time.sleep(1)
        st.rerun()
    if c2.button("Re-score only"):
        with st.spinner("Scoring..."):
            summary = pipeline.refresh_features_and_predictions(engine, service)
            st.cache_data.clear()
        AUDIT.info("RESCORE user=%s", st.session_state.get("user"))
        st.success(f"Scored {summary['scored']:,} customers.")

    st.subheader("Database")
    st.dataframe(pd.DataFrame({"table": [t.name for t in db.Base.metadata.sorted_tables],
                               "rows": [db.table_count(engine, t.name) for t in db.Base.metadata.sorted_tables]}),
                 hide_index=True, width="stretch")
    st.subheader("Users")
    st.dataframe(db.list_users(engine), hide_index=True, width="stretch")
    with st.form("new_user", clear_on_submit=True):
        u = st.text_input("New username")
        p = st.text_input("Password (min 8 chars)", type="password")
        r = st.selectbox("Role", ["Analyst", "Admin"])
        if st.form_submit_button("Create user"):
            try:
                db.create_user(engine, u.strip(), p, r)
                AUDIT.info("USER_CREATED by=%s new=%s role=%s", st.session_state.get("user"), u, r)
                st.success(f"User '{u}' created.")
            except ValueError as e:
                st.error(str(e))
    others = [x for x in db.list_users(engine).username if x != st.session_state.get("user")]
    if others:
        with st.form("del_user"):
            victim = st.selectbox("Delete user", others)
            if st.form_submit_button("Delete"):
                db.delete_user(engine, victim)
                AUDIT.info("USER_DELETED by=%s user=%s", st.session_state.get("user"), victim)
                st.success(f"User '{victim}' deleted.")
                st.rerun()


# --------------------------------------------------------------------------
def render_app(engine, service) -> None:
    role = st.session_state.get("role", "Analyst")
    pages = ["Overview", "Customer Risk Table", "Customer Detail", "Revenue at Risk", "Model Performance", "Upload & Score"]
    if role == "Admin":
        pages.append("Admin")
    with st.sidebar:
        st.title("Churn Analytics")
        st.caption(f"Signed in as **{st.session_state.get('user')}** ({role})")
        page = st.radio("Navigate", pages, label_visibility="collapsed")
        if st.button("Sign out"):
            AUDIT.info("LOGOUT user=%s", st.session_state.get("user"))
            for k in ("auth", "user", "role"):
                st.session_state.pop(k, None)
            st.rerun()
    df = load_scored(engine)
    if df.empty and page not in ("Upload & Score", "Admin", "Model Performance"):
        st.warning("No predictions yet. Open **Upload & Score** or **Admin** to import data and score customers.")
        return
    if page == "Overview":
        page_overview(df, engine)
    elif page == "Customer Risk Table":
        page_risk_table(df)
    elif page == "Customer Detail":
        page_customer(df, engine, service)
    elif page == "Revenue at Risk":
        page_revenue(df)
    elif page == "Model Performance":
        page_model(load_metrics())
    elif page == "Upload & Score":
        page_upload(engine, service)
    elif page == "Admin" and role == "Admin":
        page_admin(engine, service)
