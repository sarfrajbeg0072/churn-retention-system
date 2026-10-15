"""Documented synthetic data generator (customers, orders, support tickets).

Behaviour model (so the ML task is realistic but not trivial):
* every customer has a latent *satisfaction* score and a *purchase rate*;
* low satisfaction -> shorter lifetime, more / slower support tickets;
* before a customer leaves, their order rate fades out;
* "occasional" buyers have long natural gaps (they look like churners but are not).

Usage:  python -m src.data_generator --customers 6000 --out data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FIRST = ["Aarav", "Vivaan", "Aditya", "Arjun", "Rohan", "Karan", "Rahul", "Amit", "Sanjay", "Imran",
         "Priya", "Ananya", "Isha", "Neha", "Pooja", "Sneha", "Kavya", "Riya", "Meera", "Fatima"]
LAST = ["Sharma", "Verma", "Gupta", "Khan", "Singh", "Patel", "Reddy", "Iyer", "Das", "Mehta",
        "Joshi", "Nair", "Ansari", "Beg", "Yadav", "Chopra", "Malhotra", "Bose", "Kapoor", "Rao"]
CITIES = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
PAYMENTS = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash on Delivery", "Wallet"]
ISSUES = ["Delivery", "Refund", "Payment", "Product Quality", "Account", "Other"]


def generate_synthetic_data(n_customers: int = 6000, seed: int = 42,
                            end_date: str = "2026-09-30", horizon_days: int = 1000):
    rng = np.random.default_rng(seed)
    horizon_end = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1)
    start = horizon_end - pd.Timedelta(days=horizon_days)

    cust_rows, order_rows, ticket_rows = [], [], []
    order_seq = ticket_seq = 0

    seg_rate = np.array([2.2, 1.0, 0.45])  # orders / month for loyal, regular, occasional
    for i in range(n_customers):
        cid = f"CUST-{i + 1:05d}"
        signup_off = rng.uniform(0, horizon_days - 15)
        signup = start + pd.Timedelta(days=float(signup_off))
        seg = rng.choice(3, p=[0.30, 0.45, 0.25])
        rate = float(seg_rate[seg] * rng.lognormal(0, 0.35))
        sat = float(rng.beta(2.2, 2.2))
        lifetime = float(rng.exponential(500 + 2500 * sat ** 1.2))
        span = min(lifetime, (horizon_end - signup).total_seconds() / 86400.0)
        churn_day = lifetime if lifetime < (horizon_end - signup).days else None

        fn, ln = FIRST[rng.integers(len(FIRST))], LAST[rng.integers(len(LAST))]
        cust_rows.append({
            "customer_id": cid, "first_name": fn, "last_name": ln,
            "email": f"{fn.lower()}.{ln.lower()}{i + 1}@example.com",
            "signup_timestamp": signup.floor("s"),
            "geography": CITIES[rng.integers(len(CITIES))],
            "gender": rng.choice(["Male", "Female", "Other"], p=[0.48, 0.48, 0.04]),
        })

        # --- orders: Poisson-like process with fade-out before churn ---------
        first = float(rng.exponential(3.0))
        n_est = int(rate / 30 * span * 2) + 8
        times = first + np.concatenate([[0.0], np.cumsum(rng.exponential(30.0 / rate, n_est))])
        times = times[times < span]
        if churn_day is not None and len(times) > 1:
            fade_start = churn_day - 60
            keep_p = np.where(times < fade_start, 1.0,
                              1.0 - 0.75 * np.clip((times - fade_start) / 60.0, 0, 1))
            keep = rng.random(len(times)) < keep_p
            keep[0] = True
            times = times[keep]
        mu = float(rng.lognormal(np.log(1400), 0.5))
        order_times = []
        for t in times:
            ts = (signup + pd.Timedelta(days=float(t))).floor("s")
            if ts >= horizon_end:
                continue
            amount = round(mu * float(rng.lognormal(0, 0.35)), 2)
            order_seq += 1
            order_rows.append({
                "order_id": f"ORD-{order_seq:07d}", "customer_id": cid, "order_timestamp": ts,
                "order_amount": amount, "item_count": int(min(12, 1 + rng.poisson(amount / 900))),
                "payment_method": PAYMENTS[rng.integers(len(PAYMENTS))],
            })
            order_times.append(ts)

        # --- support tickets: unhappy customers raise more, slower-resolved ---
        lam = 0.15 * len(order_times) * (0.4 + 1.8 * (1 - sat))
        n_tix = int(rng.poisson(lam))
        tix_times = [order_times[rng.integers(len(order_times))] + pd.Timedelta(hours=float(rng.uniform(1, 48)))
                     for _ in range(n_tix)] if order_times else []
        if churn_day is not None and order_times and rng.random() < 0.5 * (1 - sat) + 0.1:
            tix_times.append(signup + pd.Timedelta(days=max(0.0, churn_day - float(rng.uniform(5, 45)))))
        for ts in tix_times:
            ts = pd.Timestamp(ts).floor("s")
            if ts >= horizon_end:
                continue
            hours = float(rng.gamma(2.0, 6 + 30 * (1 - sat)))
            r = rng.random()
            status = "Escalated" if (hours > 96 and r < 0.7) else ("Open" if r > 0.96 else "Closed")
            ticket_seq += 1
            ticket_rows.append({
                "ticket_id": f"TKT-{ticket_seq:06d}", "customer_id": cid, "created_at": ts,
                "issue_category": ISSUES[rng.integers(len(ISSUES))], "status": status,
                "resolution_hours": 0.0 if status == "Open" else round(hours, 1),
            })

    customers = pd.DataFrame(cust_rows)
    orders = pd.DataFrame(order_rows)
    tickets = pd.DataFrame(ticket_rows)
    return customers, orders, tickets


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic e-commerce data")
    p.add_argument("--customers", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--end-date", default="2026-09-30")
    p.add_argument("--out", default="data")
    a = p.parse_args()
    c, o, t = generate_synthetic_data(a.customers, a.seed, a.end_date)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c.to_csv(out / "customers.csv", index=False)
    o.to_csv(out / "orders.csv", index=False)
    t.to_csv(out / "support_tickets.csv", index=False)
    print(f"customers={len(c):,} orders={len(o):,} tickets={len(t):,} -> {out.resolve()}")


if __name__ == "__main__":
    main()
