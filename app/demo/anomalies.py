import numpy as np
import pandas as pd

from app.demo import engine, store
from app.demo.snapshot import Snapshot, is_usage_kind
from app.ml import CONTAMINATION, FEATURES, N_ESTIMATORS, RANDOM_STATE, score_records, train_isolation_forest

REPORT_WINDOW_DAYS = 30
MIN_COST_RATIO = 1.10
MIN_TRAINING_RECORDS = 200
STATUS_TRANSITIONS = {
    "acknowledge": ({"open"}, "acknowledged"),
    "investigate": ({"open", "acknowledged"}, "investigated"),
    "dismiss": ({"open", "acknowledged", "investigated"}, "dismissed"),
    "reopen": ({"acknowledged", "investigated", "dismissed"}, "open"),
}
ACTIVE_STATUSES = ("open", "acknowledged")


def build_daily_features(snap: Snapshot) -> pd.DataFrame:
    daily = snap.daily_resource
    scored_kinds = daily["kind"].map(is_usage_kind)
    complete_day = daily["date"] < snap.today
    frame = daily[scored_kinds & complete_day & (daily["running_hours"] > 0)].copy()
    resource_cost = frame.groupby("resource_id")["cost"]
    frame["svc_median"] = resource_cost.transform("median")
    frame["svc_std"] = np.maximum(resource_cost.transform("std").fillna(0), 0.05 * frame["svc_median"])
    frame["cost_ratio"] = frame["cost"] / frame["svc_median"]
    frame["cost_z"] = (frame["cost"] - frame["svc_median"]) / frame["svc_std"]
    frame["cpu_avg_pct"] = frame["cpu"]
    frame["memory_avg_pct"] = frame["mem"]
    frame["usage_hours"] = frame["active_hours"]
    frame["dayofweek"] = frame["date"].dt.dayofweek
    typical = frame.groupby("resource_id")[["cpu", "mem", "active_hours", "quantity"]].transform("median")
    frame[["typical_cpu", "typical_mem", "typical_active_hours", "typical_quantity"]] = typical.values
    return frame.reset_index(drop=True)


def detect(snap: Snapshot) -> pd.DataFrame:
    cached = getattr(snap, "_anomaly_result", None)
    if cached is not None:
        return cached
    features = build_daily_features(snap)
    if len(features) < MIN_TRAINING_RECORDS:
        result = features.iloc[0:0].assign(iforest_pred=0, anomaly_score=0.0, reported=False)
    else:
        model = train_isolation_forest(features)
        result = score_records(model, features)
        window_start = snap.today - pd.Timedelta(days=REPORT_WINDOW_DAYS)
        result["reported"] = (result["iforest_pred"] == 1) & (result["cost_ratio"] >= MIN_COST_RATIO) \
            & (result["date"] >= window_start)
    snap._anomaly_result = result
    return result


def severity(cost_ratio: float) -> str:
    if cost_ratio >= 2.0:
        return "critical"
    if cost_ratio >= 1.5:
        return "high"
    return "medium"


def explanations(row: pd.Series) -> list:
    notes = []
    if row["typical_quantity"] > 0 and row["quantity"] > 1.3 * row["typical_quantity"]:
        notes.append(f"Billed instance-hours were {row['quantity']:.0f} against a typical {row['typical_quantity']:.0f}, "
                     "so more capacity than usual was running (for example a scale-out).")
    if row["cpu"] > 1.2 * row["typical_cpu"]:
        notes.append(f"CPU averaged {row['cpu']:.1f}% against a typical {row['typical_cpu']:.1f}%, so the extra "
                     "cost came with extra workload.")
    elif row["cost_ratio"] >= MIN_COST_RATIO:
        notes.append(f"CPU ({row['cpu']:.1f}%) stayed close to its typical {row['typical_cpu']:.1f}%, which points "
                     "to a capacity or configuration change rather than higher demand.")
    if row["active_hours"] > row["typical_active_hours"] + 3:
        notes.append(f"The resource was busy for {row['active_hours']:.0f} hours against a typical "
                     f"{row['typical_active_hours']:.0f}.")
    return notes


def group_incidents(reported: pd.DataFrame) -> list:
    reported = reported.sort_values(["resource_id", "date"])
    new_run = reported.groupby("resource_id")["date"].diff().dt.days.ne(1)
    incidents = []
    for _, group in reported.groupby(new_run.cumsum()):
        first = group.iloc[0].copy()
        first["anomaly_score"] = group["anomaly_score"].max()
        first["days_flagged"] = len(group)
        first["last_date"] = group["date"].max()
        first["total_excess"] = float((group["cost"] - group["svc_median"]).sum())
        incidents.append(first)
    incidents.sort(key=lambda row: (row["last_date"], row["anomaly_score"]), reverse=True)
    return incidents


def to_item(row: pd.Series, review: dict) -> dict:
    anomaly_id = f"{row['resource_id']}:{row['date']:%Y-%m-%d}"
    return {
        "days_flagged": int(row["days_flagged"]), "last_date": f"{row['last_date']:%Y-%m-%d}",
        "total_excess_cost": float(row["total_excess"]),
        "anomaly_id": anomaly_id, "resource_id": row["resource_id"], "name": row["name"],
        "provider": row["provider"], "service": row["service"], "environment": row["environment"],
        "team": row["team"], "date": f"{row['date']:%Y-%m-%d}", "observed_cost": float(row["cost"]),
        "baseline_cost": float(row["svc_median"]), "excess_cost": float(row["cost"] - row["svc_median"]),
        "cost_ratio": float(row["cost_ratio"]), "cost_z": float(row["cost_z"]),
        "anomaly_score": float(row["anomaly_score"]), "severity": severity(row["cost_ratio"]),
        "cpu": float(row["cpu"]), "typical_cpu": float(row["typical_cpu"]), "mem": float(row["mem"]),
        "typical_mem": float(row["typical_mem"]), "active_hours": float(row["active_hours"]),
        "typical_active_hours": float(row["typical_active_hours"]), "instance_hours": float(row["quantity"]),
        "typical_instance_hours": float(row["typical_quantity"]),
        "status": review["status"], "note": review.get("note"), "detected_at": review["first_detected_at"],
        "updated_at": review.get("updated_at"), "possible_explanations": explanations(row),
    }


def sync_reviews(conn, snap: Snapshot, incidents: list) -> dict:
    detected_at = engine.to_ts(snap.clock)
    for row in incidents:
        conn.execute("INSERT OR IGNORE INTO anomaly_reviews (anomaly_id, resource_id, day, status, first_detected_at)"
                     " VALUES (?, ?, ?, 'open', ?)",
                     (f"{row['resource_id']}:{row['date']:%Y-%m-%d}", row["resource_id"], f"{row['date']:%Y-%m-%d}",
                      detected_at))
    conn.commit()
    return {row["anomaly_id"]: dict(row) for row in conn.execute("SELECT * FROM anomaly_reviews")}


def list_anomalies(conn, snap: Snapshot) -> dict:
    result = detect(snap)
    incidents = group_incidents(result[result["reported"]])
    reviews = sync_reviews(conn, snap, incidents)
    items = [to_item(row, reviews[f"{row['resource_id']}:{row['date']:%Y-%m-%d}"]) for row in incidents]
    counts = {status: sum(1 for i in items if i["status"] == status)
              for status in ("open", "acknowledged", "investigated", "dismissed")}
    return {
        "items": items,
        "counts": counts,
        "active": sum(1 for i in items if i["status"] in ACTIVE_STATUSES),
        "model": {
            "name": "Isolation Forest", "n_estimators": N_ESTIMATORS, "contamination": CONTAMINATION,
            "random_state": RANDOM_STATE, "features": FEATURES, "records_scored": int(len(result)),
            "flagged_by_model": int(result["iforest_pred"].sum()) if len(result) else 0,
            "report_rule": f"Reported when the model flags a daily resource record from the last {REPORT_WINDOW_DAYS} "
                           f"days and its cost is at least {MIN_COST_RATIO:.2f}x the resource's median daily cost.",
            "training_data": "Complete days of simulated usage for compute, database and Kubernetes resources",
        },
    }


def anomaly_detail(conn, snap: Snapshot, anomaly_id: str) -> dict:
    listing = list_anomalies(conn, snap)
    item = next((i for i in listing["items"] if i["anomaly_id"] == anomaly_id), None)
    if item is None:
        raise engine.SimulationError(f"Anomaly '{anomaly_id}' not found in the current detection results", 404)
    daily = snap.daily_resource
    history = daily[daily["resource_id"] == item["resource_id"]].tail(45)
    item["history"] = [{"date": f"{r.date:%Y-%m-%d}", "cost": float(r.cost), "cpu": None if pd.isna(r.cpu) else float(r.cpu)}
                       for r in history.itertuples()]
    return item


def update_status(conn, snap: Snapshot, anomaly_id: str, action: str, note: str = None) -> dict:
    item = anomaly_detail(conn, snap, anomaly_id)
    allowed_from, new_status = STATUS_TRANSITIONS[action]
    if item["status"] not in allowed_from:
        raise engine.SimulationError(f"Cannot {action} an anomaly that is {item['status']}")
    conn.execute("UPDATE anomaly_reviews SET status = ?, note = COALESCE(?, note), updated_at = ? WHERE anomaly_id = ?",
                 (new_status, note, engine.to_ts(snap.clock), anomaly_id))
    store.bump_version(conn, simulation_changed=False)
    return {"anomaly_id": anomaly_id, "previous_status": item["status"], "status": new_status,
            "note": note if note is not None else item["note"], "updated_at": engine.to_ts(snap.clock)}
