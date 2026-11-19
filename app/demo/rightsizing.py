from datetime import timedelta

import pandas as pd

from app.demo import engine, store
from app.demo.snapshot import Snapshot, is_usage_kind
from app.ml import SAVINGS_FRACTION, recommend

WINDOW_HOURS = 168
MIN_HOURS_OF_DATA = 24
SNOOZE_DAYS = 7
DECISION_TRANSITIONS = {
    "review": ({"open"}, "in_review"),
    "approve": ({"open", "in_review"}, "approved"),
    "reject": ({"open", "in_review", "approved"}, "rejected"),
    "snooze": ({"open", "in_review"}, "snoozed"),
    "reopen": ({"rejected", "snoozed", "approved", "in_review"}, "open"),
}


def evaluation_start(resource: pd.Series, clock) -> pd.Timestamp:
    candidates = [clock - timedelta(hours=WINDOW_HOURS), engine.parse_ts(resource["launched_at"])]
    for column in ("resized_at", "schedule_restored_at"):
        if isinstance(resource.get(column), str):
            candidates.append(engine.parse_ts(resource[column]))
    if resource["status"] == "running" and isinstance(resource.get("started_at"), str):
        candidates.append(engine.parse_ts(resource["started_at"]))
    return pd.Timestamp(max(candidates))


def rule_checks(avg_cpu: float, avg_mem: float, avg_hours: float) -> list:
    return [
        {"rule": "STOP_REVIEW", "condition": "CPU < 5%", "value": f"{avg_cpu:.1f}%", "passed": avg_cpu < 5},
        {"rule": "STOP_REVIEW", "condition": "Memory < 10%", "value": f"{avg_mem:.1f}%", "passed": avg_mem < 10},
        {"rule": "STOP_REVIEW", "condition": "Active < 9 h/day", "value": f"{avg_hours:.1f} h", "passed": avg_hours < 9},
        {"rule": "DOWNSIZE", "condition": "CPU < 20%", "value": f"{avg_cpu:.1f}%", "passed": avg_cpu < 20},
        {"rule": "DOWNSIZE", "condition": "Memory < 30%", "value": f"{avg_mem:.1f}%", "passed": avg_mem < 30},
        {"rule": "DOWNSIZE", "condition": "Active >= 9 h/day", "value": f"{avg_hours:.1f} h", "passed": avg_hours >= 9},
    ]


def explain(action: str, cpu: float, mem: float, hours: float, days: float) -> str:
    period = f"Over the last {days:.1f} days" if days < 6.95 else "Over the last 7 days"
    if action == "STOP_REVIEW":
        return (f"{period} it averaged {cpu:.1f}% CPU and {mem:.1f}% memory and did useful work for only "
                f"{hours:.1f} h/day. All three stop-review thresholds are met (CPU < 5%, memory < 10%, "
                "fewer than 9 active hours per day), so it looks idle.")
    if action == "DOWNSIZE":
        return (f"{period} it was in use {hours:.1f} h/day but averaged only {cpu:.1f}% CPU and {mem:.1f}% "
                "memory, below the downsize thresholds (CPU < 20%, memory < 30%). A smaller size should "
                "handle the same work.")
    return (f"{period} utilization was {cpu:.1f}% CPU, {mem:.1f}% memory and {hours:.1f} active h/day, which "
            "does not meet the downsize or stop-review thresholds.")


def risk_note(action: str, resource: pd.Series, p95_cpu: float, disk_monthly: float, days: float) -> str:
    notes = []
    if action == "DOWNSIZE":
        notes.append(f"Peak CPU (95th percentile) was {p95_cpu:.0f}%; short peaks must still fit on the "
                     "smaller size. Resizing needs a restart or maintenance window.")
    elif action == "STOP_REVIEW":
        if resource["kind"] == "job":
            notes.append("This host runs a scheduled job. Applying the action restores the automatic "
                         "start/stop schedule instead of switching the job off.")
        else:
            notes.append(f"Confirm with the {resource['team']} team that nothing depends on it before stopping.")
        if disk_monthly > 0:
            notes.append(f"Attached disk storage (about INR {disk_monthly:,.0f}/month) keeps billing after a stop.")
    if days < 6.95:
        notes.append(f"Based on only {days:.1f} days of data since the last change.")
    return " ".join(notes)


def evaluate_resources(snap: Snapshot) -> pd.DataFrame:
    cached = getattr(snap, "_rightsizing_result", None)
    if cached is None:
        cached = snap._rightsizing_result = _evaluate_resources(snap)
    return cached


def _evaluate_resources(snap: Snapshot) -> pd.DataFrame:
    clock = pd.Timestamp(snap.clock)
    usage = snap.usage[snap.usage["charge"] == "usage"]
    records = []
    for _, resource in snap.resources.iterrows():
        if not is_usage_kind(resource["kind"]):
            continue
        base = {"resource_id": resource["resource_id"], "status": resource["status"]}
        if resource["status"] == "stopped":
            records.append({**base, "action": None, "note": "Stopped - no recommendation"})
            continue
        start = evaluation_start(resource, clock)
        hours_available = (clock - start).total_seconds() / 3600
        rows = usage[(usage["resource_id"] == resource["resource_id"]) & (usage["ts"] >= start)]
        if hours_available < MIN_HOURS_OF_DATA or rows.empty:
            records.append({**base, "action": None,
                            "note": "Waiting for at least 24 hours of usage data after the last change"})
            continue
        days = hours_available / 24
        avg_cpu, avg_mem = float(rows["cpu"].mean()), float(rows["mem"].mean())
        avg_hours = float(rows["active"].sum()) / days
        action = recommend(avg_cpu, avg_mem, avg_hours)
        monthly = snap.expected_month.loc[resource["resource_id"]]
        p95_cpu = float(rows["cpu"].quantile(0.95))
        records.append({
            **base, "action": action, "note": None, "window_days": round(days, 2),
            "avg_cpu": avg_cpu, "avg_mem": avg_mem, "avg_hours": avg_hours, "p95_cpu": p95_cpu,
            "monthly_cost": float(monthly["total"]), "disk_monthly": float(monthly["disk"]),
            "estimated_saving": float(monthly["total"]) * SAVINGS_FRACTION[action],
            "savings_fraction": SAVINGS_FRACTION[action],
            "reason": explain(action, avg_cpu, avg_mem, avg_hours, days),
            "risk": risk_note(action, resource, p95_cpu, float(monthly["disk"]), days),
            "checks": rule_checks(avg_cpu, avg_mem, avg_hours),
        })
    return pd.DataFrame(records).set_index("resource_id", drop=False)


def load_decisions(conn) -> dict:
    return {row["resource_id"]: dict(row) for row in conn.execute("SELECT * FROM decisions")}


def decision_status(decision: dict, action: str, clock) -> str:
    if not decision or decision["action"] != action or decision["status"] == "applied":
        return "open"
    if decision["status"] == "snoozed" and decision["snoozed_until"] and \
            engine.parse_ts(decision["snoozed_until"]) <= clock:
        return "open"
    return decision["status"]


def realized_savings(snap: Snapshot) -> list:
    results = []
    actions = snap.actions
    usage = snap.usage[snap.usage["charge"] == "usage"]
    for _, action in actions.iterrows():
        if action["action"] not in ("stop", "downsize", "restore_schedule"):
            continue
        start = pd.Timestamp(engine.parse_ts(action["ts"]))
        later = actions[(actions["resource_id"] == action["resource_id"]) & (actions["action_id"] > action["action_id"])]
        end = pd.Timestamp(engine.parse_ts(later.iloc[0]["ts"])) if not later.empty else pd.Timestamp(snap.clock)
        hours = (end - start).total_seconds() / 3600
        rows = usage[(usage["resource_id"] == action["resource_id"]) & (usage["ts"] >= start) & (usage["ts"] < end)]
        before = float(action["before_hourly"] or 0)
        realized = before * hours - float(rows["cost"].sum())
        still_effective = later.empty
        expected_usage = float(snap.expected_month.loc[action["resource_id"], "usage"])
        resource = snap.resources.loc[action["resource_id"]]
        results.append({
            "action_id": int(action["action_id"]), "resource_id": action["resource_id"], "name": resource["name"],
            "action": action["action"], "source": action["source"], "applied_at": action["ts"],
            "note": action["note"], "hours_since": hours, "before_hourly": before,
            "proposed_monthly": None if pd.isna(action["proposed_monthly"]) else float(action["proposed_monthly"]),
            "realized_to_date": realized,
            "monthly_run_rate_saving": max(before * engine.HOURS_PER_MONTH - expected_usage, 0) if still_effective else 0.0,
            "still_effective": still_effective,
        })
    return results


def build_recommendations(conn, snap: Snapshot) -> dict:
    evaluated = evaluate_resources(snap)
    decisions = load_decisions(conn)
    clock = snap.clock
    items = []
    for resource_id, row in evaluated.iterrows():
        resource = snap.resources.loc[resource_id]
        if row["action"] is None or pd.isna(row["action"]):
            continue
        decision = decisions.get(resource_id)
        status = decision_status(decision, row["action"], clock) if row["action"] != "RETAIN" else "not_required"
        started = resource["started_at"]
        running_hours = (clock - engine.parse_ts(started)).total_seconds() / 3600 if isinstance(started, str) else 0.0
        items.append({
            "resource_id": resource_id, "name": resource["name"], "provider": resource["provider"],
            "service": resource["service"], "environment": resource["environment"], "team": resource["team"],
            "kind": resource["kind"], "resource_class": resource["resource_class"],
            "instances": int(resource["instances"]), "status": resource["status"],
            "running_hours": running_hours, "action": row["action"], "avg_cpu": row["avg_cpu"],
            "avg_mem": row["avg_mem"], "avg_hours": row["avg_hours"], "p95_cpu": row["p95_cpu"],
            "window_days": row["window_days"], "monthly_cost": row["monthly_cost"],
            "estimated_saving": row["estimated_saving"], "savings_fraction": row["savings_fraction"],
            "reason": row["reason"], "risk": row["risk"], "checks": row["checks"], "decision_status": status,
            "decision_note": decision["note"] if decision and status != "open" else None,
            "decision_updated_at": decision["updated_at"] if decision and status != "open" else None,
            "snoozed_until": decision["snoozed_until"] if decision and status == "snoozed" else None,
        })
    order = {"STOP_REVIEW": 0, "DOWNSIZE": 1, "RETAIN": 2}
    items.sort(key=lambda item: (order[item["action"]], -item["estimated_saving"]))
    waiting = [{"resource_id": rid, "name": snap.resources.loc[rid, "name"], "note": row["note"]}
               for rid, row in evaluated.iterrows() if row["action"] is None or pd.isna(row["action"])]
    realized = realized_savings(snap)
    open_items = [i for i in items if i["action"] != "RETAIN" and i["decision_status"] in ("open", "in_review", "approved")]
    return {
        "items": items,
        "not_evaluated": waiting,
        "applied_actions": realized,
        "summary": {
            "evaluated": len(items),
            "actionable": sum(1 for i in items if i["action"] != "RETAIN"),
            "potential_monthly_savings": sum(i["estimated_saving"] for i in open_items),
            "approved_awaiting_apply": sum(i["estimated_saving"] for i in items if i["decision_status"] == "approved"),
            "realized_monthly_run_rate": sum(a["monthly_run_rate_saving"] for a in realized),
            "realized_to_date": sum(a["realized_to_date"] for a in realized),
        },
        "policy": {
            "stop_review": "CPU < 5% and memory < 10% and fewer than 9 active hours per day",
            "downsize": "CPU < 20% and memory < 30% and at least 9 active hours per day",
            "savings_fraction": SAVINGS_FRACTION,
            "window": "Last 7 days, or the time since the last start / resize if shorter",
        },
    }


def find_item(conn, snap: Snapshot, resource_id: str) -> dict:
    for item in build_recommendations(conn, snap)["items"]:
        if item["resource_id"] == resource_id:
            return item
    if resource_id not in snap.resources.index:
        raise engine.SimulationError(f"Resource '{resource_id}' not found", 404)
    raise engine.SimulationError(f"There is no current recommendation for '{resource_id}'")


def record_decision(conn, snap: Snapshot, resource_id: str, decision: str, note: str = None) -> dict:
    item = find_item(conn, snap, resource_id)
    if item["action"] == "RETAIN":
        raise engine.SimulationError("RETAIN recommendations do not need a decision")
    allowed_from, new_status = DECISION_TRANSITIONS[decision]
    if item["decision_status"] not in allowed_from:
        raise engine.SimulationError(f"Cannot {decision} a recommendation that is {item['decision_status']}")
    clock = snap.clock
    snoozed_until = engine.to_ts(clock + timedelta(days=SNOOZE_DAYS)) if decision == "snooze" else None
    conn.execute("INSERT INTO decisions (resource_id, action, status, note, updated_at, snoozed_until)"
                 " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(resource_id) DO UPDATE SET action = excluded.action,"
                 " status = excluded.status, note = excluded.note, updated_at = excluded.updated_at,"
                 " snoozed_until = excluded.snoozed_until",
                 (resource_id, item["action"], new_status, note, engine.to_ts(clock), snoozed_until))
    store.bump_version(conn, simulation_changed=False)
    return {"resource_id": resource_id, "action": item["action"], "previous_status": item["decision_status"],
            "status": new_status, "note": note, "updated_at": engine.to_ts(clock), "snoozed_until": snoozed_until,
            "infrastructure_changed": False}


def apply_recommendation(conn, snap: Snapshot, resource_id: str) -> dict:
    item = find_item(conn, snap, resource_id)
    if item["decision_status"] != "approved":
        raise engine.SimulationError("Only approved recommendations can be applied in the simulation")
    if item["action"] == "DOWNSIZE":
        engine.apply_downsize(conn, resource_id)
        message = "Resource resized in the simulation; future hours are billed at the smaller size."
    elif item["kind"] == "job":
        engine.restore_schedule(conn, resource_id)
        message = "Job host returned to its schedule; it now runs only during the job window."
    else:
        engine.change_power_state(conn, resource_id, "stop", source="recommendation")
        message = "Resource stopped in the simulation; compute billing stops, attached disk keeps billing."
    conn.execute("UPDATE actions SET proposed_monthly = ? WHERE action_id = (SELECT MAX(action_id) FROM actions)",
                 (item["estimated_saving"],))
    conn.execute("UPDATE decisions SET status = 'applied', updated_at = ? WHERE resource_id = ?",
                 (engine.to_ts(snap.clock), resource_id))
    store.bump_version(conn, simulation_changed=True)
    return {"resource_id": resource_id, "action": item["action"], "status": "applied", "message": message,
            "estimated_monthly_saving": item["estimated_saving"], "simulated_only": True}
