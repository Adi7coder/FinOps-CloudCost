from datetime import date, timedelta

import pandas as pd

from app.demo import anomalies, catalog, engine, forecasting, rightsizing, store
from app.demo.snapshot import Snapshot, is_usage_kind

FILTER_COLUMNS = ("provider", "service", "team", "environment", "resource_id", "category")


def round_money(value: float) -> float:
    return round(float(value), 2)


def humanize_hours(hours: float) -> str:
    if hours >= 48:
        return f"{hours / 24:.0f} days"
    if hours >= 1:
        return f"{hours:.0f} h"
    return "under 1 h"


def grouped(frame: pd.DataFrame, column: str, limit: int = None) -> list:
    totals = frame.groupby(column)["cost"].sum().sort_values(ascending=False)
    if limit:
        totals = totals.head(limit)
    return [{column: key, "cost": round_money(value)} for key, value in totals.items()]


def state_summary(conn) -> dict:
    clock = engine.current_clock(conn)
    scenario_key = store.get_setting(conn, "scenario")
    scenario = catalog.SCENARIOS[scenario_key]
    return {
        "demo_mode": True,
        "notice": "Demo mode: simulated cloud resources for a fictional company. No AWS, Azure or GCP account is "
                  "connected and no real resource is changed.",
        "company": catalog.COMPANY_NAME, "company_description": catalog.COMPANY_DESCRIPTION,
        "timezone": catalog.TIMEZONE_LABEL, "clock": engine.to_ts(clock),
        "history_start": store.get_setting(conn, "history_start"),
        "scenario": scenario_key, "scenario_title": scenario["title"], "scenario_summary": scenario["summary"],
        "monthly_budget": store.get_setting(conn, "monthly_budget"),
        "state_version": store.get_setting(conn, "state_version", 0),
        "sim_version": store.get_setting(conn, "sim_version", 0),
        "active_events": [{"resource_id": e["resource_id"], "label": e["label"], "start": e["start_ts"],
                           "end": e["end_ts"]} for e in engine.load_events(conn) if e["end"] > clock],
        "scenarios": [{"key": key, "title": value["title"], "summary": value["summary"]}
                      for key, value in catalog.SCENARIOS.items()],
        "spending_events": [{"key": key, "label": value["label"]} for key, value in catalog.SPENDING_EVENTS.items()],
        "pricing_assumptions": catalog.PRICING_ASSUMPTIONS,
    }


def filter_options(snap: Snapshot) -> dict:
    resources = snap.resources
    return {
        "providers": sorted(resources["provider"].unique()), "services": sorted(resources["service"].unique()),
        "teams": sorted(resources["team"].unique()), "environments": sorted(resources["environment"].unique()),
        "categories": sorted(snap.usage["category"].unique()),
        "resources": [{"resource_id": rid, "name": name} for rid, name in resources["name"].sort_values().items()],
        "first_date": f"{snap.history_start:%Y-%m-%d}", "last_date": f"{snap.today:%Y-%m-%d}",
    }


def apply_filters(frame: pd.DataFrame, filters: dict) -> pd.DataFrame:
    for column in FILTER_COLUMNS:
        value = filters.get(column)
        if value:
            frame = frame[frame[column] == value]
    return frame


def explore_costs(snap: Snapshot, start: date, end: date, filters: dict) -> dict:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    usage = apply_filters(snap.usage, filters)
    frame = usage[(usage["date"] >= start_ts) & (usage["date"] <= end_ts)]
    days = (end_ts - start_ts).days + 1
    previous_start = start_ts - pd.Timedelta(days=days)
    previous = usage[(usage["date"] >= previous_start) & (usage["date"] < start_ts)]

    service_now = frame.groupby("service")["cost"].sum()
    service_before = previous.groupby("service")["cost"].sum()
    services = pd.concat([service_now.rename("cost"), service_before.rename("previous_cost")], axis=1).fillna(0)
    services["change"] = services["cost"] - services["previous_cost"]
    services = services.sort_values("cost", ascending=False)

    by_resource = frame.groupby(["resource_id", "name", "provider", "service", "environment", "team"])["cost"] \
        .sum().sort_values(ascending=False).reset_index()
    records = frame.groupby(["date", "resource_id", "name", "provider", "service", "category", "environment", "team",
                             "unit", "unit_price"]).agg(quantity=("quantity", "sum"), cost=("cost", "sum")).reset_index()
    records = records.sort_values(["date", "cost"], ascending=[False, False])
    trend = frame.groupby(["date", "provider"])["cost"].sum().reset_index()
    total = float(frame["cost"].sum())
    previous_total = float(previous["cost"].sum())
    includes_today = end_ts >= snap.today
    return {
        "start": f"{start_ts:%Y-%m-%d}", "end": f"{end_ts:%Y-%m-%d}", "days": days, "filters": filters,
        "includes_partial_today": bool(includes_today),
        "total": round_money(total), "previous_total": round_money(previous_total),
        "previous_start": f"{previous_start:%Y-%m-%d}",
        "change_pct": round(100 * (total - previous_total) / previous_total, 2) if previous_total else None,
        "daily_average": round_money(total / days),
        "trend": [{"date": f"{r.date:%Y-%m-%d}", "provider": r.provider, "cost": round_money(r.cost)}
                  for r in trend.itertuples()],
        "by_provider": grouped(frame, "provider"), "by_team": grouped(frame, "team"),
        "by_environment": grouped(frame, "environment"), "by_category": grouped(frame, "category"),
        "by_service": [{"service": name, "cost": round_money(row.cost), "previous_cost": round_money(row.previous_cost),
                        "change": round_money(row.change)} for name, row in services.iterrows()],
        "top_resources": [{"resource_id": r.resource_id, "name": r.name, "provider": r.provider, "service": r.service,
                           "environment": r.environment, "team": r.team, "cost": round_money(r.cost)}
                          for r in by_resource.head(15).itertuples()],
        "records": [{"date": f"{r.date:%Y-%m-%d}", "resource_id": r.resource_id, "name": r.name,
                     "provider": r.provider, "service": r.service, "category": r.category,
                     "environment": r.environment, "team": r.team, "quantity": round(float(r.quantity), 4),
                     "unit": r.unit, "unit_price": round(float(r.unit_price), 4), "cost": round_money(r.cost)}
                    for r in records.head(3000).itertuples()],
        "record_count": int(len(records)),
        "options": filter_options(snap),
    }


def workload_state(resource: pd.Series, rows: pd.DataFrame, last_active, clock) -> tuple:
    kind, status = resource["kind"], resource["status"]
    if not is_usage_kind(kind):
        return "Billed by volume", False
    if status == "stopped":
        return "Stopped", False
    hours_since_active = None if last_active is None else (clock - last_active).total_seconds() / 3600
    latest = rows[rows["ts"] == rows["ts"].max()] if not rows.empty else rows
    running_now = status == "running" or (status == "scheduled" and isinstance(resource["started_at"], str))
    active_now = bool(not latest.empty and latest["active"].iloc[0] == 1 and
                      latest["ts"].iloc[0] >= pd.Timestamp(clock - timedelta(hours=1)))
    if kind == "job":
        if not running_now:
            return f"Completed - next run at {resource['profile']['start']:02d}:00", False
        if active_now:
            return "Job running", False
        if hours_since_active is not None and hours_since_active >= 3:
            return f"Running idle - job finished {humanize_hours(hours_since_active)} ago", True
        return "Job finishing", False
    if kind == "batch":
        return ("Training run in progress", False) if running_now else ("Stopped between scheduled runs", False)
    if kind == "worker":
        return ("Processing queue", False) if active_now else ("Waiting for queue messages", False)
    if kind in ("web", "service", "database"):
        return "Serving traffic", False
    if kind == "continuous":
        return "Long-running job in progress", False
    if hours_since_active is None or hours_since_active >= 24:
        idle_for = "since launch" if hours_since_active is None else f"for {humanize_hours(hours_since_active)}"
        return f"Running but idle {idle_for}", True
    return ("In use" if active_now else "Idle outside working hours"), False


def resource_rows(snap: Snapshot, conn=None) -> list:
    clock = snap.clock
    usage = snap.usage
    week = usage[usage["ts"] >= pd.Timestamp(clock - timedelta(hours=168))]
    compute_week = week[week["charge"] == "usage"]
    last_active = usage[usage["active"] == 1].groupby("resource_id")["ts"].max()
    total_cost = usage.groupby("resource_id")["cost"].sum()
    month_cost = usage[usage["date"] >= snap.month_start].groupby("resource_id")["cost"].sum()
    recommendations = {}
    if conn is not None:
        recommendations = {item["resource_id"]: item for item in rightsizing.build_recommendations(conn, snap)["items"]}
    rows = []
    for rid, resource in snap.resources.iterrows():
        own = compute_week[compute_week["resource_id"] == rid]
        if not is_usage_kind(resource["kind"]):
            own = own.iloc[0:0]
        active_ts = last_active.get(rid)
        last_activity = None if active_ts is None or pd.isna(active_ts) else (active_ts + pd.Timedelta(hours=1)).to_pydatetime()
        workload, flagged_idle = workload_state(resource, own, last_activity, clock)
        started = resource["started_at"]
        running_hours = (clock - engine.parse_ts(started)).total_seconds() / 3600 if isinstance(started, str) else 0.0
        expected = snap.expected_month.loc[rid]
        recommendation = recommendations.get(rid)
        rows.append({
            "resource_id": rid, "name": resource["name"], "provider": resource["provider"],
            "service": resource["service"], "category": resource["category"], "environment": resource["environment"],
            "team": resource["team"], "region": resource["region"], "resource_class": resource["resource_class"],
            "kind": resource["kind"], "status": resource["status"], "purpose": resource["purpose"],
            "launched_at": resource["launched_at"], "started_at": started if isinstance(started, str) else None,
            "running_hours": running_hours, "uptime_hours_7d": float(len(own)),
            "active_hours_7d": float(own["active"].sum()) if len(own) else 0.0,
            "idle_running_cost_7d": float(own.loc[own["active"] == 0, "cost"].sum()),
            "avg_cpu_7d": None if own.empty else float(own["cpu"].mean()),
            "avg_mem_7d": None if own.empty else float(own["mem"].mean()),
            "last_activity": None if last_activity is None else engine.to_ts(last_activity),
            "hourly_cost": float(expected["next_hour"]), "monthly_estimate": float(expected["total"]),
            "accumulated_cost": float(total_cost.get(rid, 0.0)), "month_to_date_cost": float(month_cost.get(rid, 0.0)),
            "workload_status": workload, "idle_flag": flagged_idle,
            "recommendation": recommendation["action"] if recommendation else None,
            "recommendation_status": recommendation["decision_status"] if recommendation else None,
            "can_stop": is_usage_kind(resource["kind"]) and resource["status"] != "stopped",
            "can_start": is_usage_kind(resource["kind"]) and resource["status"] == "stopped",
        })
    return rows


def inventory(conn, snap: Snapshot) -> dict:
    rows = resource_rows(snap, conn)
    active_alerts = {}
    for item in anomalies.list_anomalies(conn, snap)["items"]:
        if item["status"] in anomalies.ACTIVE_STATUSES:
            active_alerts[item["resource_id"]] = active_alerts.get(item["resource_id"], 0) + 1
    for row in rows:
        row["active_anomalies"] = active_alerts.get(row["resource_id"], 0)
    totals = {"accumulated_cost": sum(r["accumulated_cost"] for r in rows),
              "month_to_date_cost": sum(r["month_to_date_cost"] for r in rows),
              "hourly_cost": sum(r["hourly_cost"] for r in rows)}
    return {"items": rows, "totals": totals}


def resource_detail(conn, snap: Snapshot, resource_id: str) -> dict:
    if resource_id not in snap.resources.index:
        raise engine.SimulationError(f"Resource '{resource_id}' not found", 404)
    row = next(r for r in inventory(conn, snap)["items"] if r["resource_id"] == resource_id)
    daily = snap.daily_resource
    own_daily = daily[daily["resource_id"] == resource_id].tail(21)
    hourly = snap.resource_window(resource_id, hours=72)
    compute_hourly = hourly[hourly["charge"] == "usage"]
    cost_hourly = hourly.groupby("ts")["cost"].sum()
    recommendation = next((i for i in rightsizing.build_recommendations(conn, snap)["items"]
                           if i["resource_id"] == resource_id), None)
    resource_anomalies = [i for i in anomalies.list_anomalies(conn, snap)["items"] if i["resource_id"] == resource_id]
    actions = snap.actions[snap.actions["resource_id"] == resource_id]
    charges = snap.usage[snap.usage["resource_id"] == resource_id].groupby(["charge", "unit", "unit_price"]) \
        .agg(quantity=("quantity", "sum"), cost=("cost", "sum")).reset_index()
    return {
        **row,
        "profile": snap.resources.loc[resource_id, "profile"],
        "unit": snap.resources.loc[resource_id, "unit"], "unit_price": float(snap.resources.loc[resource_id, "unit_price"]),
        "instances": int(snap.resources.loc[resource_id, "instances"]),
        "disk_gb": float(snap.resources.loc[resource_id, "disk_gb"]),
        "daily": [{"date": f"{r.date:%Y-%m-%d}", "cost": float(r.cost), "running_hours": float(r.running_hours),
                   "active_hours": float(r.active_hours), "cpu": None if pd.isna(r.cpu) else float(r.cpu),
                   "mem": None if pd.isna(r.mem) else float(r.mem)} for r in own_daily.itertuples()],
        "hourly": [{"ts": engine.to_ts(ts), "cost": float(cost_hourly.get(ts, 0.0)),
                    "cpu": float(r.cpu), "mem": float(r.mem), "active": int(r.active), "instances": float(r.quantity)}
                   for ts, r in compute_hourly.set_index("ts").iterrows()],
        "charges": [{"charge": r.charge, "unit": r.unit, "unit_price": float(r.unit_price),
                     "quantity": float(r.quantity), "cost": float(r.cost)} for r in charges.itertuples()],
        "recommendation_detail": recommendation,
        "anomalies": resource_anomalies,
        "actions": actions.to_dict(orient="records"),
    }


def attention_items(snap: Snapshot, recommendations: dict, anomaly_list: dict, budget: dict, rows: list) -> list:
    items = []
    by_id = {r["resource_id"]: r for r in rows}
    for rec in recommendations["items"]:
        if rec["action"] == "RETAIN" or rec["decision_status"] in ("rejected", "snoozed"):
            continue
        row = by_id[rec["resource_id"]]
        if rec["action"] == "STOP_REVIEW":
            title = f"{rec['name']} has been running for {humanize_hours(rec['running_hours'])} with very low utilization"
            severity, action = "critical", "Review for shutdown"
        else:
            title = f"{rec['name']} is larger than its workload needs"
            severity, action = "warning", "Downsize"
        items.append({
            "severity": severity, "title": title, "resource_id": rec["resource_id"], "resource": rec["name"],
            "detail": f"{rec['avg_cpu']:.1f}% CPU, {rec['avg_mem']:.1f}% memory, {rec['avg_hours']:.1f} active h/day "
                      f"({rec['provider']} {rec['service']}, {rec['environment']}).",
            "unnecessary_spend_7d": row["idle_running_cost_7d"], "monthly_impact": rec["estimated_saving"],
            "duration": humanize_hours(rec["running_hours"]), "recommended_action": action, "page": "Optimization",
        })
    for anomaly in anomaly_list["items"]:
        if anomaly["status"] not in anomalies.ACTIVE_STATUSES or anomaly["severity"] == "medium":
            continue
        span = "1 day" if anomaly["days_flagged"] == 1 else f"{anomaly['days_flagged']} days"
        items.append({
            "severity": "critical" if anomaly["severity"] == "critical" else "warning",
            "title": f"Cost spike on {anomaly['name']} ({anomaly['date']})", "resource_id": anomaly["resource_id"],
            "resource": anomaly["name"],
            "detail": f"Observed INR {anomaly['observed_cost']:,.0f} against a typical INR {anomaly['baseline_cost']:,.0f} "
                      f"per day ({anomaly['cost_ratio']:.1f}x).",
            "unnecessary_spend_7d": anomaly["total_excess_cost"], "monthly_impact": None, "duration": span,
            "recommended_action": "Investigate", "page": "Anomalies",
        })
    if budget["status"] != "on_track":
        items.append({
            "severity": "critical" if budget["status"] == "over_budget" else "warning",
            "title": "Projected month-end spend is " + ("above" if budget["variance"] > 0 else "close to") + " the budget",
            "resource_id": None, "resource": "All resources",
            "detail": f"Projected INR {budget['projected_month_end']:,.0f} against a budget of "
                      f"INR {budget['monthly_budget']:,.0f}.",
            "unnecessary_spend_7d": None, "monthly_impact": max(budget["variance"], 0), "duration": "This month",
            "recommended_action": "Review forecast and savings", "page": "Forecast",
        })
    rank = {"critical": 0, "warning": 1}
    items.sort(key=lambda item: (rank[item["severity"]], -(item["monthly_impact"] or item["unnecessary_spend_7d"] or 0)))
    return items[:6]


def overview(conn, snap: Snapshot) -> dict:
    budget_amount = store.get_setting(conn, "monthly_budget")
    budget = forecasting.budget_view(snap, budget_amount)
    forecast = forecasting.build_forecast(snap)
    recommendations = rightsizing.build_recommendations(conn, snap)
    anomaly_list = anomalies.list_anomalies(conn, snap)
    rows = resource_rows(snap, conn)
    usage = snap.usage
    month = usage[usage["date"] >= snap.month_start]
    yesterday = snap.today - pd.Timedelta(days=1)
    last_30 = usage[usage["date"] >= snap.today - pd.Timedelta(days=29)]
    daily = last_30.groupby(["date", "provider"])["cost"].sum().reset_index()
    flagged = {r["resource_id"] for r in rows if r["idle_flag"]} | \
        {i["resource_id"] for i in recommendations["items"] if i["action"] != "RETAIN"}
    running = [r for r in rows if is_usage_kind(r["kind"]) and r["started_at"] is not None]
    return {
        "clock": engine.to_ts(snap.clock),
        "spend": {
            "today": snap.spend_between(snap.today), "yesterday": snap.spend_between(yesterday, snap.today),
            "this_week": snap.spend_between(snap.week_start), "month_to_date": snap.spend_between(snap.month_start),
            "hours_into_today": snap.clock.hour, "week_start": f"{snap.week_start:%Y-%m-%d}",
            "month_start": f"{snap.month_start:%Y-%m-%d}",
        },
        "budget": budget,
        "forecast_model": forecast["model"],
        "projected_range": forecast["projected_range"],
        "potential_monthly_savings": recommendations["summary"]["potential_monthly_savings"],
        "realized_monthly_run_rate": recommendations["summary"]["realized_monthly_run_rate"],
        "active_anomalies": anomaly_list["active"],
        "running_resources": len(running),
        "total_resources": len(rows),
        "idle_or_underutilized": len(flagged),
        "daily": [{"date": f"{r.date:%Y-%m-%d}", "provider": r.provider, "cost": round_money(r.cost)}
                  for r in daily.itertuples()],
        "forecast": [p for p in forecast["forecast"] if p["date"] <= forecast["month_end"]],
        "by_provider_month": grouped(month, "provider"),
        "top_services_month": grouped(month, "service", limit=6),
        "top_resources_month": [
            {"resource_id": rid, "name": snap.resources.loc[rid, "name"], "provider": snap.resources.loc[rid, "provider"],
             "service": snap.resources.loc[rid, "service"], "cost": round_money(cost),
             "hourly_cost": round_money(snap.expected_month.loc[rid, "next_hour"])}
            for rid, cost in month.groupby("resource_id")["cost"].sum().sort_values(ascending=False).head(6).items()],
        "attention": attention_items(snap, recommendations, anomaly_list, budget, rows),
    }
