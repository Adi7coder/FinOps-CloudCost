import json
import re
import zlib
from datetime import datetime, time, timedelta

import numpy as np

from app.demo import catalog, store

TS_FORMAT = "%Y-%m-%d %H:%M"
HOURS_PER_MONTH = 730
SIMULATION_SEED = 2026
DOWNSIZE_FACTOR = 0.55


class SimulationError(Exception):
    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


def to_ts(value: datetime) -> str:
    return value.strftime(TS_FORMAT)


def parse_ts(text: str) -> datetime:
    return datetime.strptime(text, TS_FORMAT)


def load_resources(conn) -> list:
    resources = []
    for row in conn.execute("SELECT * FROM resources ORDER BY provider, resource_id"):
        item = dict(row)
        item["profile"] = json.loads(item["profile"])
        resources.append(item)
    return resources


def load_events(conn) -> list:
    return [
        {**dict(row), "start": parse_ts(row["start_ts"]), "end": parse_ts(row["end_ts"])}
        for row in conn.execute("SELECT * FROM events ORDER BY start_ts")
    ]


def day_draws(resource_id: str, day) -> dict:
    rng = np.random.default_rng([SIMULATION_SEED, zlib.crc32(resource_id.encode()), day.toordinal()])
    return {
        "level": rng.normal(1.0, 0.04),
        "traffic": rng.normal(1.0, 0.05, 24),
        "cpu": rng.normal(1.0, 0.08, 24),
        "mem": rng.normal(1.0, 0.04, 24),
        "activity": rng.random(24),
    }


def traffic_curve(day, history_start: datetime, draws: dict) -> np.ndarray:
    hours = np.arange(24)
    diurnal = 0.30 + 0.70 * np.exp(-((hours - 14) / 5.0) ** 2)
    weekend = 0.75 if day.weekday() >= 5 else 1.0
    growth = 1 + 0.0015 * (day - history_start.date()).days
    return np.clip(diurnal * weekend * growth * draws["level"] * draws["traffic"], 0.05, None)


def hourly_profile(resource: dict, day, history_start: datetime) -> dict:
    profile, kind = resource["profile"], resource["kind"]
    draws = day_draws(resource["resource_id"], day)
    load = traffic_curve(day, history_start, draws)
    hours = np.arange(24)
    in_window = (hours >= profile.get("start", 0)) & (hours < profile.get("end", 24))
    running = np.ones(24, dtype=bool)
    active = np.ones(24, dtype=bool)
    instances = np.full(24, float(resource["instances"]))

    if kind == "web":
        share = np.clip((load - 0.45) / 0.55, 0, 1)
        instances = np.round(resource["instances"] + (resource["max_instances"] - resource["instances"]) * share)
        cpu = profile["cpu_low"] + (profile["cpu_high"] - profile["cpu_low"]) * np.clip(load, 0, 1)
        mem = profile["mem_low"] + (profile["mem_high"] - profile["mem_low"]) * np.clip(load, 0, 1)
    elif kind in ("service", "database"):
        cpu = profile["cpu"] * (0.65 + 0.5 * load)
        mem = profile["mem"] * (0.95 + 0.1 * load)
    elif kind in ("worker", "dev"):
        if kind == "worker":
            probability = np.where(in_window, profile["p_active"], 0.08)
        else:
            weekday_probability = profile["p_weekday"] if day.weekday() < 5 else profile["p_weekend"]
            probability = np.where(in_window, weekday_probability, 0.0)
        active = draws["activity"] < probability
        cpu = np.where(active, profile["cpu"], profile["idle_cpu"])
        mem = np.where(active, profile["mem"], profile["idle_mem"])
    elif kind == "job":
        active = in_window.copy()
        running = in_window.copy()
        cpu = np.where(active, profile["cpu"], profile["idle_cpu"])
        mem = np.where(active, profile["mem"], profile["idle_mem"])
    elif kind == "idle":
        active = draws["activity"] < 0.01
        cpu = np.where(active, profile["cpu"], profile["idle_cpu"])
        mem = np.where(active, profile["mem"], profile["idle_mem"])
    elif kind == "batch":
        running = in_window & (day.weekday() in profile["days"])
        active = running.copy()
        cpu = np.full(24, float(profile["cpu"]))
        mem = np.full(24, float(profile["mem"]))
    else:
        cpu = np.full(24, float(profile["cpu"]))
        mem = np.full(24, float(profile["mem"]))

    return {"load": load, "running": running, "active": active, "instances": instances,
            "cpu": cpu * draws["cpu"], "mem": mem * draws["mem"]}


def usage_rows_for_day(resource: dict, day, hour_indexes, events: list, history_start: datetime) -> list:
    rid, kind, profile = resource["resource_id"], resource["kind"], resource["profile"]
    launched = parse_ts(resource["launched_at"])
    day_start = datetime.combine(day, time())
    rows = []

    if kind in ("storage", "transfer"):
        load = traffic_curve(day, history_start, day_draws(rid, day))
        for hour in hour_indexes:
            ts = day_start + timedelta(hours=hour)
            if ts < launched:
                continue
            if kind == "storage":
                stored_gb = profile["gb"] * (1 + profile["growth"] * (day - history_start.date()).days)
                quantity = stored_gb / HOURS_PER_MONTH
            else:
                quantity = profile["gb_per_hour"] * load[hour]
            rows.append((to_ts(ts), rid, resource["category"], "usage", resource["unit"], quantity,
                         resource["unit_price"], quantity * resource["unit_price"], None, None, None))
        return rows

    hourly = hourly_profile(resource, day, history_start)
    size_factor = resource["size_factor"]
    always_on_since = profile.get("always_on_since")
    busy_until = launched + timedelta(days=profile.get("busy_days", 0))
    disk_quantity = resource["disk_gb"] / HOURS_PER_MONTH

    for hour in hour_indexes:
        ts = day_start + timedelta(hours=hour)
        if ts < launched:
            continue
        running, active = bool(hourly["running"][hour]), bool(hourly["active"][hour])
        instances = hourly["instances"][hour]
        cpu, mem = hourly["cpu"][hour], hourly["mem"][hour]

        if kind == "job" and resource["status"] == "running" and always_on_since and ts >= parse_ts(always_on_since):
            running = True
        if kind == "idle" and ts < busy_until:
            active, cpu, mem = True, profile["cpu"], profile["mem"]
        if resource["status"] == "stopped":
            running = False

        for event in events:
            if event["resource_id"] == rid and event["start"] <= ts < event["end"] and running:
                instances = round(instances * event["factor"])
                cpu, mem, active = max(cpu, 78.0), max(mem, 60.0), True

        if running:
            cpu = float(np.clip(cpu / size_factor, 0.5, 98))
            mem = float(np.clip(mem / size_factor, 1, 97))
            unit_price = resource["unit_price"] * size_factor
            rows.append((to_ts(ts), rid, resource["category"], "usage", resource["unit"], float(instances),
                         unit_price, float(instances) * unit_price, round(cpu, 2), round(mem, 2), int(active)))
        if resource["disk_gb"]:
            rows.append((to_ts(ts), rid, "Storage", "disk", "GB-month", disk_quantity, resource["disk_price"],
                         disk_quantity * resource["disk_price"], None, None, None))
    return rows


def generate_usage(resources: list, events: list, start: datetime, hours: int, history_start: datetime) -> list:
    end = start + timedelta(hours=hours)
    rows = []
    day = start.date()
    while datetime.combine(day, time()) < end:
        day_start = datetime.combine(day, time())
        first, last = max(start, day_start), min(end, day_start + timedelta(days=1))
        hour_indexes = range(first.hour, first.hour + int((last - first).total_seconds() // 3600))
        for resource in resources:
            rows.extend(usage_rows_for_day(resource, day, hour_indexes, events, history_start))
        day += timedelta(days=1)
    return rows


def insert_usage(conn, rows: list):
    conn.executemany("INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)


def current_clock(conn) -> datetime:
    return parse_ts(store.get_setting(conn, "clock"))


def history_start(conn) -> datetime:
    return parse_ts(store.get_setting(conn, "history_start"))


def insert_resource(conn, spec: dict, launched_at: datetime, profile: dict):
    started_at = None if spec["status"] in ("scheduled", "stopped") else to_ts(launched_at)
    conn.execute(
        "INSERT INTO resources (resource_id, name, provider, service, category, kind, environment, team, region,"
        " resource_class, purpose, unit, unit_price, instances, max_instances, disk_gb, disk_price, status,"
        " launched_at, started_at, profile) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (spec["resource_id"], spec["name"], spec["provider"], spec["service"], spec["category"], spec["kind"],
         spec["environment"], spec["team"], spec["region"], spec["resource_class"], spec["purpose"], spec["unit"],
         spec["unit_price"], spec["instances"], spec["max_instances"], spec["disk_gb"], spec["disk_price"],
         spec["status"], to_ts(launched_at), started_at, json.dumps(profile)))


def load_scenario(conn, scenario_key: str):
    if scenario_key not in catalog.SCENARIOS:
        raise SimulationError(f"Unknown scenario '{scenario_key}'", 404)
    scenario = catalog.SCENARIOS[scenario_key]
    store.create_schema(conn)
    store.clear_runtime_data(conn)

    clock = catalog.BASELINE_CLOCK
    for spec in catalog.BASE_RESOURCES + scenario["extra_resources"]:
        spec = dict(spec)
        profile = dict(spec["profile"])
        launched_at = catalog.HISTORY_START if spec["launched_days_ago"] is None \
            else clock - timedelta(days=spec["launched_days_ago"])
        days_always_on = scenario["always_on"].get(spec["resource_id"])
        if days_always_on:
            profile["always_on_since"] = to_ts((clock - timedelta(days=days_always_on)).replace(hour=0))
            spec["status"] = "running"
        insert_resource(conn, spec, launched_at, profile)
        if days_always_on:
            conn.execute("UPDATE resources SET started_at = ? WHERE resource_id = ?",
                         (profile["always_on_since"], spec["resource_id"]))

    for event in scenario["events"]:
        start = clock - timedelta(hours=event["start_hours_ago"])
        conn.execute("INSERT INTO events (resource_id, start_ts, end_ts, factor, label) VALUES (?, ?, ?, ?, ?)",
                     (event["resource_id"], to_ts(start), to_ts(start + timedelta(hours=event["duration_hours"])),
                      event["factor"], event["label"]))

    store.set_setting(conn, "scenario", scenario_key)
    store.set_setting(conn, "history_start", to_ts(catalog.HISTORY_START))
    store.set_setting(conn, "clock", to_ts(clock))
    store.set_setting(conn, "monthly_budget", catalog.DEFAULT_MONTHLY_BUDGET)
    refresh_job_states(conn, clock)
    hours = int((clock - catalog.HISTORY_START).total_seconds() // 3600)
    insert_usage(conn, generate_usage(load_resources(conn), load_events(conn), catalog.HISTORY_START, hours,
                                      catalog.HISTORY_START))
    store.bump_version(conn, simulation_changed=True)


def ensure_initialized():
    with store.session() as conn:
        store.create_schema(conn)
        if not store.is_initialized(conn):
            load_scenario(conn, _default_scenario())


def _default_scenario() -> str:
    from app.config import default_scenario

    key = default_scenario()
    return key if key in catalog.SCENARIOS else "normal"


def refresh_job_states(conn, clock: datetime):
    for resource in load_resources(conn):
        if resource["kind"] in ("job", "batch") and resource["status"] == "scheduled":
            profile = resource["profile"]
            in_window = profile["start"] <= clock.hour < profile["end"]
            if resource["kind"] == "batch":
                in_window = in_window and clock.weekday() in profile["days"]
            started = to_ts(clock.replace(hour=profile["start"])) if in_window else None
            conn.execute("UPDATE resources SET started_at = ? WHERE resource_id = ?",
                         (started, resource["resource_id"]))


def advance(conn, hours: int) -> dict:
    clock = current_clock(conn)
    rows = generate_usage(load_resources(conn), load_events(conn), clock, hours, history_start(conn))
    insert_usage(conn, rows)
    new_clock = clock + timedelta(hours=hours)
    store.set_setting(conn, "clock", to_ts(new_clock))
    refresh_job_states(conn, new_clock)
    store.bump_version(conn, simulation_changed=True)
    return {"previous_clock": to_ts(clock), "clock": to_ts(new_clock), "hours": hours,
            "cost_added": round(sum(row[7] for row in rows), 2), "records_added": len(rows)}


def get_resource(conn, resource_id: str) -> dict:
    for resource in load_resources(conn):
        if resource["resource_id"] == resource_id:
            return resource
    raise SimulationError(f"Resource '{resource_id}' not found", 404)


def average_hourly_usage_cost(conn, resource_id: str, clock: datetime, hours: int = 168) -> float:
    start = clock - timedelta(hours=hours)
    total = conn.execute("SELECT COALESCE(SUM(cost), 0) FROM usage WHERE resource_id = ? AND charge = 'usage'"
                         " AND ts >= ? AND ts < ?", (resource_id, to_ts(start), to_ts(clock))).fetchone()[0]
    return total / hours


def record_action(conn, resource_id: str, action: str, source: str, note: str, before_hourly=None):
    conn.execute("INSERT INTO actions (ts, resource_id, action, source, before_hourly, note) VALUES (?, ?, ?, ?, ?, ?)",
                 (to_ts(current_clock(conn)), resource_id, action, source, before_hourly, note))


def change_power_state(conn, resource_id: str, action: str, source: str = "manual") -> dict:
    resource = get_resource(conn, resource_id)
    clock = current_clock(conn)
    if resource["kind"] not in catalog.USAGE_KINDS:
        raise SimulationError(f"{resource['name']} is billed by stored or transferred GB and cannot be stopped", 422)
    if action == "stop":
        if resource["status"] == "stopped":
            raise SimulationError(f"{resource['name']} is already stopped")
        before = average_hourly_usage_cost(conn, resource_id, clock)
        conn.execute("UPDATE resources SET status = 'stopped', started_at = NULL WHERE resource_id = ?", (resource_id,))
        record_action(conn, resource_id, "stop", source, "Compute billing stops; attached disk keeps billing", before)
    else:
        if resource["status"] != "stopped":
            raise SimulationError(f"{resource['name']} is not stopped")
        new_status = "scheduled" if resource["kind"] in ("job", "batch") else "running"
        started_at = None if new_status == "scheduled" else to_ts(clock)
        conn.execute("UPDATE resources SET status = ?, started_at = ? WHERE resource_id = ?",
                     (new_status, started_at, resource_id))
        record_action(conn, resource_id, "start", source, f"Resource returned to {new_status} state")
    store.bump_version(conn, simulation_changed=True)
    return get_resource(conn, resource_id)


def smaller_class(resource_class: str) -> str:
    steps = [("2xlarge", "xlarge"), ("xlarge", "large"), ("large", "medium"), ("standard-8", "standard-4"),
             ("standard-4", "standard-2"), ("4 vCore", "2 vCore"), ("D4s", "D2s"), ("B2ms", "B1ms")]
    for bigger, smaller in steps:
        if bigger in resource_class:
            return resource_class.replace(bigger, smaller, 1)
    return re.sub(r"\s*$", " (one size smaller)", resource_class, count=1)


def apply_downsize(conn, resource_id: str) -> dict:
    resource = get_resource(conn, resource_id)
    clock = current_clock(conn)
    before = average_hourly_usage_cost(conn, resource_id, clock)
    new_class = smaller_class(resource["resource_class"])
    conn.execute("UPDATE resources SET size_factor = ?, previous_size_factor = ?, previous_class = ?,"
                 " resource_class = ?, resized_at = ? WHERE resource_id = ?",
                 (resource["size_factor"] * DOWNSIZE_FACTOR, resource["size_factor"], resource["resource_class"],
                  new_class, to_ts(clock), resource_id))
    record_action(conn, resource_id, "downsize", "recommendation",
                  f"{resource['resource_class']} -> {new_class} (simulated)", before)
    store.bump_version(conn, simulation_changed=True)
    return get_resource(conn, resource_id)


def restore_schedule(conn, resource_id: str) -> dict:
    resource = get_resource(conn, resource_id)
    clock = current_clock(conn)
    before = average_hourly_usage_cost(conn, resource_id, clock)
    profile = dict(resource["profile"])
    profile.pop("always_on_since", None)
    conn.execute("UPDATE resources SET status = 'scheduled', profile = ?, schedule_restored_at = ?, started_at = NULL"
                 " WHERE resource_id = ?", (json.dumps(profile), to_ts(clock), resource_id))
    record_action(conn, resource_id, "restore_schedule", "recommendation",
                  f"VM now runs only during its {profile['start']:02d}:00-{profile['end']:02d}:00 job window", before)
    refresh_job_states(conn, clock)
    store.bump_version(conn, simulation_changed=True)
    return get_resource(conn, resource_id)


def trigger_event(conn, event_key: str) -> dict:
    if event_key not in catalog.SPENDING_EVENTS:
        raise SimulationError(f"Unknown event '{event_key}'", 404)
    template = catalog.SPENDING_EVENTS[event_key]
    resource = get_resource(conn, template["resource_id"])
    if resource["status"] == "stopped":
        raise SimulationError(f"{resource['name']} is stopped, so it cannot scale out")
    clock = current_clock(conn)
    for event in load_events(conn):
        if event["resource_id"] == template["resource_id"] and event["end"] > clock:
            raise SimulationError("This spending event is already in progress")
    end = clock + timedelta(hours=template["duration_hours"])
    conn.execute("INSERT INTO events (resource_id, start_ts, end_ts, factor, label) VALUES (?, ?, ?, ?, ?)",
                 (template["resource_id"], to_ts(clock), to_ts(end), template["factor"], template["label"]))
    store.bump_version(conn, simulation_changed=True)
    return {"resource_id": template["resource_id"], "resource_name": resource["name"], "start": to_ts(clock),
            "end": to_ts(end), "label": template["label"]}
