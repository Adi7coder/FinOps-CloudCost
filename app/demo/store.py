import json
import sqlite3
from contextlib import contextmanager

from app.config import database_path

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS resources ("
    " resource_id TEXT PRIMARY KEY, name TEXT NOT NULL, provider TEXT NOT NULL, service TEXT NOT NULL,"
    " category TEXT NOT NULL, kind TEXT NOT NULL, environment TEXT NOT NULL, team TEXT NOT NULL,"
    " region TEXT NOT NULL, resource_class TEXT NOT NULL, purpose TEXT NOT NULL, unit TEXT NOT NULL,"
    " unit_price REAL NOT NULL, instances INTEGER NOT NULL, max_instances INTEGER, disk_gb REAL NOT NULL,"
    " disk_price REAL NOT NULL, size_factor REAL NOT NULL DEFAULT 1.0, status TEXT NOT NULL,"
    " launched_at TEXT NOT NULL, started_at TEXT, profile TEXT NOT NULL, previous_class TEXT,"
    " resized_at TEXT, previous_size_factor REAL, schedule_restored_at TEXT)",
    "CREATE TABLE IF NOT EXISTS usage ("
    " ts TEXT NOT NULL, resource_id TEXT NOT NULL, category TEXT NOT NULL, charge TEXT NOT NULL,"
    " unit TEXT NOT NULL, quantity REAL NOT NULL, unit_price REAL NOT NULL, cost REAL NOT NULL,"
    " cpu REAL, mem REAL, active INTEGER)",
    "CREATE INDEX IF NOT EXISTS usage_ts ON usage (ts)",
    "CREATE INDEX IF NOT EXISTS usage_resource ON usage (resource_id, ts)",
    "CREATE TABLE IF NOT EXISTS events ("
    " event_id INTEGER PRIMARY KEY AUTOINCREMENT, resource_id TEXT NOT NULL, start_ts TEXT NOT NULL,"
    " end_ts TEXT NOT NULL, factor REAL NOT NULL, label TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS actions ("
    " action_id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, resource_id TEXT NOT NULL,"
    " action TEXT NOT NULL, source TEXT NOT NULL, before_hourly REAL, note TEXT, proposed_monthly REAL)",
    "CREATE TABLE IF NOT EXISTS anomaly_reviews ("
    " anomaly_id TEXT PRIMARY KEY, resource_id TEXT NOT NULL, day TEXT NOT NULL, status TEXT NOT NULL,"
    " note TEXT, first_detected_at TEXT NOT NULL, updated_at TEXT)",
    "CREATE TABLE IF NOT EXISTS decisions ("
    " resource_id TEXT PRIMARY KEY, action TEXT NOT NULL, status TEXT NOT NULL, note TEXT,"
    " updated_at TEXT NOT NULL, snoozed_until TEXT)",
)
RUNTIME_TABLES = ("usage", "events", "actions", "anomaly_reviews", "decisions", "resources")


def connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def session():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_schema(conn: sqlite3.Connection):
    for statement in SCHEMA:
        conn.execute(statement)
    action_columns = {row["name"] for row in conn.execute("PRAGMA table_info(actions)")}
    if "proposed_monthly" not in action_columns:
        conn.execute("ALTER TABLE actions ADD COLUMN proposed_monthly REAL")


def clear_runtime_data(conn: sqlite3.Connection):
    for table in RUNTIME_TABLES:
        conn.execute(f"DELETE FROM {table}")


def get_setting(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_setting(conn: sqlite3.Connection, key: str, value):
    conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, json.dumps(value)))


def bump_version(conn: sqlite3.Connection, simulation_changed: bool):
    set_setting(conn, "state_version", get_setting(conn, "state_version", 0) + 1)
    if simulation_changed:
        set_setting(conn, "sim_version", get_setting(conn, "sim_version", 0) + 1)


def is_initialized(conn: sqlite3.Connection) -> bool:
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "settings" not in tables or "resources" not in tables:
        return False
    return get_setting(conn, "clock") is not None and conn.execute("SELECT COUNT(*) FROM resources").fetchone()[0] > 0
