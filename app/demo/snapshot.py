import threading
from datetime import timedelta
from functools import cached_property

import pandas as pd

from app.config import database_path
from app.demo import catalog, engine, store

DIMENSIONS = ["name", "provider", "service", "environment", "team", "kind"]
_cache = {}
_cache_lock = threading.Lock()


class Snapshot:
    def __init__(self, conn):
        self.sim_version = store.get_setting(conn, "sim_version", 0)
        self.scenario = store.get_setting(conn, "scenario")
        self.clock = engine.current_clock(conn)
        self.history_start = engine.history_start(conn)
        self.resource_list = engine.load_resources(conn)
        self.resources = pd.DataFrame(self.resource_list).set_index("resource_id", drop=False)
        self.events = engine.load_events(conn)
        self.actions = pd.read_sql("SELECT * FROM actions ORDER BY ts, action_id", conn)
        usage = pd.read_sql("SELECT * FROM usage", conn)
        usage["ts"] = pd.to_datetime(usage["ts"], format=engine.TS_FORMAT)
        usage["date"] = usage["ts"].dt.normalize()
        self.usage = usage.join(self.resources[DIMENSIONS], on="resource_id")

    @property
    def today(self) -> pd.Timestamp:
        return pd.Timestamp(self.clock.date())

    @property
    def month_start(self) -> pd.Timestamp:
        return self.today.replace(day=1)

    @property
    def week_start(self) -> pd.Timestamp:
        return self.today - pd.Timedelta(days=self.today.weekday())

    @property
    def month_end(self) -> pd.Timestamp:
        return (self.month_start + pd.offsets.MonthEnd(1)).normalize()

    @cached_property
    def daily_resource(self) -> pd.DataFrame:
        usage = self.usage
        totals = usage.groupby(["resource_id", "date"])["cost"].sum().rename("cost")
        compute = usage[usage["charge"] == "usage"].groupby(["resource_id", "date"]).agg(
            usage_cost=("cost", "sum"), quantity=("quantity", "sum"), running_hours=("ts", "count"),
            active_hours=("active", "sum"), cpu=("cpu", "mean"), mem=("mem", "mean"))
        daily = pd.concat([totals, compute], axis=1).reset_index()
        daily[["usage_cost", "quantity", "running_hours", "active_hours"]] = \
            daily[["usage_cost", "quantity", "running_hours", "active_hours"]].fillna(0)
        return daily.join(self.resources[DIMENSIONS], on="resource_id")

    @cached_property
    def expected_month(self) -> pd.DataFrame:
        hours = 30 * 24
        rows = engine.generate_usage(self.resource_list, [], self.clock, hours, self.history_start)
        frame = pd.DataFrame(rows, columns=["ts", "resource_id", "category", "charge", "unit", "quantity",
                                            "unit_price", "cost", "cpu", "mem", "active"])
        scale = engine.HOURS_PER_MONTH / hours
        monthly = frame.pivot_table(index="resource_id", columns="charge", values="cost", aggfunc="sum").fillna(0)
        monthly = monthly.reindex(columns=["usage", "disk"], fill_value=0) * scale
        monthly["total"] = monthly["usage"] + monthly["disk"]
        next_hour = frame[frame["ts"] == engine.to_ts(self.clock)].groupby("resource_id")["cost"].sum()
        monthly["next_hour"] = next_hour
        return monthly.reindex(self.resources.index).fillna(0)

    def spend_between(self, start: pd.Timestamp, end: pd.Timestamp = None, frame: pd.DataFrame = None) -> float:
        frame = self.usage if frame is None else frame
        mask = frame["ts"] >= start
        if end is not None:
            mask &= frame["ts"] < end
        return float(frame.loc[mask, "cost"].sum())

    def resource_window(self, resource_id: str, hours: int = 168) -> pd.DataFrame:
        start = pd.Timestamp(self.clock - timedelta(hours=hours))
        usage = self.usage
        return usage[(usage["resource_id"] == resource_id) & (usage["ts"] >= start)]


def get_snapshot(conn) -> Snapshot:
    key = (str(database_path()), store.get_setting(conn, "sim_version", 0))
    with _cache_lock:
        snapshot = _cache.get(key)
        if snapshot is None:
            _cache.clear()
            snapshot = Snapshot(conn)
            _cache[key] = snapshot
        return snapshot


def is_usage_kind(kind: str) -> bool:
    return kind in catalog.USAGE_KINDS
