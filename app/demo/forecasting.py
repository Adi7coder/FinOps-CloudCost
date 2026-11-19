import importlib.util
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from app.demo import anomalies, engine
from app.demo.snapshot import Snapshot, is_usage_kind
from app.ml import RIDGE_ALPHA, SEASONAL_LAG, build_seasonal_features, evaluate_forecast

NEW_RESOURCE_DAYS = 14
RUN_RATE_HOURS = 72
BACKTEST_DAYS = 14
MIN_HISTORY_DAYS = 28
Z_95 = 1.96


def restated_usage(snap: Snapshot) -> pd.DataFrame:
    usage = snap.usage[["ts", "date", "resource_id", "charge", "cost", "active"]].copy()
    usage["restated_cost"] = usage["cost"]
    for _, resource in snap.resources.iterrows():
        rid = resource["resource_id"]
        own_usage = (usage["resource_id"] == rid) & (usage["charge"] == "usage")
        if resource["status"] == "stopped":
            usage.loc[own_usage, "restated_cost"] = 0.0
        if isinstance(resource.get("resized_at"), str):
            before = own_usage & (usage["ts"] < pd.Timestamp(engine.parse_ts(resource["resized_at"])))
            ratio = resource["size_factor"] / resource["previous_size_factor"]
            usage.loc[before, "restated_cost"] = usage.loc[before, "cost"] * ratio
        if isinstance(resource.get("schedule_restored_at"), str):
            before = own_usage & (usage["ts"] < pd.Timestamp(engine.parse_ts(resource["schedule_restored_at"])))
            usage.loc[before & (usage["active"] == 0), "restated_cost"] = 0.0
    return remove_anomalous_days(snap, usage)


def remove_anomalous_days(snap: Snapshot, usage: pd.DataFrame) -> pd.DataFrame:
    detected = anomalies.detect(snap)
    for row in detected[detected["reported"]].itertuples():
        own_day = (usage["resource_id"] == row.resource_id) & (usage["date"] == row.date)
        usage.loc[own_day, "restated_cost"] *= row.svc_median / row.cost
    return usage


def split_new_resources(snap: Snapshot):
    cutoff = snap.clock - timedelta(days=NEW_RESOURCE_DAYS)
    launched = snap.resources["launched_at"].map(engine.parse_ts)
    new_ids = snap.resources.index[(launched > cutoff) & (snap.resources["status"] != "stopped")
                                   & snap.resources["kind"].map(is_usage_kind)]
    return list(new_ids)


def new_resource_daily_run_rate(snap: Snapshot, restated: pd.DataFrame, new_ids: list) -> list:
    rates = []
    for rid in new_ids:
        launched = engine.parse_ts(snap.resources.loc[rid, "launched_at"])
        start = max(launched, snap.clock - timedelta(hours=RUN_RATE_HOURS))
        hours = max((snap.clock - start).total_seconds() / 3600, 1)
        recent = restated[(restated["resource_id"] == rid) & (restated["ts"] >= pd.Timestamp(start))]
        rates.append({"resource_id": rid, "name": snap.resources.loc[rid, "name"],
                      "daily_run_rate": float(recent["restated_cost"].sum()) * 24 / hours})
    return rates


def ridge_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    features = build_seasonal_features(len(series) + horizon)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(features[:len(series)], series)
    return model.predict(features[len(series):])


def backtest(series: np.ndarray, dates: list) -> dict:
    train, holdout = series[:-BACKTEST_DAYS], series[-BACKTEST_DAYS:]
    additive = ridge_forecast(train, BACKTEST_DAYS)
    holdout_index = np.arange(len(series) - BACKTEST_DAYS, len(series))
    naive = series[holdout_index - SEASONAL_LAG]
    return {
        "days": BACKTEST_DAYS,
        "metrics": [
            {"model": "Seasonal Naive", **evaluate_forecast(holdout, naive)},
            {"model": "Additive Seasonal", **evaluate_forecast(holdout, additive)},
        ],
        "series": [{"date": d, "actual": float(a), "additive_seasonal": float(p), "seasonal_naive": float(n)}
                   for d, a, p, n in zip(dates[-BACKTEST_DAYS:], holdout, additive, naive)],
        "residual_std": {"Seasonal Naive": float(np.std(holdout - naive, ddof=1)),
                         "Additive Seasonal": float(np.std(holdout - additive, ddof=1))},
    }


def seasonal_naive_projection(series: np.ndarray, horizon: int) -> np.ndarray:
    last_week = series[-SEASONAL_LAG:]
    return np.array([last_week[h % SEASONAL_LAG] for h in range(horizon)])


def lstm_status() -> str:
    if importlib.util.find_spec("torch") is None:
        return "Not run live: PyTorch is not installed in this environment. Archived benchmark result only."
    return "Not run live: the 450-epoch LSTM is too slow for interactive use. Archived benchmark result only."


def build_forecast(snap: Snapshot) -> dict:
    cached = getattr(snap, "_forecast_result", None)
    if cached is not None:
        return cached
    restated = restated_usage(snap)
    new_ids = split_new_resources(snap)
    established = restated[~restated["resource_id"].isin(new_ids)]
    complete = established[established["date"] < snap.today]
    daily = complete.groupby("date")["restated_cost"].sum().sort_index()
    dates = [f"{d:%Y-%m-%d}" for d in daily.index]
    series = daily.to_numpy()
    new_rates = new_resource_daily_run_rate(snap, restated, new_ids)
    new_daily = sum(rate["daily_run_rate"] for rate in new_rates)

    month_end = snap.month_end
    horizon_days = max((month_end - snap.today).days + 1, 7)
    forecast_dates = [snap.today + pd.Timedelta(days=i) for i in range(horizon_days)]

    if len(series) >= MIN_HISTORY_DAYS:
        evaluation = backtest(series, dates)
        model_name = min(evaluation["metrics"], key=lambda metric: metric["MAPE_pct"])["model"]
        if model_name == "Seasonal Naive":
            base = seasonal_naive_projection(series, horizon_days)
        else:
            base = ridge_forecast(series, horizon_days)
        predicted = base + new_daily
        sigma = evaluation["residual_std"][model_name]
        selection_note = (f"{model_name} had the lowest MAPE on the last {BACKTEST_DAYS} complete days, so it is "
                          "used for the projection (same selection rule as the Review-2 benchmark).")
    else:
        model_name = "Run-rate fallback"
        predicted = np.full(horizon_days, series[-7:].mean() if len(series) else 0.0) + new_daily
        evaluation, sigma = None, 0.0
        selection_note = f"Fewer than {MIN_HISTORY_DAYS} complete days of history, so the last-7-day average is used."
    predicted = np.maximum(predicted, 0)

    actual_daily = snap.usage.groupby("date")["cost"].sum()
    month_to_date = float(actual_daily[actual_daily.index >= snap.month_start].sum())
    remaining_today_share = (24 - snap.clock.hour) / 24
    in_month = [i for i, d in enumerate(forecast_dates) if d <= month_end]
    remaining = predicted[0] * remaining_today_share + sum(predicted[i] for i in in_month[1:])
    remaining_days = remaining_today_share + max(len(in_month) - 1, 0)
    margin = Z_95 * sigma * np.sqrt(remaining_days)

    history = actual_daily[actual_daily.index >= snap.today - pd.Timedelta(days=59)]
    adjustments = []
    for _, resource in snap.resources.iterrows():
        rid = resource["resource_id"]
        own = (restated["resource_id"] == rid) & (restated["date"] >= snap.today - pd.Timedelta(days=7)) \
            & (restated["date"] < snap.today)
        removed = float((restated.loc[own, "cost"] - restated.loc[own, "restated_cost"]).sum()) / 7
        if removed > 1:
            adjustments.append({"resource_id": rid, "name": resource["name"], "status": resource["status"],
                                "removed_daily_cost": removed})

    result = {
        "model": model_name,
        "selection_note": selection_note,
        "history": [{"date": f"{d:%Y-%m-%d}", "cost": float(c), "partial_day": d == snap.today}
                    for d, c in history.items()],
        "forecast": [{"date": f"{d:%Y-%m-%d}", "predicted": float(p), "lower": float(max(p - Z_95 * sigma, 0)),
                      "upper": float(p + Z_95 * sigma)} for d, p in zip(forecast_dates, predicted)],
        "month_to_date": month_to_date,
        "projected_month_end": month_to_date + float(remaining),
        "projected_range": [month_to_date + float(remaining) - float(margin), month_to_date + float(remaining) + float(margin)],
        "remaining_month_forecast": float(remaining),
        "month_start": f"{snap.month_start:%Y-%m-%d}", "month_end": f"{month_end:%Y-%m-%d}",
        "backtest": evaluation,
        "new_resource_run_rates": new_rates,
        "configuration_adjustments": adjustments,
        "uncertainty_note": "Range = forecast +/- 1.96 x standard deviation of the 14-day backtest errors "
                            "(approximate 95% band).",
        "method_note": "The model is fitted on complete days of simulated spend. History of stopped resources is "
                       "removed and resized resources are re-priced, so the forecast reflects the current setup. "
                       f"Resources launched in the last {NEW_RESOURCE_DAYS} days are added at their current run rate. "
                       "Days reported as cost anomalies are replaced by the resource's typical daily cost so a "
                       "one-off spike is not repeated in the forecast.",
        "lstm": lstm_status(),
    }
    snap._forecast_result = result
    return result


def budget_view(snap: Snapshot, budget: float) -> dict:
    forecast = build_forecast(snap)
    projected = forecast["projected_month_end"]
    utilization = 100 * forecast["month_to_date"] / budget
    projected_pct = 100 * projected / budget
    if projected_pct > 100:
        status = "over_budget"
    elif projected_pct >= 90:
        status = "at_risk"
    else:
        status = "on_track"
    days_in_month = snap.month_end.day
    elapsed_days = (snap.today - snap.month_start).days + snap.clock.hour / 24
    return {
        "monthly_budget": budget, "month_to_date": forecast["month_to_date"], "utilization_pct": utilization,
        "projected_month_end": projected, "projected_pct": projected_pct, "variance": projected - budget,
        "status": status, "expected_utilization_pct": 100 * elapsed_days / days_in_month,
    }
