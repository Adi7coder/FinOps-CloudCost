
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
HOLDOUT_DAYS = 45
FEATURES = ["cost_ratio", "cost_z", "cpu_avg_pct", "memory_avg_pct", "usage_hours", "dayofweek"]

failures = []


def report(name, recomputed, stored, tol=1e-6):
    ok = abs(float(recomputed) - float(stored)) <= tol
    print(f"[{'MATCH' if ok else 'DIFF '}] {name}: recomputed={recomputed:.6f} stored={float(stored):.6f}")
    if not ok:
        failures.append(name)


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df.isna().any().any():
        raise ValueError("dataset contains missing values")
    return df.sort_values(["date", "record_id"]).reset_index(drop=True)


def add_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby("service")["cost_inr"]
    out["svc_median"] = grp.transform("median")
    out["svc_std"] = grp.transform("std")
    out["cost_ratio"] = out["cost_inr"] / out["svc_median"]
    out["cost_z"] = (out["cost_inr"] - out["svc_median"]) / out["svc_std"]
    out["dayofweek"] = out["date"].dt.dayofweek
    return out


def forecast_errors(actual, pred):
    return (100 * np.mean(np.abs((actual - pred) / actual)),
            np.sqrt(np.mean((actual - pred) ** 2)),
            np.mean(np.abs(actual - pred)))


def rightsizing_rule(row) -> str:
    if row.avg_cpu < 5 and row.avg_mem < 10 and row.avg_hours < 9:
        return "STOP_REVIEW"
    if row.avg_cpu < 20 and row.avg_mem < 30 and row.avg_hours >= 9:
        return "DOWNSIZE"
    return "RETAIN"


def main():
    raw = pd.read_csv(DATA / "multi_cloud_finops_dataset.csv")
    df = preprocess(raw)
    summary = pd.read_csv(DATA / "experiment_summary.csv")
    S = dict(zip(summary.iloc[:, 0], summary.iloc[:, 1]))

    print("== Dataset")
    print(f"rows={len(df)} days={df['date'].nunique()} clouds={df['cloud'].nunique()} services={df['service'].nunique()} "
          f"anomalies={int(df['anomaly_label'].sum())} range={df['date'].min().date()} -> {df['date'].max().date()}")
    report("true anomaly rate %", 100 * df["anomaly_label"].mean(), S["true_anomaly_rate_pct"])

    print("\n== Forecasting (chronological split, no shuffling)")
    daily = df.groupby("date")["cost_inr"].sum().values
    train_days = len(daily) - HOLDOUT_DAYS
    actual = daily[train_days:]
    seasonal_naive = daily[train_days - 7:len(daily) - 7]
    stored = pd.read_csv(DATA / "forecast_metrics.csv").set_index("model")
    print(f"train days={train_days} holdout days={HOLDOUT_DAYS}")
    for label, value in zip(["MAPE_pct", "RMSE_INR", "MAE_INR"], forecast_errors(actual, seasonal_naive)):
        report(f"Seasonal Naive {label}", value, stored.loc["Seasonal Naive", label])
    for model in ["Additive Seasonal", "LSTM"]:
        row = stored.loc[model]
        print(f"[STORED] {model}: MAPE={row['MAPE_pct']:.6f}% RMSE={row['RMSE_INR']:.6f} MAE={row['MAE_INR']:.6f} (not retrained)")

    print("\n== Isolation Forest (unsupervised; labels used only for evaluation)")
    feats = add_anomaly_features(df)
    model = IsolationForest(n_estimators=300, contamination=45 / 3240, random_state=42).fit(feats[FEATURES])
    pred = (model.predict(feats[FEATURES]) == -1).astype(int)
    scored = pd.read_csv(DATA / "anomaly_scored_dataset.csv")
    same_pred = bool((pred == scored["iforest_pred"].values).all())
    same_score = bool(np.allclose(-model.score_samples(feats[FEATURES]), scored["anomaly_score"].values))
    print(f"[{'MATCH' if same_pred else 'DIFF '}] predictions identical to anomaly_scored_dataset.csv")
    print(f"[{'MATCH' if same_score else 'DIFF '}] anomaly scores identical to anomaly_scored_dataset.csv")
    if not (same_pred and same_score):
        failures.append("Isolation Forest predictions/scores")
    y = feats["anomaly_label"].values
    tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    print(f"TP={tp} FP={fp} FN={fn}")
    am = pd.read_csv(DATA / "anomaly_metrics.csv").iloc[0]
    report("precision", precision, am["precision"])
    report("recall", recall, am["recall"])
    report("F1", 2 * precision * recall / (precision + recall), am["f1"])

    print("\n== Rightsizing (transparent rules)")
    recs = pd.read_csv(DATA / "rightsizing_recommendations.csv")
    rm = pd.read_csv(DATA / "rightsizing_metrics.csv").iloc[0]
    recomputed = recs.apply(rightsizing_rule, axis=1)
    rules_ok = bool((recomputed == recs["recommendation"]).all())
    print(f"[{'MATCH' if rules_ok else 'DIFF '}] rules reproduce all {len(recs)} stored recommendations")
    if not rules_ok:
        failures.append("rightsizing rules")
    last30 = df[df["date"] > df["date"].max() - pd.Timedelta(days=30)]
    monthly = last30.groupby(["cloud", "service"])["cost_inr"].sum()
    cost_ok = bool(np.allclose(recs.set_index(["cloud", "service"]).loc[monthly.index, "monthly_cost"], monthly.values))
    print(f"[{'MATCH' if cost_ok else 'DIFF '}] monthly_cost equals last-30-day spend per resource group")
    if not cost_ok:
        failures.append("rightsizing monthly_cost")
    actionable = recs[recomputed != "RETAIN"]
    report("agreement with gold actions", (recomputed == recs["gold_action"]).mean(), rm["accuracy"])
    report("estimated monthly savings INR", actionable["estimated_saving"].sum(), rm["estimated_monthly_savings_INR"], 1e-4)
    report("portfolio saving %", 100 * actionable["estimated_saving"].sum() / recs["monthly_cost"].sum(), rm["portfolio_saving_pct"])
    print(f"actionable resources={len(actionable)} of {len(recs)}")

    print()
    if failures:
        print(f"REPRODUCTION DIFFERENCES: {failures}")
        return 1
    print("ALL REPRODUCIBLE RESULTS MATCH THE STORED ARTIFACTS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
