
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
FIG = BASE / "figures"
TOL = 1e-5

CSVS = [
    "multi_cloud_finops_dataset.csv",
    "anomaly_scored_dataset.csv",
    "forecast_metrics.csv",
    "anomaly_metrics.csv",
    "rightsizing_metrics.csv",
    "rightsizing_recommendations.csv",
    "experiment_summary.csv",
]
FIGURES = ["system_architecture.png", "forecast_comparison.png", "anomaly_detection.png", "rightsizing_savings.png"]
DATASET_COLUMNS = ["date", "cloud", "service", "team", "region", "cost_inr", "cpu_avg_pct", "memory_avg_pct",
                   "usage_hours", "anomaly_label", "anomaly_type"]
EXPECTED_SERVICES = {
    "AWS": {"EC2", "RDS", "S3", "EKS"},
    "Azure": {"VirtualMachines", "SQLDatabase", "BlobStorage", "AKS"},
    "GCP": {"ComputeEngine", "CloudSQL", "CloudStorage", "GKE"},
}
FORECAST = {
    "Seasonal Naive": (6.917335, 1207.332632, 963.200889),
    "Additive Seasonal": (3.044425, 587.700308, 426.683423),
    "LSTM": (8.447425, 1474.514274, 1167.730429),
}

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


def close(actual, expected, tol=TOL):
    return abs(float(actual) - float(expected)) <= tol


def main():
    print("== Artifact files")
    frames = {}
    for name in CSVS:
        path = DATA / name
        try:
            frames[name] = pd.read_csv(path)
            check(f"load data/{name}", True, f"{len(frames[name])} rows")
        except Exception as exc:
            check(f"load data/{name}", False, str(exc))
    for name in FIGURES:
        path = FIG / name
        check(f"figure figures/{name}", path.exists() and path.stat().st_size > 0)
    # Presence only: the generator is a deterministic reconstruction (not value-identical), so its comparison
    # against the canonical CSV is run separately via `python scripts/generate_benchmark.py`.
    check("benchmark generator scripts/generate_benchmark.py", (BASE / "scripts" / "generate_benchmark.py").is_file())
    if len(frames) != len(CSVS):
        return finish()

    print("\n== Dataset")
    d = frames["multi_cloud_finops_dataset.csv"]
    missing_cols = [c for c in DATASET_COLUMNS if c not in d.columns]
    check("required columns present", not missing_cols, f"missing: {missing_cols}" if missing_cols else "")
    if missing_cols:
        return finish()
    check("rows == 3240", len(d) == 3240, str(len(d)))
    check("unique dates == 270", d["date"].nunique() == 270, str(d["date"].nunique()))
    check("clouds == 3", d["cloud"].nunique() == 3, str(sorted(d["cloud"].unique())))
    check("services == 12", d["service"].nunique() == 12, str(d["service"].nunique()))
    actual_services = {c: set(g["service"]) for c, g in d.groupby("cloud")}
    check("provider/service mapping", actual_services == EXPECTED_SERVICES, str({k: sorted(v) for k, v in actual_services.items()}))
    check("anomaly_label == 1 count == 45", int(d["anomaly_label"].sum()) == 45, str(int(d["anomaly_label"].sum())))
    check("anomaly prevalence == 1.3888888889%", close(100 * d["anomaly_label"].mean(), 1.3888888889, 1e-8))
    check("no missing values", int(d[DATASET_COLUMNS].isna().sum().sum()) == 0)
    check("parseable dates", pd.to_datetime(d["date"], errors="coerce").notna().all(), f"{d['date'].min()} -> {d['date'].max()}")
    check("one row per date/cloud/service", not d.duplicated(["date", "cloud", "service"]).any())

    print("\n== Forecast metrics")
    f = frames["forecast_metrics.csv"].set_index("model")
    check("exactly 3 forecast models", sorted(f.index) == sorted(FORECAST), str(list(f.index)))
    for model, (mape, rmse, mae) in FORECAST.items():
        if model in f.index:
            row = f.loc[model]
            ok = close(row["MAPE_pct"], mape) and close(row["RMSE_INR"], rmse) and close(row["MAE_INR"], mae)
            check(f"{model} MAPE/RMSE/MAE", ok, f"{row['MAPE_pct']:.6f}% / {row['RMSE_INR']:.6f} / {row['MAE_INR']:.6f}")
    check("best model == Additive Seasonal", f["MAPE_pct"].idxmin() == "Additive Seasonal")

    print("\n== Anomaly detection")
    a = frames["anomaly_metrics.csv"].iloc[0]
    for col, exp in [("precision", 0.888889), ("recall", 0.888889), ("f1", 0.888889)]:
        check(f"Isolation Forest {col} == {exp}", close(a[col], exp), f"{a[col]:.6f}")
    check("true_anomalies == 45 and flagged == 45", int(a["true_anomalies"]) == 45 and int(a["flagged"]) == 45)
    s = frames["anomaly_scored_dataset.csv"]
    check("scored dataset matches raw dataset", s[list(d.columns)].equals(d))
    tp = int(((s["iforest_pred"] == 1) & (s["anomaly_label"] == 1)).sum())
    fp = int(((s["iforest_pred"] == 1) & (s["anomaly_label"] == 0)).sum())
    fn = int(((s["iforest_pred"] == 0) & (s["anomaly_label"] == 1)).sum())
    check("confusion TP/FP/FN == 40/5/5", (tp, fp, fn) == (40, 5, 5), f"TP={tp} FP={fp} FN={fn}")
    p, r = tp / (tp + fp), tp / (tp + fn)
    check("predictions reproduce stored precision/recall/F1",
          close(p, a["precision"], 1e-9) and close(r, a["recall"], 1e-9) and close(2 * p * r / (p + r), a["f1"], 1e-9))

    print("\n== Rightsizing")
    rm = frames["rightsizing_metrics.csv"].iloc[0]
    rr = frames["rightsizing_recommendations.csv"]
    check("agreement == 0.916667", close(rm["accuracy"], 0.916667), f"{rm['accuracy']:.6f}")
    check("agreement == 11/12 (gold vs recommendation)", int((rr["gold_action"] == rr["recommendation"]).sum()) == 11 and len(rr) == 12)
    check("actionable == 5, total == 12", int(rm["actionable_resources"]) == 5 and int(rm["total_resources"]) == 12)
    check("monthly savings == 88331.6925", close(rm["estimated_monthly_savings_INR"], 88331.6925, 1e-4), f"{rm['estimated_monthly_savings_INR']:.4f}")
    check("portfolio saving == 21.247132%", close(rm["portfolio_saving_pct"], 21.247132), f"{rm['portfolio_saving_pct']:.6f}")
    actionable = rr[rr["recommendation"] != "RETAIN"]
    check("recommendations CSV sums to stored savings", close(actionable["estimated_saving"].sum(), rm["estimated_monthly_savings_INR"], 1e-6))
    check("recommendations CSV reproduces portfolio %", close(100 * actionable["estimated_saving"].sum() / rr["monthly_cost"].sum(), rm["portfolio_saving_pct"], 1e-6))
    check("recommendation labels in {RETAIN, DOWNSIZE, STOP_REVIEW}", set(rr["recommendation"]) <= {"RETAIN", "DOWNSIZE", "STOP_REVIEW"})

    def rule(x):
        if x.avg_cpu < 5 and x.avg_mem < 10 and x.avg_hours < 9:
            return "STOP_REVIEW"
        if x.avg_cpu < 20 and x.avg_mem < 30 and x.avg_hours >= 9:
            return "DOWNSIZE"
        return "RETAIN"
    check("documented rules reproduce every recommendation", (rr.apply(rule, axis=1) == rr["recommendation"]).all())

    print("\n== Experiment summary consistency")
    es = frames["experiment_summary.csv"]
    S = dict(zip(es.iloc[:, 0], es.iloc[:, 1]))
    expected = {
        "records": 3240, "days": 270, "clouds": 3, "services": 12, "true_anomaly_rate_pct": 1.3888888889,
        "forecast_holdout_days": 45, "best_mape_pct": 3.044425, "best_rmse_inr": 587.700308,
        "if_precision": 0.888889, "if_recall": 0.888889, "if_f1": 0.888889, "rightsizing_accuracy": 0.916667,
        "estimated_monthly_savings_inr": 88331.6925, "portfolio_saving_pct": 21.247132,
    }
    for key, exp in expected.items():
        check(f"summary {key} == {exp}", key in S and close(S[key], exp, 1e-4 if key == "estimated_monthly_savings_inr" else TOL), str(S.get(key)))
    check("summary best_forecast_model == Additive Seasonal", S.get("best_forecast_model") == "Additive Seasonal")

    return finish()


def finish():
    print()
    if failures:
        print(f"VALIDATION FAILED: {len(failures)} check(s) failed:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("ALL ARTIFACT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
