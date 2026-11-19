
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
RAW_DATASET = DATA / "multi_cloud_finops_dataset.csv"

# Normalizing
REQUIRED_COLUMNS = [
    "record_id", "date", "cloud", "service", "team", "region",
    "cost_inr", "cpu_avg_pct", "memory_avg_pct", "usage_hours",
    "anomaly_label", "anomaly_type",
]
NUMERIC_COLUMNS = ["cost_inr", "cpu_avg_pct", "memory_avg_pct", "usage_hours"]

FEATURES = [
    "cost_ratio",
    "cost_z",
    "cpu_avg_pct",
    "memory_avg_pct",
    "usage_hours",
    "dayofweek",
]


def load_dataset(path: Path = RAW_DATASET) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame) -> dict:
   
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    missing_values = int(df[REQUIRED_COLUMNS].isna().sum().sum())
    duplicates = int(df.duplicated(["date", "cloud", "service"]).sum())
    if missing_values or duplicates:
        raise ValueError(f"Data integrity failure: {missing_values} missing values, {duplicates} duplicate records")

    return {
        "rows": len(df),
        "days": df["date"].nunique(),
        "clouds": df["cloud"].nunique(),
        "services": df["service"].nunique(),
        "missing_values": missing_values,
        "duplicate_records": duplicates,
        "true_anomalies": int(df["anomaly_label"].sum()),
    }


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
   
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    for column in NUMERIC_COLUMNS:
        out[column] = pd.to_numeric(out[column])
    out = out.sort_values(["date", "record_id"]).reset_index(drop=True)
    return out


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
  
    out = df.copy()
    service_cost = out.groupby("service")["cost_inr"]

    out["svc_median"] = service_cost.transform("median")   # typical daily cost of the service
    out["svc_std"] = service_cost.transform("std")         # spread of the service's daily cost
    out["cost_ratio"] = out["cost_inr"] / out["svc_median"]
    out["cost_z"] = (out["cost_inr"] - out["svc_median"]) / out["svc_std"]
    out["dayofweek"] = out["date"].dt.dayofweek            # 0 = Monday ... 6 = Sunday
    return out


def build_daily_forecast_series(df: pd.DataFrame) -> pd.Series:

    daily_cost = df.groupby("date")["cost_inr"].sum().sort_index()
    daily_cost.name = "daily_cost"
    return daily_cost


def load_preprocessed() -> pd.DataFrame:
    raw = load_dataset()
    validate_dataset(raw)
    return preprocess_dataset(raw)


def main():
    raw = load_dataset()
    report = validate_dataset(raw)
    df = preprocess_dataset(raw)
    features = build_anomaly_features(df)
    daily = build_daily_forecast_series(df)

    print("== Validation")
    for key, value in report.items():
        print(f"{key:18s}{value}")
    print(f"date range        {df['date'].min().date()} -> {df['date'].max().date()}")

    print("\n== Anomaly features (first 3 rows)")
    print(features[["date", "cloud", "service"] + FEATURES].head(3).to_string(index=False))

    print("\n== Daily forecast series")
    print(f"{len(daily)} daily values, mean INR {daily.mean():,.2f}, "
          f"min INR {daily.min():,.2f}, max INR {daily.max():,.2f}")


if __name__ == "__main__":
    main()
