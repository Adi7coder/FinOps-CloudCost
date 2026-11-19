
import pandas as pd

from preprocess_features import DATA, load_preprocessed

SAVINGS_FRACTION = {
    "RETAIN": 0.00,
    "DOWNSIZE": 0.45,
    "STOP_REVIEW": 0.85,
}
MONTHLY_WINDOW_DAYS = 30


def recommend(avg_cpu, avg_mem, avg_hours):
    if avg_cpu < 5 and avg_mem < 10 and avg_hours < 9:
        return "STOP_REVIEW"     # nearly idle: candidate for stop, after human review

    if avg_cpu < 20 and avg_mem < 30 and avg_hours >= 9:
        return "DOWNSIZE"        # running but under-utilized: candidate for a smaller size

    return "RETAIN"


def monthly_cost_from_raw() -> pd.Series:
    df = load_preprocessed()
    window_start = df["date"].max() - pd.Timedelta(days=MONTHLY_WINDOW_DAYS)
    last_30_days = df[df["date"] > window_start]
    return last_30_days.groupby(["cloud", "service"])["cost_inr"].sum()


def apply_engine(groups: pd.DataFrame) -> pd.DataFrame:
    out = groups.copy()
    out["recommendation_recomputed"] = [
        recommend(row.avg_cpu, row.avg_mem, row.avg_hours) for row in out.itertuples()
    ]
    out["savings_fraction"] = out["recommendation_recomputed"].map(SAVINGS_FRACTION)
    out["estimated_saving_recomputed"] = out["monthly_cost"] * out["savings_fraction"]
    return out


def summarize(result: pd.DataFrame) -> dict:
    actionable = result[result["recommendation_recomputed"] != "RETAIN"]
    total_saving = actionable["estimated_saving_recomputed"].sum()
    return {
        "accuracy": (result["recommendation_recomputed"] == result["gold_action"]).mean(),
        "actionable_resources": len(actionable),
        "total_resources": len(result),
        "estimated_monthly_savings_INR": total_saving,
        "portfolio_saving_pct": 100 * total_saving / result["monthly_cost"].sum(),
    }


def main():
    groups = pd.read_csv(DATA / "rightsizing_recommendations.csv")
    canonical = pd.read_csv(DATA / "rightsizing_metrics.csv").iloc[0]

    raw_monthly = monthly_cost_from_raw()
    artifact_monthly = groups.set_index(["cloud", "service"])["monthly_cost"].loc[raw_monthly.index]
    monthly_ok = ((artifact_monthly - raw_monthly).abs() < 1e-6).all()
    print(f"monthly_cost equals last-{MONTHLY_WINDOW_DAYS}-day spend from raw data: {monthly_ok}\n")

    result = apply_engine(groups)
    columns = ["resource_id", "avg_cpu", "avg_mem", "avg_hours", "monthly_cost",
               "recommendation_recomputed", "gold_action", "estimated_saving_recomputed"]
    print(result[columns].round(2).to_string(index=False))

    same_recommendations = (result["recommendation_recomputed"] == result["recommendation"]).all()
    print(f"\nrecommendations identical to artifact: {same_recommendations}")

    metrics = summarize(result)
    print(f"\n{'metric':32s}{'recomputed':>20s}{'canonical':>20s}")
    for name, value in metrics.items():
        print(f"{name:32s}{value:>20.10f}{float(canonical[name]):>20.10f}")
    agree = int(round(metrics["accuracy"] * metrics["total_resources"]))
    print(f"\nagreement with gold actions: {agree}/{metrics['total_resources']}")
    print("Savings are estimated benchmark optimization opportunities, not realized savings; "
          "no automatic remediation is performed.")


if __name__ == "__main__":
    main()
