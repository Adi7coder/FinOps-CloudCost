
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # file output only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch

from preprocess_features import BASE, DATA, build_daily_forecast_series, load_preprocessed

FIGURES = BASE / "figures"
DEFAULT_OUTPUT_DIR = FIGURES / "recreated"
DPI = 220                       
FIGSIZE = (7.2, 3.8)
ACTION_COLORS = {"STOP_REVIEW": "#d62728", "DOWNSIZE": "#ff7f0e", "RETAIN": "#7f7f7f"}


def save(fig, output_dir: Path, name: str):
    path = output_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(BASE)}")


# A. system architecture

def draw_box(ax, x, y, w, h, text, face="#ffffff", edge="black", bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor=edge, linewidth=1.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold" if bold else "normal")


def draw_arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.6))


def plot_system_architecture(output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")

    w, h = 6.4, 1.3
    draw_box(ax, 5, 19, w, h, "Multi-Cloud Cost & Usage Data\nAWS | Azure | GCP", face="#e8f0fe", bold=True)
    draw_box(ax, 5, 16.9, w, h, "Preprocessing & Schema Normalization")
    draw_box(ax, 5, 14.8, w, h, "Feature Engineering\ncost_ratio, cost_z, utilization, day-of-week, daily cost")
    draw_box(ax, 2.75, 12.5, 4.1, 1.5, "Forecasting Engine\nSeasonal Naive |\nAdditive Seasonal | LSTM", face="#eef7ee")
    draw_box(ax, 7.6, 12.5, 3.2, 1.5, "Isolation Forest\nAnomaly Detection", face="#eef7ee")
    draw_box(ax, 5, 10.2, w, h, "Explainable Rightsizing / FinOps Decision Engine\nRETAIN | DOWNSIZE | STOP_REVIEW",
             face="#fff4e5")
    draw_box(ax, 5, 8.1, w, h, "FastAPI REST Backend")
    draw_box(ax, 5, 6.0, w, h, "Streamlit Dashboard")
    draw_box(ax, 5, 3.9, w, h, "Human Review / Approval", face="#e8f0fe", bold=True)
    draw_box(ax, 5, 1.8, w, h, "NO automatic remediation in Phase-I", face="#fde8e8", edge="#d62728", bold=True)

    draw_arrow(ax, (5, 18.35), (5, 17.55))
    draw_arrow(ax, (5, 16.25), (5, 15.45))
    draw_arrow(ax, (4.2, 14.15), (2.75, 13.25))
    draw_arrow(ax, (5.8, 14.15), (7.6, 13.25))
    draw_arrow(ax, (2.75, 11.75), (4.2, 10.85))
    draw_arrow(ax, (7.6, 11.75), (5.8, 10.85))
    for y_top in [9.55, 7.45, 5.35, 3.25]:
        draw_arrow(ax, (5, y_top), (5, y_top - 0.8))
    save(fig, output_dir, "system_architecture.png")


# B. forecasting

def plot_forecast_comparison(output_dir: Path, with_lstm: bool):
    # The Additive Seasonal line is the documented reconstruction; reported metrics stay in data/forecast_metrics.csv.
    import train_forecasting_models as tfm

    daily = build_daily_forecast_series(load_preprocessed())
    dates, values = daily.index, daily.values
    train_tail = slice(tfm.TRAIN_DAYS - tfm.HOLDOUT_DAYS, tfm.TRAIN_DAYS)
    holdout = slice(tfm.TRAIN_DAYS, len(values))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(dates[train_tail], values[train_tail], label="Actual (train)")
    ax.plot(dates[holdout], values[holdout], label="Actual (test)", linewidth=2)
    ax.plot(dates[holdout], tfm.additive_seasonal_forecast(values), "--", label="Additive forecast")
    if with_lstm:
        ax.plot(dates[holdout], tfm.train_lstm_forecast(values), ":", label="LSTM forecast", linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily cost (INR)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25)
    save(fig, output_dir, "forecast_comparison.png")


def plot_forecast_metric_comparison(output_dir: Path):
    metrics = pd.read_csv(DATA / "forecast_metrics.csv")
    best_model = metrics.loc[metrics["MAPE_pct"].idxmin(), "model"]
    colors = ["#2ca02c" if m == best_model else "#9fb7d4" for m in metrics["model"]]

    fig, (ax_mape, ax_rmse) = plt.subplots(1, 2, figsize=(9, 3.8))
    fig.subplots_adjust(wspace=0.35)
    for ax, column, label in [(ax_mape, "MAPE_pct", "MAPE (%)"), (ax_rmse, "RMSE_INR", "RMSE (INR)")]:
        bars = ax.bar(metrics["model"], metrics[column], color=colors)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"45-day holdout forecasting error (best: {best_model})", fontsize=11)
    save(fig, output_dir, "forecast_metric_comparison.png")


# anomaly detection

def load_scored() -> pd.DataFrame:
    scored = pd.read_csv(DATA / "anomaly_scored_dataset.csv")
    scored["date"] = pd.to_datetime(scored["date"])
    return scored


def plot_anomaly_detection(output_dir: Path):
    scored = load_scored()
    daily_cost = scored.groupby("date")["cost_inr"].sum()
    flagged_days = scored.loc[scored["iforest_pred"] == 1, "date"].unique()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(daily_cost.index, daily_cost.values, linewidth=1.2, label="Daily cloud cost")
    ax.scatter(flagged_days, daily_cost.loc[flagged_days], marker="x", s=30, zorder=3,
               label="Detected anomaly day")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily cost (INR)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25)
    save(fig, output_dir, "anomaly_detection.png")


def plot_anomaly_records(output_dir: Path):
    scored = load_scored()
    normal = scored[scored["iforest_pred"] == 0]
    flagged = scored[scored["iforest_pred"] == 1]
    injected = scored[scored["anomaly_label"] == 1]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(normal["date"], normal["cost_inr"], s=4, color="#9fb7d4", label="Normal (not flagged)")
    ax.scatter(flagged["date"], flagged["cost_inr"], s=22, color="#d62728", label="Isolation Forest flagged")
    ax.scatter(injected["date"], injected["cost_inr"], s=60, facecolors="none", edgecolors="black",
               linewidths=0.8, label="Injected anomaly (label)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Service-day cost (INR)")
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(axis="x", rotation=25)
    save(fig, output_dir, "anomaly_records.png")


def plot_confusion_matrix(output_dir: Path):
    from sklearn.metrics import ConfusionMatrixDisplay

    scored = load_scored()
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay.from_predictions(
        scored["anomaly_label"],
        scored["iforest_pred"],
        display_labels=["Normal", "Anomaly"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Isolation Forest confusion matrix", fontsize=10)
    save(fig, output_dir, "confusion_matrix.png")


# ---------------------------------------------------------------- D. rightsizing

TOP_RESOURCE_GROUPS = 10   # the canonical figure shows the 10 groups with the highest estimated saving


def plot_rightsizing_savings(output_dir: Path):
    groups = pd.read_csv(DATA / "rightsizing_recommendations.csv")
    total = pd.read_csv(DATA / "rightsizing_metrics.csv").iloc[0]["estimated_monthly_savings_INR"]
    top = groups.sort_values("estimated_saving", ascending=False).head(TOP_RESOURCE_GROUPS)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(top["resource_id"], top["estimated_saving"], color=top["recommendation"].map(ACTION_COLORS))
    legend_entries = [Patch(color=color, label=action) for action, color in ACTION_COLORS.items()]
    ax.set_title(f"Estimated Monthly Optimization Opportunity (benchmark estimate: INR {total:,.2f}/month)",
                 fontsize=9)
    ax.set_xlabel("Resource group")
    ax.set_ylabel("Estimated monthly saving (INR)")
    ax.legend(handles=legend_entries, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    save(fig, output_dir, "rightsizing_savings.png")


# additional cost breakdowns

def plot_cost_breakdown(output_dir: Path, group_by: str, file_name: str):
    df = load_preprocessed()
    totals = df.groupby(group_by)["cost_inr"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(totals.index, totals.values / 1e5)
    ax.set_xlabel(group_by.capitalize())
    ax.set_ylabel("Total cost (INR lakh, 270 days)")
    if group_by == "service":
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    save(fig, output_dir, file_name)


def main():
    parser = argparse.ArgumentParser(description="Recreate the Review-2 presentation figures.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--with-lstm", action="store_true", help="train the LSTM for the forecast plot (PyTorch)")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir == FIGURES.resolve():
        sys.exit("Refusing to write into figures/ - the canonical PNGs must not be overwritten.")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_system_architecture(output_dir)
    plot_forecast_comparison(output_dir, args.with_lstm)
    plot_forecast_metric_comparison(output_dir)
    plot_anomaly_detection(output_dir)
    plot_anomaly_records(output_dir)
    plot_confusion_matrix(output_dir)
    plot_rightsizing_savings(output_dir)
    plot_cost_breakdown(output_dir, "cloud", "cloud_cost_breakdown.png")
    plot_cost_breakdown(output_dir, "service", "service_cost_breakdown.png")


if __name__ == "__main__":
    main()
