from pathlib import Path

import pandas as pd
import streamlit as st

from ui import api
from ui.components import kpi, money_column, page_header, section
from ui.format import inr

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
FIGURES = [("forecast_comparison.png", "Forecast comparison on the 45-day holdout"),
           ("anomaly_detection.png", "Daily total cost with Isolation Forest flags"),
           ("rightsizing_savings.png", "Estimated monthly saving by resource group"),
           ("system_architecture.png", "System architecture")]
SOURCES = [
    ("Benchmark dataset (3,240 rows)", "data/multi_cloud_finops_dataset.csv", "scripts/generate_benchmark.py (reconstruction)"),
    ("Forecast metrics", "data/forecast_metrics.csv", "scripts/train_forecasting_models.py (Seasonal Naive exact; "
     "Additive Seasonal and LSTM stored results)"),
    ("Anomaly metrics", "data/anomaly_metrics.csv", "scripts/train_anomaly_detection.py (exact)"),
    ("Rightsizing metrics", "data/rightsizing_metrics.csv", "scripts/run_rightsizing.py (exact)"),
    ("Artifact checks", "all of the above", "scripts/validate_artifacts.py, scripts/reproduce_experiments.py"),
]


def render(state: dict):
    page_header("Model Evaluation (Review-2 benchmark)",
                "Archived results on the 270-day simulated benchmark dataset. They are separate from the live demo.")
    st.info("These numbers come from the stored Review-2 experiment files. They are not recomputed from the "
            "interactive simulation and do not change when you use the demo controls.", icon=":material/history:")
    summary = api.get("/api/v1/summary", state["state_version"])
    dataset, forecast = summary["dataset"], summary["forecast"]
    anomaly, rights = summary["anomaly_detection"], summary["rightsizing"]
    row = st.columns(5)
    kpi(row[0], "Benchmark records", f"{dataset['records']:,}", f"{dataset['days']} days · {dataset['clouds']} clouds · "
                                                                 f"{dataset['services']} services")
    kpi(row[1], "Best forecast MAPE", f"{forecast['mape_pct']:.2f}%", f"{forecast['best_model']} · RMSE {inr(forecast['rmse_inr'], 2)}")
    kpi(row[2], "Isolation Forest F1", f"{anomaly['f1']:.3f}", f"Precision {anomaly['precision']:.3f} · recall {anomaly['recall']:.3f}")
    kpi(row[3], "Rightsizing agreement", f"{100 * rights['agreement']:.2f}%", f"{rights['actionable_resources']} of "
                                                                              f"{rights['total_resources']} actionable")
    kpi(row[4], "Estimated savings", f"{inr(rights['estimated_monthly_savings_inr'], 2)}/mo",
        f"{rights['portfolio_saving_pct']:.2f}% of benchmark portfolio")

    left, right = st.columns(2)
    with left, st.container(border=True, key="panel-bench-forecast"):
        section("Forecasting models", "Chronological split: 225 training days, 45 holdout days")
        st.dataframe(pd.DataFrame(api.get("/api/v1/forecast/metrics", state["state_version"])).round(4), hide_index=True,
                     width="stretch")
        st.caption("LSTM metrics are the stored experiment result; the LSTM is not retrained by this application.")
    with right, st.container(border=True, key="panel-bench-rightsizing"):
        section("Rightsizing on the benchmark", "12 aggregated resource groups")
        table = pd.DataFrame(api.get("/api/v1/rightsizing", state["state_version"], actionable_only="false")["items"])
        st.dataframe(table[["resource_id", "avg_cpu", "avg_mem", "avg_hours", "recommendation", "estimated_saving"]].round(2),
                     hide_index=True, width="stretch", column_config={"estimated_saving": money_column("Est. saving (₹)")})

    with st.container(border=True, key="panel-bench-figures"):
        section("Review-2 figures", "Stored in figures/")
        tabs = st.tabs([caption for _, caption in FIGURES])
        for tab, (name, caption) in zip(tabs, FIGURES):
            with tab:
                path = FIGURES_DIR / name
                if path.exists():
                    st.image(str(path), caption=caption, width="stretch")
                else:
                    st.error(f"Missing figure: figures/{name}")

    with st.container(border=True, key="panel-bench-sources"):
        section("Where each result comes from")
        st.dataframe(pd.DataFrame(SOURCES, columns=["Result", "Stored in", "Reproduced by"]), hide_index=True, width="stretch")
        st.caption(summary["disclaimer"])
