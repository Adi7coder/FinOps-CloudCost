import altair as alt
import pandas as pd
import streamlit as st

from ui import api
from ui.components import (ACCENT, CRITICAL, MUTED, TEAL, kpi, money_axis, money_column, page_header, run_action,
                           section, show_chart)
from ui.format import day_label, inr, pct

STATUS_MESSAGE = {
    "on_track": ("success", "Projected spend is within the monthly budget."),
    "at_risk": ("warning", "Projected spend is within 10% of the monthly budget."),
    "over_budget": ("error", "Projected spend is above the monthly budget."),
}


def daily_forecast_chart(data: dict):
    history = pd.DataFrame(data["history"])
    forecast = pd.DataFrame(data["forecast"])
    today = history["date"].iloc[-1]
    future = forecast[forecast["date"] > today]
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0, tickCount=10))
    actual = alt.Chart(history[~history["partial_day"]]).mark_line(color=ACCENT, strokeWidth=2).encode(
        x=x, y=alt.Y("cost:Q", axis=money_axis(), scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y", title="Date"), alt.Tooltip("cost:Q", format=",.0f", title="Actual (₹)")])
    band = alt.Chart(future).mark_area(color=TEAL, opacity=0.15).encode(x=x, y="lower:Q", y2="upper:Q")
    line = alt.Chart(future).mark_line(color=TEAL, strokeDash=[5, 4], strokeWidth=2).encode(
        x=x, y="predicted:Q",
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y", title="Date"), alt.Tooltip("predicted:Q", format=",.0f", title="Forecast (₹)"),
                 alt.Tooltip("lower:Q", format=",.0f", title="Low (₹)"), alt.Tooltip("upper:Q", format=",.0f", title="High (₹)")])
    show_chart(alt.layer(band, actual, line).properties(height=300))
    st.caption("Blue = actual daily spend (complete days). Teal dashed = forecast, shaded = approximate 95% range.")


def cumulative_chart(data: dict, budget: float):
    history = pd.DataFrame(data["history"])
    history = history[history["date"] >= data["month_start"]]
    actual = history.assign(cumulative=history["cost"].cumsum(), series="Actual")[["date", "cumulative", "series"]]
    forecast = pd.DataFrame(data["forecast"])
    future = forecast[(forecast["date"] > actual["date"].iloc[-1]) & (forecast["date"] <= data["month_end"])]
    frame = actual
    if not future.empty:
        rest_of_today = data["remaining_month_forecast"] - future["predicted"].sum()
        anchor = actual.iloc[[-1]].assign(series="Projected")
        projected = future.assign(series="Projected", cumulative=actual["cumulative"].iloc[-1] + rest_of_today
                                  + future["predicted"].cumsum())[["date", "cumulative", "series"]]
        frame = pd.concat([actual, anchor, projected])
    x = alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0))
    lines = alt.Chart(frame).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=20)).encode(
        x=x, y=alt.Y("cumulative:Q", axis=money_axis()),
        color=alt.Color("series:N", scale=alt.Scale(domain=["Actual", "Projected"], range=[ACCENT, TEAL]),
                        legend=alt.Legend(title=None)),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=["Actual", "Projected"], range=[[1, 0], [5, 4]]),
                                  legend=None),
        tooltip=[alt.Tooltip("date:T", format="%d %b", title="Date"), alt.Tooltip("series:N"),
                 alt.Tooltip("cumulative:Q", format=",.0f", title="Cumulative (₹)")])
    rule = alt.Chart(pd.DataFrame({"budget": [budget]})).mark_rule(color=CRITICAL, strokeDash=[3, 3], strokeWidth=1.5)         .encode(y="budget:Q", tooltip=[alt.Tooltip("budget:Q", format=",.0f", title="Budget (₹)")])
    show_chart(alt.layer(lines, rule).properties(height=300))
    st.caption("Cumulative spend this month against the budget (red dashed line). Today's point is partial.")


def render(state: dict):
    page_header("Forecast & Budget", "Month-end projection from the simulated spending history, compared with the budget")
    data = api.get("/api/v1/demo/forecast", state["state_version"])
    budget = data["budget"]
    row = st.columns(5)
    kpi(row[0], "Month to date", inr(data["month_to_date"]), f"{pct(budget['utilization_pct'])} of budget used")
    kpi(row[1], "Projected month-end", inr(data["projected_month_end"]),
        f"Range {inr(data['projected_range'][0])} - {inr(data['projected_range'][1])}")
    kpi(row[2], "Monthly budget", inr(budget["monthly_budget"]))
    over = budget["variance"] > 0
    kpi(row[3], "Expected over budget" if over else "Expected under budget", inr(abs(budget["variance"])),
        f"{budget['projected_pct']:.1f}% of budget", delta_color="inverse" if over else "off")
    kpi(row[4], "Forecast model", data["model"], "Lowest backtest MAPE")

    kind, message = STATUS_MESSAGE[budget["status"]]
    getattr(st, kind)(f"{message} Projected {inr(data['projected_month_end'])} against {inr(budget['monthly_budget'])}"
                      + (f" - about {inr(budget['variance'])} over. See the Optimization page for savings." if over else "."))

    left, right = st.columns(2)
    with left, st.container(border=True, key="panel-forecast-daily"):
        section("Daily spend and forecast", "Last 60 days of actual spend and the forecast to month end")
        daily_forecast_chart(data)
    with right, st.container(border=True, key="panel-forecast-cumulative"):
        section("This month against the budget")
        cumulative_chart(data, budget["monthly_budget"])

    with st.container(border=True, key="panel-budget-edit"):
        section("Budget planning", "Change the monthly budget to see the effect on budget status")
        columns = st.columns([1.2, 1, 3])
        new_budget = columns[0].number_input("Monthly budget (₹)", min_value=1000.0, max_value=100000000.0,
                                             value=float(budget["monthly_budget"]), step=10000.0, key="budget_input")
        columns[1].write("")
        if columns[1].button("Save budget", type="primary", key="save_budget", width="stretch"):
            run_action("PUT", "/api/v1/demo/budget", {"monthly_budget": new_budget},
                       lambda result: f"Budget saved: {inr(result['monthly_budget'])}. Status is now "
                                      f"{result['status'].replace('_', ' ')}.")
        with columns[2]:
            st.caption(f"Elapsed: {pct(budget['expected_utilization_pct'])} of the month · used: "
                       f"{pct(budget['utilization_pct'])} of the budget. Spending faster than the calendar is an early "
                       "warning even before the projection crosses the budget.")

    with st.container(border=True, key="panel-forecast-method"):
        section("How the projection is calculated")
        st.markdown(f"- {data['selection_note']}\n- {data['method_note']}\n- {data['uncertainty_note']}")
        adjustments = data["configuration_adjustments"]
        if adjustments:
            st.markdown("**Adjusted for configuration changes:** " + "; ".join(
                f"{a['name']} ({a['status']}): -{inr(a['removed_daily_cost'])}/day" for a in adjustments))
        if data["new_resource_run_rates"]:
            st.markdown("**New resources added at run rate:** " + "; ".join(
                f"{r['name']}: {inr(r['daily_run_rate'])}/day" for r in data["new_resource_run_rates"]))
        backtest = data["backtest"]
        if backtest:
            left, right = st.columns([1, 1.3])
            with left:
                section(f"Live backtest on the last {backtest['days']} complete days")
                metrics = pd.DataFrame(backtest["metrics"]).round(2)
                st.dataframe(metrics, hide_index=True, width="stretch",
                             column_config={"model": "Model", "MAPE_pct": "MAPE (%)", "RMSE_INR": money_column("RMSE (₹)"),
                                            "MAE_INR": money_column("MAE (₹)")})
                st.caption(f"LSTM: {data['lstm']}")
            with right:
                frame = pd.DataFrame(backtest["series"]).melt("date", var_name="series", value_name="cost")
                frame["series"] = frame["series"].map({"actual": "Actual", "additive_seasonal": "Additive Seasonal",
                                                       "seasonal_naive": "Seasonal Naive"})
                chart = alt.Chart(frame).mark_line(point=True, strokeWidth=2).encode(
                    x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0)),
                    y=alt.Y("cost:Q", axis=money_axis(), scale=alt.Scale(zero=False)),
                    color=alt.Color("series:N", scale=alt.Scale(domain=["Actual", "Seasonal Naive", "Additive Seasonal"],
                                                                range=[ACCENT, MUTED, TEAL]), legend=alt.Legend(title=None)),
                    tooltip=["date:T", "series:N", alt.Tooltip("cost:Q", format=",.0f")])
                show_chart(chart.properties(height=230))
                st.caption("Actual vs predicted daily spend on the backtest window.")

    with st.container(border=True, key="panel-forecast-benchmark"):
        section("Review-2 benchmark evaluation (archived)",
                "45-day holdout on the separate 270-day benchmark dataset. These are stored results, not this simulation.")
        benchmark = pd.DataFrame(api.get("/api/v1/forecast/metrics", state["state_version"])).round(4)
        st.dataframe(benchmark, hide_index=True, width="stretch",
                     column_config={"model": "Model", "MAPE_pct": "MAPE (%)", "RMSE_INR": "RMSE (₹)", "MAE_INR": "MAE (₹)"})
        st.caption(f"Benchmark source: data/forecast_metrics.csv. Month shown: {day_label(data['month_start'])} to "
                   f"{day_label(data['month_end'])}.")
