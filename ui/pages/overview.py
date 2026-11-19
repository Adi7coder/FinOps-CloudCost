import html

import altair as alt
import pandas as pd
import streamlit as st

from ui import api
from ui.components import (ACCENT, bar_list, kpi, money_axis, money_column, page_header, pill, provider_scale,
                           section, show_chart)
from ui.format import day_label, inr, inr_compact, ist, pct

BUDGET_TEXT = {"on_track": "On track", "at_risk": "At risk", "over_budget": "Over budget"}


def spend_chart(data: dict):
    actual = pd.DataFrame(data["daily"])
    forecast = pd.DataFrame(data["forecast"])
    today = data["clock"][:10]
    bars = alt.Chart(actual).mark_bar(size=11, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0, tickCount=10, grid=False)),
        y=alt.Y("sum(cost):Q", axis=money_axis()),
        color=alt.Color("provider:N", scale=provider_scale(), legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y", title="Date"), alt.Tooltip("provider:N", title="Provider"),
                 alt.Tooltip("cost:Q", format=",.0f", title="Cost (₹)")],
    )
    future = forecast[forecast["date"] > today]
    line = alt.Chart(future).mark_line(color="#475569", strokeDash=[5, 4], strokeWidth=2,
                                       point=alt.OverlayMarkDef(size=28, color="#475569")).encode(
        x="date:T", y="predicted:Q",
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y", title="Date"),
                 alt.Tooltip("predicted:Q", format=",.0f", title="Forecast (₹)")])
    return alt.layer(bars, line).properties(height=310)


def attention_panel(items: list):
    if not items:
        st.success("Nothing needs attention right now.")
        return
    rows = []
    for item in items:
        impact = []
        if item.get("unnecessary_spend_7d"):
            label = "Excess cost" if item["page"] == "Anomalies" else "Spent while idle (7 days)"
            impact.append(f"{label}: <b>{inr(item['unnecessary_spend_7d'])}</b>")
        if item.get("monthly_impact"):
            label = "Over budget by" if item["page"] == "Forecast" else "Possible saving"
            impact.append(f"{label}: <b>{inr(item['monthly_impact'])}/month</b>")
        rows.append(
            f'<div class="fo-item">{pill("Critical" if item["severity"] == "critical" else "Warning", item["severity"])}'
            f'<span class="fo-item-title">{html.escape(item["title"])}</span>'
            f'<div class="fo-item-detail">{html.escape(item["detail"])}</div>'
            f'<div class="fo-item-detail">{" · ".join(impact)}{" · " if impact else ""}Duration: {html.escape(item["duration"])}'
            f' · Next step: <b>{html.escape(item["recommended_action"])}</b> ({html.escape(item["page"])} page)</div></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)


def render(state: dict):
    data = api.get("/api/v1/demo/overview", state["state_version"])
    spend, budget = data["spend"], data["budget"]
    page_header("Overview", f"{state['company']} · simulation time {ist(state['clock'])} · {state['scenario_title']}")

    row = st.columns(5)
    kpi(row[0], "Spend today", inr(spend["today"]), f"First {spend['hours_into_today']} h · yesterday {inr(spend['yesterday'])}")
    kpi(row[1], "Spend this week", inr(spend["this_week"]), f"Since Mon {day_label(spend['week_start'])}")
    kpi(row[2], "Month to date", inr(spend["month_to_date"]), f"Since {day_label(spend['month_start'])}")
    kpi(row[3], "Forecast month-end", inr(budget["projected_month_end"]),
        f"Range {inr_compact(data['projected_range'][0])} - {inr_compact(data['projected_range'][1])}",
        help_text=f"Model: {data['forecast_model']}. See the Forecast & Budget page for the method.")
    kpi(row[4], "Monthly budget", inr(budget["monthly_budget"]), BUDGET_TEXT[budget["status"]],
        delta_color="inverse" if budget["status"] != "on_track" else "off")
    row = st.columns(5)
    kpi(row[0], "Budget used", pct(budget["utilization_pct"]), f"{pct(budget['expected_utilization_pct'])} of month elapsed")
    kpi(row[1], "Potential savings", f"{inr(data['potential_monthly_savings'])}/mo", "Open recommendations",
        help_text="Sum of estimated savings for recommendations that are not rejected, snoozed or already applied.")
    kpi(row[2], "Active anomalies", str(data["active_anomalies"]), "Open or acknowledged")
    kpi(row[3], "Running resources", f"{data['running_resources']}", f"of {data['total_resources']} in inventory")
    kpi(row[4], "Idle or underutilized", str(data["idle_or_underutilized"]), "Flagged by usage rules")

    used = min(budget["utilization_pct"], 100) / 100
    st.progress(used, text=f"Budget used {pct(budget['utilization_pct'])} · projected {pct(budget['projected_pct'])} "
                           f"of {inr(budget['monthly_budget'])} by month end")

    left, right = st.columns([3, 2])
    with left, st.container(border=True, key="panel-overview-chart"):
        section("Daily spend by provider", "Last 30 days of simulated charges; dashed line = forecast to month end")
        show_chart(spend_chart(data))
    with right, st.container(border=True, key="panel-overview-attention"):
        section("Needs attention today", "Computed from the current simulation state")
        attention_panel(data["attention"])

    columns = st.columns(3)
    with columns[0], st.container(border=True, key="panel-overview-provider"):
        section("Spend by provider", "Month to date")
        show_chart(bar_list(data["by_provider_month"], "provider", color_field="provider"))
    with columns[1], st.container(border=True, key="panel-overview-services"):
        section("Most expensive services", "Month to date")
        show_chart(bar_list(data["top_services_month"], "service", color=ACCENT))
    with columns[2], st.container(border=True, key="panel-overview-resources"):
        section("Highest-cost resources", "Month to date and current hourly rate")
        table = pd.DataFrame(data["top_resources_month"]).round({"cost": 0, "hourly_cost": 1})
        st.dataframe(table[["name", "provider", "cost", "hourly_cost"]], hide_index=True, width="stretch",
                     column_config={"name": "Resource", "provider": "Provider",
                                    "cost": money_column("Month to date (₹)"),
                                    "hourly_cost": money_column("Per hour (₹)")})
