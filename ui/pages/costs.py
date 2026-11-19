from datetime import date, timedelta

import pandas as pd
import streamlit as st

from ui import api
from ui.components import ACCENT, TEAL, bar_list, kpi, money_column, page_header, section, show_chart, stacked_daily_chart
from ui.format import day_label, inr, pct

PERIODS = ["Month to date", "Today", "Last 7 days", "Last 30 days", "Custom range"]


def period_dates(period: str, today: date, first: date):
    if period == "Today":
        return today, today
    if period == "Last 7 days":
        return today - timedelta(days=6), today
    if period == "Last 30 days":
        return today - timedelta(days=29), today
    if period == "Month to date":
        return today.replace(day=1), today
    picked = st.session_state.get("costs_custom")
    if isinstance(picked, (list, tuple)) and len(picked) == 2:
        return picked
    return max(first, today - timedelta(days=13)), today


def filters_bar(options: dict, today: date, first: date) -> dict:
    row = st.columns([1.3, 1, 1, 1, 1, 1, 1.4])
    period = row[0].selectbox("Period", PERIODS, key="costs_period")
    if period == "Custom range":
        st.date_input("Custom dates", value=(max(first, today - timedelta(days=13)), today), min_value=first,
                      max_value=today, key="costs_custom")
    start, end = period_dates(period, today, first)
    choose = {
        "provider": row[1].selectbox("Provider", ["All", *options["providers"]], key="costs_provider"),
        "service": row[2].selectbox("Service", ["All", *options["services"]], key="costs_service"),
        "team": row[3].selectbox("Team", ["All", *options["teams"]], key="costs_team"),
        "environment": row[4].selectbox("Environment", ["All", *options["environments"]], key="costs_environment"),
        "category": row[5].selectbox("Cost category", ["All", *options["categories"]], key="costs_category"),
    }
    names = {item["resource_id"]: item["name"] for item in options["resources"]}
    choose["resource_id"] = row[6].selectbox("Resource", ["All", *names], key="costs_resource",
                                             format_func=lambda rid: "All" if rid == "All" else f"{names[rid]} ({rid})")
    return {"start": start, "end": end, **choose}


def render(state: dict):
    page_header("Cost Explorer", "Filter simulated charges by period, provider, service, team, environment and resource")
    base = api.get("/api/v1/demo/costs", state["state_version"])
    options = base["options"]
    today = date.fromisoformat(options["last_date"])
    first = date.fromisoformat(options["first_date"])
    chosen = filters_bar(options, today, first)
    data = api.get("/api/v1/demo/costs", state["state_version"], **chosen)

    active = [f"{key.replace('_id', '')}: {value}" for key, value in chosen.items()
              if key not in ("start", "end") and value != "All"]
    st.caption(f"{day_label(data['start'])} to {day_label(data['end'])} ({data['days']} days)"
               + (" · includes today up to the current simulation hour" if data["includes_partial_today"] else "")
               + (" · " + ", ".join(active) if active else " · all resources"))

    row = st.columns(4)
    kpi(row[0], "Total cost", inr(data["total"]), f"{data['record_count']:,} usage line items")
    change = data["change_pct"]
    kpi(row[1], "Previous period", inr(data["previous_total"]),
        None if change is None else f"{change:+.1f}% vs previous {data['days']} days",
        delta_color="inverse" if change is not None else "off")
    kpi(row[2], "Daily average", inr(data["daily_average"]))
    services = data["by_service"]
    riser = max(services, key=lambda item: item["change"]) if services else None
    kpi(row[3], "Largest increase", riser["service"] if riser and riser["change"] > 0 else "None",
        f"{inr(riser['change'])} vs previous period" if riser and riser["change"] > 0 else "No service increased")

    if not data["trend"]:
        st.info("No charges match these filters.")
        return

    with st.container(border=True, key="panel-costs-trend"):
        section("Daily cost trend", "Stacked by provider")
        show_chart(stacked_daily_chart(data["trend"], height=280))

    tabs = st.tabs(["Provider", "Service", "Team", "Environment", "Cost category"])
    with tabs[0]:
        show_chart(bar_list(data["by_provider"], "provider", color_field="provider"))
    with tabs[1]:
        left, right = st.columns([3, 2])
        with left:
            show_chart(bar_list([{"service": s["service"], "cost": s["cost"]} for s in services], "service", color=ACCENT))
        with right:
            table = pd.DataFrame(services).round(0)
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={"service": "Service", "cost": money_column("This period (₹)"),
                                        "previous_cost": money_column("Previous period (₹)"),
                                        "change": money_column("Change (₹)")})
    with tabs[2]:
        show_chart(bar_list(data["by_team"], "team", color=TEAL))
    with tabs[3]:
        show_chart(bar_list(data["by_environment"], "environment", color=TEAL))
        environments = {item["environment"]: item["cost"] for item in data["by_environment"]}
        non_production = sum(v for k, v in environments.items() if k != "production")
        if data["total"]:
            st.caption(f"Development and staging: {inr(non_production)} ({pct(100 * non_production / data['total'])} "
                       f"of the filtered total)")
    with tabs[4]:
        show_chart(bar_list(data["by_category"], "category", color=ACCENT))

    with st.container(border=True, key="panel-costs-resources"):
        section("Top resources by cost")
        table = pd.DataFrame(data["top_resources"]).round(0)
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"resource_id": "Resource ID", "name": "Resource", "provider": "Provider",
                                    "service": "Service", "environment": "Environment", "team": "Team",
                                    "cost": money_column("Cost (₹)")})

    with st.container(border=True, key="panel-costs-records"):
        section("Usage records", "Daily line items per resource and charge type. Click a column header to sort.")
        records = pd.DataFrame(data["records"])
        st.dataframe(records, hide_index=True, width="stretch", height=360,
                     column_config={"date": "Date", "resource_id": "Resource ID", "name": "Resource",
                                    "provider": "Provider", "service": "Service", "category": "Category",
                                    "environment": "Environment", "team": "Team", "quantity": "Quantity",
                                    "unit": "Unit", "unit_price": money_column("Unit price (₹)"),
                                    "cost": money_column("Cost (₹)")})
        if data["record_count"] > len(records):
            st.caption(f"Showing the latest {len(records):,} of {data['record_count']:,} line items.")
        st.download_button("Download line items (CSV)", records.to_csv(index=False).encode("utf-8"),
                           file_name=f"finops_costs_{data['start']}_{data['end']}.csv", mime="text/csv")
