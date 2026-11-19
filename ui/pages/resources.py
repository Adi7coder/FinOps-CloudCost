import html

import altair as alt
import pandas as pd
import streamlit as st

from ui import api
from ui.components import ACCENT, kpi, money_axis, money_column, page_header, pill, run_action, section, show_chart
from ui.format import duration, inr, ist, pct

WORKLOAD_KINDS = {"worker", "job", "batch", "continuous"}


def inventory_table(items: list):
    rows = []
    for item in items:
        alert = item["recommendation"] if item["recommendation"] not in (None, "RETAIN") else ""
        if item["active_anomalies"]:
            alert = (alert + " · " if alert else "") + f"{item['active_anomalies']} anomaly"
        if item["idle_flag"] and not alert:
            alert = "Idle"
        rows.append({
            "Resource": item["name"], "ID": item["resource_id"], "Provider": item["provider"],
            "Service": item["service"], "Environment": item["environment"], "Status": item["status"],
            "Running for": duration(item["running_hours"]), "Per hour (₹)": round(item["hourly_cost"], 2),
            "Accumulated (₹)": round(item["accumulated_cost"]), "Avg CPU %": item["avg_cpu_7d"],
            "Avg memory %": item["avg_mem_7d"], "Last activity": ist(item["last_activity"]) if item["last_activity"] else "-",
            "Workload": item["workload_status"], "Alert": alert or "-",
        })
    table = pd.DataFrame(rows)
    table[["Avg CPU %", "Avg memory %"]] = table[["Avg CPU %", "Avg memory %"]].apply(pd.to_numeric)
    st.dataframe(table, hide_index=True, width="stretch", height=min(38 * (len(rows) + 1), 620),
                 column_config={"Per hour (₹)": money_column("Per hour (₹)"),
                                "Accumulated (₹)": money_column("Accumulated (₹)"),
                                "Avg CPU %": st.column_config.NumberColumn(format="%.1f"),
                                "Avg memory %": st.column_config.NumberColumn(format="%.1f")})


def workloads_table(items: list):
    rows = []
    for item in items:
        if item["kind"] not in WORKLOAD_KINDS:
            continue
        uptime, active = item["uptime_hours_7d"], item["active_hours_7d"]
        rows.append({"Workload": item["name"], "Type": {"worker": "Queue worker", "job": "Scheduled job",
                                                        "batch": "Scheduled training", "continuous": "Long-running job"}[item["kind"]],
                     "Status": item["workload_status"], "Machine uptime (7 d)": f"{uptime:.0f} h",
                     "Useful work (7 d)": f"{active:.0f} h", "Idle share": pct(100 * (1 - active / uptime)) if uptime else "-",
                     "Cost while idle (7 d, ₹)": round(item["idle_running_cost_7d"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Cost while idle (7 d, ₹)": money_column("Cost while idle (7 d, ₹)")})


def usage_chart(hourly: list):
    frame = pd.DataFrame(hourly)
    if frame.empty:
        st.caption("No compute hours in the last 72 hours (resource stopped or outside its schedule).")
        return
    frame["active_pct"] = frame["active"] * 100
    base = alt.Chart(frame).encode(x=alt.X("ts:T", title=None, axis=alt.Axis(format="%d %b %H:%M", labelAngle=0, tickCount=6)))
    active = base.mark_bar(color="#dbe7f5", opacity=0.9).encode(
        y=alt.Y("active_pct:Q", title="CPU % / active hours", scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("ts:T", format="%d %b %H:%M", title="Hour"), alt.Tooltip("active:Q", title="Doing work (1 = yes)")])
    cpu = base.mark_line(color=ACCENT, strokeWidth=2).encode(
        y="cpu:Q", tooltip=[alt.Tooltip("ts:T", format="%d %b %H:%M", title="Hour"), alt.Tooltip("cpu:Q", format=".1f", title="CPU %"),
                            alt.Tooltip("mem:Q", format=".1f", title="Memory %")])
    show_chart(alt.layer(active, cpu).properties(height=220))
    st.caption("Line = CPU %. Shaded bars = hours in which the workload was doing useful work.")


def daily_chart(daily: list):
    frame = pd.DataFrame(daily)
    chart = alt.Chart(frame).mark_bar(color=ACCENT, size=14, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0)), y=alt.Y("cost:Q", axis=money_axis()),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y", title="Date"), alt.Tooltip("cost:Q", format=",.0f", title="Cost (₹)"),
                 alt.Tooltip("running_hours:Q", title="Running hours"), alt.Tooltip("active_hours:Q", title="Active hours")])
    show_chart(chart.properties(height=220))


def power_controls(detail: dict):
    if detail["can_stop"]:
        with st.popover("Stop resource (simulation)", icon=":material/stop_circle:"):
            st.write(f"Stop **{detail['name']}** in the simulation? Compute charges stop from the next simulated hour; "
                     "attached disk storage keeps billing. No real cloud resource is touched.")
            if st.button("Confirm stop", key=f"stop-{detail['resource_id']}", type="primary"):
                run_action("POST", f"/api/v1/demo/resources/{detail['resource_id']}/power", {"action": "stop"},
                           lambda result: f"{result['name']} stopped. {result['message']}")
    elif detail["can_start"]:
        if st.button("Start resource (simulation)", key=f"start-{detail['resource_id']}", icon=":material/play_circle:"):
            run_action("POST", f"/api/v1/demo/resources/{detail['resource_id']}/power", {"action": "start"},
                       lambda result: f"{result['name']} started. {result['message']}")
    else:
        st.caption("Storage and data transfer are billed by volume and cannot be stopped.")


def resource_detail(state: dict, resource_id: str):
    detail = api.get(f"/api/v1/demo/resources/{resource_id}", state["state_version"])
    st.markdown(f"#### {detail['name']} {pill(detail['status'])}{pill('Flagged: idle', 'warning') if detail['idle_flag'] else ''}",
                unsafe_allow_html=True)
    st.caption(f"{detail['resource_id']} · {detail['provider']} {detail['service']} · {detail['region']} · "
               f"{detail['environment']} · {detail['team']} team")
    row = st.columns(5)
    kpi(row[0], "Running for", duration(detail["running_hours"]),
        f"Since {ist(detail['started_at'])}" if detail["started_at"] else "Not running now")
    kpi(row[1], "Cost per hour now", inr(detail["hourly_cost"], 2), detail["resource_class"])
    kpi(row[2], "Monthly estimate", inr(detail["monthly_estimate"]), "At current configuration")
    kpi(row[3], "Accumulated cost", inr(detail["accumulated_cost"]), f"Since {ist(detail['launched_at'])}")
    kpi(row[4], "Month to date", inr(detail["month_to_date_cost"]))

    left, right = st.columns([1.1, 1])
    with left:
        uptime, active = detail["uptime_hours_7d"], detail["active_hours_7d"]
        facts = [
            f"<b>Used for:</b> {html.escape(detail['purpose'])}",
            f"<b>Workload status:</b> {html.escape(detail['workload_status'])}",
            f"<b>Last useful activity:</b> {ist(detail['last_activity']) if detail['last_activity'] else 'none recorded'}",
            f"<b>Last 7 days:</b> machine up {uptime:.0f} h, doing useful work {active:.0f} h"
            + (f", cost while idle {inr(detail['idle_running_cost_7d'])}" if detail["idle_running_cost_7d"] else ""),
        ]
        if detail["avg_cpu_7d"] is not None:
            facts.append(f"<b>Average utilization (7 days):</b> CPU {pct(detail['avg_cpu_7d'])}, memory {pct(detail['avg_mem_7d'])}")
        st.markdown('<div class="fo-kv">' + "<br>".join(facts) + "</div>", unsafe_allow_html=True)
        recommendation = detail["recommendation_detail"]
        if recommendation:
            action = recommendation["action"]
            st.markdown(f"**Recommendation:** {pill(action)} {pill(recommendation['decision_status'])}",
                        unsafe_allow_html=True)
            st.write(recommendation["reason"])
            if action != "RETAIN":
                st.caption(f"Estimated saving {inr(recommendation['estimated_saving'])}/month. {recommendation['risk']}")
        for anomaly in detail["anomalies"]:
            st.warning(f"Anomaly on {anomaly['date']}: {inr(anomaly['observed_cost'])} vs typical "
                       f"{inr(anomaly['baseline_cost'])} ({anomaly['status']}).")
        power_controls(detail)
    with right:
        section("Last 72 hours", "Hourly CPU and activity")
        usage_chart(detail["hourly"])
        section("Daily cost", "Last 21 days, all charges")
        daily_chart(detail["daily"])
        with st.expander("How this resource is billed"):
            charges = pd.DataFrame(detail["charges"]).round({"quantity": 2, "unit_price": 2, "cost": 0})
            st.dataframe(charges, hide_index=True, width="stretch",
                         column_config={"charge": "Charge", "unit": "Unit", "unit_price": money_column("Unit price (₹)"),
                                        "quantity": "Quantity (all time)", "cost": money_column("Cost (₹)")})


def render(state: dict):
    page_header("Resources", "Every simulated resource with its running time, cost, utilization and workload state")
    data = api.get("/api/v1/demo/resources", state["state_version"])
    items = data["items"]

    row = st.columns(4)
    running = [i for i in items if i["started_at"] and i["can_stop"]]
    kpi(row[0], "Resources", str(len(items)), f"{len(running)} running right now")
    kpi(row[1], "Current hourly cost", inr(data["totals"]["hourly_cost"], 0), "All resources, next simulated hour")
    kpi(row[2], "Accumulated cost", inr(data["totals"]["accumulated_cost"]), "Since the simulated history began")
    flagged = [i for i in items if i["idle_flag"] or i["recommendation"] in ("DOWNSIZE", "STOP_REVIEW")]
    kpi(row[3], "Idle or underutilized", str(len(flagged)), ", ".join(i["name"] for i in flagged[:2]) or "None")

    filters = st.columns([1, 1, 1, 1.2])
    provider = filters[0].selectbox("Provider", ["All", "AWS", "Azure", "GCP"], key="inv_provider")
    environment = filters[1].selectbox("Environment", ["All", "production", "staging", "development"], key="inv_env")
    status = filters[2].selectbox("Status", ["All", "running", "scheduled", "stopped"], key="inv_status")
    only_flagged = filters[3].toggle("Only idle or flagged", key="inv_flagged")
    shown = [i for i in items if (provider == "All" or i["provider"] == provider)
             and (environment == "All" or i["environment"] == environment)
             and (status == "All" or i["status"] == status)
             and (not only_flagged or i["idle_flag"] or i["recommendation"] in ("DOWNSIZE", "STOP_REVIEW")
                  or i["active_anomalies"])]

    with st.container(border=True, key="panel-inventory"):
        section("Inventory", f"{len(shown)} of {len(items)} resources")
        if shown:
            inventory_table(shown)
        else:
            st.info("No resources match these filters.")

    with st.container(border=True, key="panel-workloads"):
        section("Background workers and scheduled jobs",
                "Machine uptime compared with the hours the job or worker actually did useful work")
        workloads_table(items)

    with st.container(border=True, key="panel-resource-detail"):
        section("Resource details")
        names = {i["resource_id"]: i["name"] for i in sorted(items, key=lambda i: (not i["idle_flag"], i["name"]))}
        picked = st.selectbox("Resource", list(names), key="inv_pick",
                              format_func=lambda rid: f"{names[rid]} ({rid})")
        resource_detail(state, picked)
