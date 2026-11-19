import altair as alt
import pandas as pd
import streamlit as st

from ui import api
from ui.components import CRITICAL, MUTED, kpi, money_axis, money_column, page_header, pill, run_action, section, show_chart
from ui.format import day_label, inr, ist, pct

ACTIONS = [("acknowledge", "Acknowledge", {"open"}), ("investigate", "Mark as investigated", {"open", "acknowledged"}),
           ("dismiss", "Dismiss", {"open", "acknowledged", "investigated"}),
           ("reopen", "Reopen", {"acknowledged", "investigated", "dismissed"})]


def history_chart(item: dict):
    frame = pd.DataFrame(item["history"])
    frame["flagged"] = (frame["date"] >= item["date"]) & (frame["date"] <= item["last_date"])
    base = alt.Chart(frame).encode(x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0)))
    bars = base.mark_bar(size=10, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        y=alt.Y("cost:Q", axis=money_axis()),
        color=alt.condition("datum.flagged", alt.value(CRITICAL), alt.value("#9fb8d9")),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y", title="Date"), alt.Tooltip("cost:Q", format=",.0f", title="Cost (₹)"),
                 alt.Tooltip("cpu:Q", format=".1f", title="CPU %")])
    baseline = alt.Chart(pd.DataFrame({"baseline": [item["baseline_cost"]]})).mark_rule(
        color=MUTED, strokeDash=[5, 4]).encode(y="baseline:Q")
    show_chart(alt.layer(bars, baseline).properties(height=240))
    st.caption("Red = flagged day(s). Dashed line = the resource's median daily cost (baseline).")


def evidence_table(item: dict):
    rows = [
        ("Daily cost", inr(item["observed_cost"]), inr(item["baseline_cost"])),
        ("Billed instance-hours", f"{item['instance_hours']:.0f}", f"{item['typical_instance_hours']:.0f}"),
        ("Average CPU", pct(item["cpu"]), pct(item["typical_cpu"])),
        ("Average memory", pct(item["mem"]), pct(item["typical_mem"])),
        ("Active hours", f"{item['active_hours']:.0f}", f"{item['typical_active_hours']:.0f}"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Measure", "Flagged day", "Typical day"]), hide_index=True, width="stretch")


def detail_panel(state: dict, anomaly_id: str):
    item = api.get(f"/api/v1/demo/anomalies/{anomaly_id}", state["state_version"])
    st.markdown(f"#### {item['name']} {pill(item['severity'])} {pill(item['status'])}", unsafe_allow_html=True)
    span = day_label(item["date"]) if item["days_flagged"] == 1 else \
        f"{day_label(item['date'])} to {day_label(item['last_date'])} ({item['days_flagged']} days)"
    st.caption(f"{item['provider']} {item['service']} · {item['environment']} · {item['team']} team · {span} · "
               f"detected {ist(item['detected_at'])}")
    row = st.columns(5)
    kpi(row[0], "Observed cost", inr(item["observed_cost"]), "First flagged day")
    kpi(row[1], "Expected (baseline)", inr(item["baseline_cost"]), "Median daily cost")
    kpi(row[2], "Excess cost", inr(item["total_excess_cost"]), f"{item['cost_ratio']:.2f}x the baseline")
    kpi(row[3], "Anomaly score", f"{item['anomaly_score']:.3f}", "Higher = more unusual")
    kpi(row[4], "Cost z-score", f"{item['cost_z']:.1f}", "Above the median")

    left, right = st.columns([1.2, 1])
    with left:
        history_chart(item)
    with right:
        section("Evidence from recorded metrics")
        evidence_table(item)
        section("Possible explanations", "Based on the metrics above - not a confirmed root cause")
        for text in item["possible_explanations"] or ["No single metric explains the change."]:
            st.markdown(f"- {text}")

    section("Investigation")
    if item.get("note"):
        st.caption(f"Last note: {item['note']}")
    note = st.text_input("Note (optional)", key=f"note-{anomaly_id}", placeholder="e.g. Checked autoscaling group history")
    buttons = st.columns(len(ACTIONS))
    for column, (action, label, allowed) in zip(buttons, ACTIONS):
        if column.button(label, key=f"{action}-{anomaly_id}", disabled=item["status"] not in allowed, width="stretch"):
            run_action("POST", f"/api/v1/demo/anomalies/{anomaly_id}/status", {"action": action, "note": note or None},
                       lambda result: f"Anomaly status changed from {result['previous_status']} to {result['status']}.")


def render(state: dict):
    page_header("Anomaly Center", "Unusual daily spend detected by the Isolation Forest model on the simulated usage")
    data = api.get("/api/v1/demo/anomalies", state["state_version"])
    counts, model = data["counts"], data["model"]
    row = st.columns(5)
    kpi(row[0], "Active", str(data["active"]), "Open or acknowledged")
    kpi(row[1], "Open", str(counts["open"]))
    kpi(row[2], "Acknowledged", str(counts["acknowledged"]))
    kpi(row[3], "Investigated", str(counts["investigated"]))
    kpi(row[4], "Dismissed", str(counts["dismissed"]))
    st.caption(f"Model: Isolation Forest ({model['n_estimators']} trees, contamination {model['contamination']:.4f}, "
               f"random_state {model['random_state']}) on {model['records_scored']:,} daily resource records with "
               f"features {', '.join(model['features'])}. The model flagged {model['flagged_by_model']} records; "
               f"{model['report_rule']}")

    items = data["items"]
    if not items:
        st.info("No cost anomalies in the last 30 days. To see one, use 'Trigger spending event' in the sidebar and "
                "advance the simulation by 1 day, or load scenario C.")
        return
    with st.container(border=True, key="panel-anomaly-list"):
        section("Detected anomalies", "Consecutive flagged days for the same resource are grouped into one incident")
        table = pd.DataFrame([{
            "Date": item["date"], "Days": item["days_flagged"], "Resource": item["name"], "Provider": item["provider"],
            "Service": item["service"], "Observed (₹)": round(item["observed_cost"]),
            "Baseline (₹)": round(item["baseline_cost"]), "Ratio": round(item["cost_ratio"], 2),
            "Score": round(item["anomaly_score"], 3), "Severity": item["severity"], "Status": item["status"],
            "CPU %": round(item["cpu"], 1), "Detected": ist(item["detected_at"]),
        } for item in items])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"Observed (₹)": money_column("Observed (₹)"), "Baseline (₹)": money_column("Baseline (₹)")})

    with st.container(border=True, key="panel-anomaly-detail"):
        labels = {item["anomaly_id"]: f"{item['date']} · {item['name']} · {item['severity']} · {item['status']}"
                  for item in items}
        picked = st.selectbox("Inspect anomaly", list(labels), format_func=labels.get, key="anomaly_pick")
        detail_panel(state, picked)
