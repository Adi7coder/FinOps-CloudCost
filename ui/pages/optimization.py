import pandas as pd
import streamlit as st

from ui import api
from ui.components import kpi, money_column, page_header, pill, run_action, section
from ui.format import duration, inr, ist, pct, title_case

DECISIONS = [("review", "Start review", {"open"}), ("approve", "Approve", {"open", "in_review"}),
             ("reject", "Reject", {"open", "in_review", "approved"}), ("snooze", "Snooze 7 days", {"open", "in_review"}),
             ("reopen", "Reopen", {"in_review", "approved", "rejected", "snoozed"})]
ACTION_TEXT = {"STOP_REVIEW": "Review for shutdown", "DOWNSIZE": "Downsize", "RETAIN": "Keep as is"}


def recommendation_table(items: list):
    rows = [{
        "Resource": item["name"], "Suggested action": ACTION_TEXT[item["action"]], "Decision": title_case(item["decision_status"]),
        "Provider": item["provider"], "Configuration": item["resource_class"], "CPU %": round(item["avg_cpu"], 1),
        "Memory %": round(item["avg_mem"], 1), "Active h/day": round(item["avg_hours"], 1),
        "Running for": duration(item["running_hours"]), "Monthly cost (₹)": round(item["monthly_cost"]),
        "Est. saving (₹/mo)": round(item["estimated_saving"]),
    } for item in items]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Monthly cost (₹)": money_column("Monthly cost (₹)"),
                                "Est. saving (₹/mo)": money_column("Est. saving (₹/mo)")})


def rule_table(checks: list):
    frame = pd.DataFrame([{"Rule": c["rule"], "Condition": c["condition"], "Measured": c["value"],
                           "Met": "Yes" if c["passed"] else "No"} for c in checks])
    st.dataframe(frame, hide_index=True, width="stretch")


def detail_panel(item: dict):
    resource_id = item["resource_id"]
    st.markdown(f"#### {item['name']} {pill(item['action'])} {pill(item['decision_status'])}", unsafe_allow_html=True)
    st.caption(f"{resource_id} · {item['provider']} {item['service']} · {item['environment']} · {item['team']} team")
    row = st.columns(5)
    kpi(row[0], "Current configuration", item["resource_class"], f"{item['instances']} instance(s)")
    kpi(row[1], "Utilization", f"{pct(item['avg_cpu'])} CPU", f"{pct(item['avg_mem'])} mem · {item['avg_hours']:.1f} h/day")
    kpi(row[2], "Running for", duration(item["running_hours"]), f"Evaluated over {item['window_days']:.1f} days")
    kpi(row[3], "Current monthly cost", inr(item["monthly_cost"]), "At current configuration")
    kpi(row[4], "Estimated saving", f"{inr(item['estimated_saving'])}/mo",
        f"Policy estimate ({item['savings_fraction'] * 100:.0f}%)")

    left, right = st.columns([1.2, 1])
    with left:
        section("Why this is recommended")
        st.write(item["reason"])
        if item["risk"]:
            st.warning(item["risk"], icon=":material/info:")
        if item["decision_note"]:
            st.caption(f"Decision note: {item['decision_note']} ({ist(item['decision_updated_at'])})")
        if item["snoozed_until"]:
            st.caption(f"Snoozed until {ist(item['snoozed_until'])}")
    with right:
        section("Rule checks", "Original Review-2 thresholds")
        rule_table(item["checks"])

    if item["action"] == "RETAIN":
        st.caption("No decision needed: the resource does not meet the downsize or stop-review thresholds.")
        return
    section("Decision")
    note = st.text_input("Note (optional)", key=f"decision-note-{resource_id}",
                         placeholder="e.g. Confirmed with the Backend team")
    buttons = st.columns(len(DECISIONS) + 1)
    for column, (decision, label, allowed) in zip(buttons, DECISIONS):
        if column.button(label, key=f"{decision}-{resource_id}", disabled=item["decision_status"] not in allowed,
                         width="stretch", type="primary" if decision == "approve" else "secondary"):
            run_action("POST", f"/api/v1/demo/recommendations/{resource_id}/decision",
                       {"decision": decision, "note": note or None},
                       lambda result: f"Decision recorded: {result['previous_status']} -> {result['status']}. "
                                      "No infrastructure was changed.")
    with buttons[-1]:
        with st.popover("Apply in simulation", disabled=item["decision_status"] != "approved", width="stretch"):
            what = "resize it to a smaller size" if item["action"] == "DOWNSIZE" else (
                "restore its start/stop schedule" if item["kind"] == "job" else "stop it")
            st.write(f"This will {what} **in the simulation only**. Future simulated hours will be billed at the new "
                     "configuration so you can compare proposed and realized savings. No real cloud resource is touched.")
            if st.button("Confirm and apply", key=f"apply-{resource_id}", type="primary"):
                run_action("POST", f"/api/v1/demo/recommendations/{resource_id}/apply", None,
                           lambda result: result["message"])
    st.caption("Approving only records the decision. 'Apply in simulation' becomes available after approval.")


def render(state: dict):
    page_header("Optimization", "Rightsizing recommendations from the original utilization rules, with a review workflow")
    data = api.get("/api/v1/demo/recommendations", state["state_version"])
    summary = data["summary"]
    row = st.columns(5)
    kpi(row[0], "Potential savings", f"{inr(summary['potential_monthly_savings'])}/mo", "Open, in review or approved")
    kpi(row[1], "Approved, not applied", f"{inr(summary['approved_awaiting_apply'])}/mo", "Waiting for action")
    kpi(row[2], "Realized (run rate)", f"{inr(summary['realized_monthly_run_rate'])}/mo", "Measured")
    kpi(row[3], "Realized so far", inr(summary["realized_to_date"]), "Since applied")
    kpi(row[4], "Recommendations", f"{summary['actionable']} actionable", f"of {summary['evaluated']} resources evaluated")
    st.caption(f"Policy: STOP_REVIEW if {data['policy']['stop_review']}; DOWNSIZE if {data['policy']['downsize']}; "
               f"otherwise RETAIN. Window: {data['policy']['window']}. Savings estimate = monthly cost x 45% (downsize) "
               "or x 85% (stop review). Potential savings are estimates; realized savings are measured in the simulation.")

    items = data["items"]
    show_all = st.toggle("Show resources that should be kept (RETAIN)", key="opt_show_all")
    visible = items if show_all else [i for i in items if i["action"] != "RETAIN"]
    with st.container(border=True, key="panel-opt-list"):
        section("Recommendations", "Highest savings first")
        if visible:
            recommendation_table(visible)
        else:
            st.success("No resource currently meets the downsize or stop-review thresholds.")

    if visible:
        with st.container(border=True, key="panel-opt-detail"):
            labels = {i["resource_id"]: f"{i['name']} · {ACTION_TEXT[i['action']]} · {inr(i['estimated_saving'])}/mo · "
                                         f"{i['decision_status'].replace('_', ' ')}" for i in visible}
            picked = st.selectbox("Review recommendation", list(labels), format_func=labels.get, key="opt_pick")
            detail_panel(next(i for i in visible if i["resource_id"] == picked))

    with st.container(border=True, key="panel-opt-applied"):
        section("Applied in the simulation", "Proposed (policy estimate) compared with savings measured afterwards")
        applied = data["applied_actions"]
        if not applied:
            st.caption("Nothing has been applied yet. Approve a recommendation, apply it, then advance the clock.")
        else:
            frame = pd.DataFrame([{
                "Resource": a["name"], "Action": a["action"].replace("_", " "), "Applied at": ist(a["applied_at"]),
                "Hours since": round(a["hours_since"]), "Cost/hour before (₹)": round(a["before_hourly"], 2),
                "Proposed (₹/mo)": None if a["proposed_monthly"] is None else round(a["proposed_monthly"]),
                "Realized so far (₹)": round(a["realized_to_date"]),
                "Realized run rate (₹/mo)": round(a["monthly_run_rate_saving"]), "Note": a["note"],
            } for a in applied])
            st.dataframe(frame, hide_index=True, width="stretch",
                         column_config={"Proposed (₹/mo)": money_column("Proposed (₹/mo)"),
                                        "Realized so far (₹)": money_column("Realized so far (₹)"),
                                        "Realized run rate (₹/mo)": money_column("Realized run rate (₹/mo)")})
    if data["not_evaluated"]:
        st.caption("Not evaluated: " + "; ".join(f"{n['name']} ({n['note']})" for n in data["not_evaluated"]))
