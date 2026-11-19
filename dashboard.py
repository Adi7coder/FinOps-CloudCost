import streamlit as st

from ui import api
from ui.components import apply_style, flash_messages, run_action
from ui.format import inr, ist
from ui.pages import anomalies, benchmark, costs, forecast, optimization, overview, resources

st.set_page_config(page_title="FinOps Console", page_icon=":material/cloud:", layout="wide",
                   initial_sidebar_state="expanded")
apply_style()

PAGES = {
    "Overview": (overview, "dashboard"),
    "Cost Explorer": (costs, "query_stats"),
    "Resources": (resources, "dns"),
    "Anomalies": (anomalies, "crisis_alert"),
    "Forecast & Budget": (forecast, "trending_up"),
    "Optimization": (optimization, "savings"),
    "Model Evaluation": (benchmark, "science"),
}


def show_backend_offline(reason: str):
    st.title("FinOps Console")
    st.error(f"The backend API is not reachable at {api.API_URL}. The dashboard needs it for all data.")
    st.markdown("Start the backend in a separate terminal from the project folder, then refresh this page:")
    st.code(".venv\\Scripts\\Activate.ps1\npython -m uvicorn app.main:app --host 127.0.0.1 --port 8000", language="powershell")
    st.caption("If the backend runs on another address, set FINOPS_API_URL before starting Streamlit.")
    with st.expander("Technical detail"):
        st.code(reason)
    st.stop()


def simulation_controls(state: dict):
    st.markdown(f"**Scenario:** {state['scenario_title']}")
    st.caption(state["scenario_summary"])
    st.caption("Advance the simulation clock")
    columns = st.columns(3)
    for column, (label, hours) in zip(columns, [("+1 hour", 1), ("+1 day", 24), ("+7 days", 168)]):
        if column.button(label, key=f"advance-{hours}", width="stretch"):
            run_action("POST", "/api/v1/demo/advance", {"hours": hours},
                       lambda result: f"Clock moved to {ist(result['clock'])}. {inr(result['cost_added'])} of simulated "
                                      f"charges were added ({result['records_added']:,} usage records).")
    for event in state["active_events"]:
        st.caption(f"In progress until {ist(event['end'])}: {event['label']}")
    spending_event = state["spending_events"][0]
    if st.button("Trigger spending event", key="trigger-event", width="stretch", icon=":material/bolt:",
                 help=spending_event["label"]):
        run_action("POST", "/api/v1/demo/events", {"event": spending_event["key"]},
                   lambda result: f"{result['label']} on {result['resource_name']} until {ist(result['end'])}. "
                                  "Advance the clock by 1 day, then open the Anomalies page.")
    with st.popover("Load scenario", width="stretch", icon=":material/playlist_play:"):
        titles = {s["key"]: s["title"] for s in state["scenarios"]}
        summaries = {s["key"]: s["summary"] for s in state["scenarios"]}
        choice = st.selectbox("Scenario", list(titles), format_func=titles.get, key="scenario-choice",
                              index=list(titles).index(state["scenario"]))
        st.caption(summaries[choice])
        st.caption("Loading a scenario replaces the current simulation state, decisions and budget.")
        if st.button("Load this scenario", type="primary", key="load-scenario"):
            run_action("POST", "/api/v1/demo/scenario", {"scenario": choice},
                       lambda result: f"{result['message']}. Simulation clock: {ist(result['clock'])}.")
    with st.popover("Reset demo", width="stretch", icon=":material/restart_alt:"):
        st.write("Restore scenario A, the default budget and the original clock. All decisions and simulated actions "
                 "are cleared.")
        if st.button("Confirm reset", type="primary", key="confirm-reset"):
            run_action("POST", "/api/v1/demo/reset", None, lambda result: result["message"])


def sidebar(state: dict) -> str:
    with st.sidebar:
        st.markdown("### FinOps Console")
        st.markdown('<div class="fo-demo">DEMO MODE · simulated cloud data for a fictional company</div>',
                    unsafe_allow_html=True)
        st.markdown(f"**{state['company']}**  \nSimulation time: **{ist(state['clock'])}**")
        page = st.radio("Navigate", list(PAGES), key="nav", label_visibility="collapsed",
                        format_func=lambda name: f":material/{PAGES[name][1]}: {name}")
        st.divider()
        simulation_controls(state)
        st.divider()
        api_docs = "http://127.0.0.1:8000/docs" if api.API_URL == "inprocess" else f"{api.API_URL}/docs"
        st.caption(f"Backend connected · [API documentation]({api_docs})")
        st.caption("No AWS, Azure or GCP account is connected. Actions only change the simulation.")
    return page


try:
    current_state = api.state()
except api.BackendUnavailable as exc:
    show_backend_offline(str(exc))
except api.ApiError as exc:
    st.error(f"The backend returned an error: {exc.detail}")
    st.stop()

selected = sidebar(current_state)
flash_messages()
try:
    with st.spinner("Loading..."):
        PAGES[selected][0].render(current_state)
except api.BackendUnavailable as exc:
    show_backend_offline(str(exc))
except api.ApiError as exc:
    st.error(f"Could not load this page: {exc.detail}")
