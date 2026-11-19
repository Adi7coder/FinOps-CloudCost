from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.demo import analytics, anomalies, catalog, engine, forecasting, rightsizing, store
from app.demo.schemas import (
    AdvanceRequest, AdvanceResponse, AnomalyStatusRequest, AnomalyStatusResponse, ApplyResponse, BudgetResponse,
    BudgetUpdate, DecisionRequest, DecisionResponse, EventRequest, EventResponse, PowerRequest, PowerResponse,
    ScenarioRequest, ScenarioResponse,
)
from app.demo.snapshot import get_snapshot

router = APIRouter(prefix="/api/v1/demo", tags=["Interactive startup demo (simulated)"])


def get_conn():
    engine.ensure_initialized()
    with store.session() as conn:
        yield conn


@router.get("/state")
def state(conn=Depends(get_conn)):
    return analytics.state_summary(conn)


@router.post("/scenario", response_model=ScenarioResponse)
def load_scenario(payload: ScenarioRequest, conn=Depends(get_conn)):
    engine.load_scenario(conn, payload.scenario.value)
    scenario = catalog.SCENARIOS[payload.scenario.value]
    return {"scenario": payload.scenario.value, "title": scenario["title"],
            "clock": store.get_setting(conn, "clock"), "message": f"Loaded {scenario['title']}"}


@router.post("/reset", response_model=ScenarioResponse)
def reset_demo(conn=Depends(get_conn)):
    engine.load_scenario(conn, "normal")
    return {"scenario": "normal", "title": catalog.SCENARIOS["normal"]["title"],
            "clock": store.get_setting(conn, "clock"), "message": "Demo reset to the baseline (scenario A, default budget)"}


@router.post("/advance", response_model=AdvanceResponse)
def advance(payload: AdvanceRequest, conn=Depends(get_conn)):
    return engine.advance(conn, payload.hours)


@router.post("/events", response_model=EventResponse, status_code=201)
def trigger_event(payload: EventRequest, conn=Depends(get_conn)):
    return engine.trigger_event(conn, payload.event.value)


@router.get("/overview")
def overview(conn=Depends(get_conn)):
    return analytics.overview(conn, get_snapshot(conn))


@router.get("/costs")
def costs(
    start: Optional[date] = None, end: Optional[date] = None, provider: Optional[str] = None,
    service: Optional[str] = None, team: Optional[str] = None, environment: Optional[str] = None,
    resource_id: Optional[str] = None, category: Optional[str] = None, conn=Depends(get_conn),
):
    snap = get_snapshot(conn)
    end = end or snap.today.date()
    start = start or end.replace(day=1)
    if start > end:
        raise HTTPException(status_code=422, detail="start must be on or before end")
    first, last = snap.history_start.date(), snap.today.date()
    if end < first or start > last:
        raise HTTPException(status_code=422, detail=f"Simulated data is available from {first} to {last}")
    filters = {"provider": provider, "service": service, "team": team, "environment": environment,
               "resource_id": resource_id, "category": category}
    return analytics.explore_costs(snap, max(start, first), min(end, last), filters)


@router.get("/resources")
def resources(conn=Depends(get_conn)):
    return analytics.inventory(conn, get_snapshot(conn))


@router.get("/resources/{resource_id}")
def resource_detail(resource_id: str, conn=Depends(get_conn)):
    return analytics.resource_detail(conn, get_snapshot(conn), resource_id)


@router.post("/resources/{resource_id}/power", response_model=PowerResponse)
def change_power(resource_id: str, payload: PowerRequest, conn=Depends(get_conn)):
    resource = engine.change_power_state(conn, resource_id, payload.action)
    message = ("Stopped in the simulation. Compute charges stop from the next hour; attached disk keeps billing."
               if payload.action == "stop" else "Started in the simulation. Compute charges resume from the next hour.")
    return {"resource_id": resource_id, "name": resource["name"], "status": resource["status"], "message": message}


@router.get("/budget", response_model=BudgetResponse)
def get_budget(conn=Depends(get_conn)):
    return forecasting.budget_view(get_snapshot(conn), store.get_setting(conn, "monthly_budget"))


@router.put("/budget", response_model=BudgetResponse)
def update_budget(payload: BudgetUpdate, conn=Depends(get_conn)):
    store.set_setting(conn, "monthly_budget", payload.monthly_budget)
    store.bump_version(conn, simulation_changed=False)
    return forecasting.budget_view(get_snapshot(conn), payload.monthly_budget)


@router.get("/forecast")
def forecast(conn=Depends(get_conn)):
    snap = get_snapshot(conn)
    return {**forecasting.build_forecast(snap),
            "budget": forecasting.budget_view(snap, store.get_setting(conn, "monthly_budget"))}


@router.get("/anomalies")
def list_anomalies(conn=Depends(get_conn)):
    return anomalies.list_anomalies(conn, get_snapshot(conn))


@router.get("/anomalies/{anomaly_id}")
def anomaly_detail(anomaly_id: str, conn=Depends(get_conn)):
    return anomalies.anomaly_detail(conn, get_snapshot(conn), anomaly_id)


@router.post("/anomalies/{anomaly_id}/status", response_model=AnomalyStatusResponse)
def update_anomaly(anomaly_id: str, payload: AnomalyStatusRequest, conn=Depends(get_conn)):
    return anomalies.update_status(conn, get_snapshot(conn), anomaly_id, payload.action, payload.note)


@router.get("/recommendations")
def recommendations(conn=Depends(get_conn)):
    return rightsizing.build_recommendations(conn, get_snapshot(conn))


@router.post("/recommendations/{resource_id}/decision", response_model=DecisionResponse)
def decide(resource_id: str, payload: DecisionRequest, conn=Depends(get_conn)):
    return rightsizing.record_decision(conn, get_snapshot(conn), resource_id, payload.decision, payload.note)


@router.post("/recommendations/{resource_id}/apply", response_model=ApplyResponse)
def apply(resource_id: str, conn=Depends(get_conn)):
    return rightsizing.apply_recommendation(conn, get_snapshot(conn), resource_id)
