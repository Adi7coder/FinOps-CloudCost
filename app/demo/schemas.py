from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app.demo.catalog import SCENARIOS, SPENDING_EVENTS

ScenarioKey = Enum("ScenarioKey", {key: key for key in SCENARIOS}, type=str)
SpendingEventKey = Enum("SpendingEventKey", {key: key for key in SPENDING_EVENTS}, type=str)


class AdvanceRequest(BaseModel):
    hours: int = Field(..., ge=1, le=720, examples=[24])


class AdvanceResponse(BaseModel):
    previous_clock: str
    clock: str
    hours: int
    cost_added: float
    records_added: int


class ScenarioRequest(BaseModel):
    scenario: ScenarioKey


class ScenarioResponse(BaseModel):
    scenario: str
    title: str
    clock: str
    message: str


class EventRequest(BaseModel):
    event: SpendingEventKey


class EventResponse(BaseModel):
    resource_id: str
    resource_name: str
    start: str
    end: str
    label: str


class BudgetUpdate(BaseModel):
    monthly_budget: float = Field(..., gt=0, le=1_000_000_000, examples=[400000])


class BudgetResponse(BaseModel):
    monthly_budget: float
    month_to_date: float
    utilization_pct: float
    projected_month_end: float
    projected_pct: float
    variance: float
    status: Literal["on_track", "at_risk", "over_budget"]
    expected_utilization_pct: float


class PowerRequest(BaseModel):
    action: Literal["stop", "start"]


class PowerResponse(BaseModel):
    resource_id: str
    name: str
    status: str
    message: str
    simulated_only: bool = True


class AnomalyStatusRequest(BaseModel):
    action: Literal["acknowledge", "investigate", "dismiss", "reopen"]
    note: Optional[str] = Field(None, max_length=500)


class AnomalyStatusResponse(BaseModel):
    anomaly_id: str
    previous_status: str
    status: str
    note: Optional[str]
    updated_at: str


class DecisionRequest(BaseModel):
    decision: Literal["review", "approve", "reject", "snooze", "reopen"]
    note: Optional[str] = Field(None, max_length=500)


class DecisionResponse(BaseModel):
    resource_id: str
    action: str
    previous_status: str
    status: str
    note: Optional[str]
    updated_at: str
    snoozed_until: Optional[str]
    infrastructure_changed: bool


class ApplyResponse(BaseModel):
    resource_id: str
    action: str
    status: str
    message: str
    estimated_monthly_saving: float
    simulated_only: bool


class HealthResponse(BaseModel):
    status: str
    phase: str
    version: str
    demo_database: str
    missing_benchmark_artifacts: List[str]
