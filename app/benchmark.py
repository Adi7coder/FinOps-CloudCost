from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import DATA_DIR

DISCLAIMER = (
    "Prototype benchmark results use controlled simulated multi-cloud billing/utilization data; "
    "savings are estimates, not realized production savings. No cloud resource is modified automatically."
)
REQUIRED_ARTIFACTS = [
    "experiment_summary.csv", "forecast_metrics.csv", "anomaly_metrics.csv", "anomaly_scored_dataset.csv",
    "rightsizing_metrics.csv", "rightsizing_recommendations.csv", "multi_cloud_finops_dataset.csv",
]

router = APIRouter(tags=["Review-2 benchmark (archived results)"])


@lru_cache(maxsize=None)
def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def _read(name: str) -> pd.DataFrame:
    if not (DATA_DIR / name).exists():
        raise HTTPException(status_code=500, detail=f"Missing prototype artifact: {name}")
    return _load(name).copy()


def _safe_records(df: pd.DataFrame):
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


def missing_artifacts() -> list:
    return [name for name in REQUIRED_ARTIFACTS if not (DATA_DIR / name).exists()]


class ReviewStatus(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DEFERRED = "DEFERRED"
    PENDING_REVIEW = "PENDING_REVIEW"


class RecommendationStatus(BaseModel):
    resource_id: str = Field(..., examples=["GC-GKE-INCentral"])
    status: ReviewStatus = ReviewStatus.PENDING_REVIEW
    reviewer_note: Optional[str] = Field(None, examples=["Reviewed during academic prototype demonstration"])


REVIEW_LOG: list = []


@router.get("/api/v1/summary")
def summary():
    s = _read("experiment_summary.csv")
    summary_map = dict(zip(s.iloc[:, 0], s.iloc[:, 1]))
    anomaly = _read("anomaly_metrics.csv").iloc[0]
    rights = _read("rightsizing_metrics.csv").iloc[0]
    return {
        "dataset": {
            "records": int(float(summary_map.get("records", 0))),
            "days": int(float(summary_map.get("days", 0))),
            "clouds": int(float(summary_map.get("clouds", 0))),
            "services": int(float(summary_map.get("services", 0))),
            "true_anomaly_rate_pct": float(summary_map.get("true_anomaly_rate_pct", 0)),
        },
        "forecast": {
            "best_model": summary_map.get("best_forecast_model"),
            "mape_pct": float(summary_map.get("best_mape_pct", 0)),
            "rmse_inr": float(summary_map.get("best_rmse_inr", 0)),
            "holdout_days": int(float(summary_map.get("forecast_holdout_days", 0))),
        },
        "anomaly_detection": {
            "model": str(anomaly["model"]),
            "precision": float(summary_map.get("if_precision", 0)),
            "recall": float(summary_map.get("if_recall", 0)),
            "f1": float(summary_map.get("if_f1", 0)),
            "true_anomalies": int(anomaly["true_anomalies"]),
            "flagged": int(anomaly["flagged"]),
        },
        "rightsizing": {
            "agreement": float(summary_map.get("rightsizing_accuracy", 0)),
            "estimated_monthly_savings_inr": float(summary_map.get("estimated_monthly_savings_inr", 0)),
            "portfolio_saving_pct": float(summary_map.get("portfolio_saving_pct", 0)),
            "actionable_resources": int(rights["actionable_resources"]),
            "total_resources": int(rights["total_resources"]),
        },
        "disclaimer": DISCLAIMER,
    }


@router.get("/api/v1/forecast/metrics")
def forecast_metrics():
    return _safe_records(_read("forecast_metrics.csv"))


@router.get("/api/v1/anomalies")
def anomalies(
    limit: int = Query(25, ge=1, le=200),
    service: Optional[str] = None,
    cloud: Optional[str] = None,
):
    df = _read("anomaly_scored_dataset.csv")
    df = df[df["iforest_pred"] == 1]
    if service:
        df = df[df["service"].str.lower() == service.lower()]
    if cloud:
        df = df[df["cloud"].str.lower() == cloud.lower()]
    cols = [c for c in ["date", "cloud", "service", "team", "region", "cost_inr", "cpu_avg_pct", "memory_avg_pct",
                        "usage_hours", "anomaly_score", "anomaly_label"] if c in df.columns]
    return {
        "model": "Isolation Forest",
        "count": int(len(df)),
        "items": _safe_records(df.sort_values("anomaly_score", ascending=False)[cols].head(limit)),
    }


@router.get("/api/v1/rightsizing")
def rightsizing(actionable_only: bool = True):
    df = _read("rightsizing_recommendations.csv")
    if actionable_only:
        df = df[df["recommendation"] != "RETAIN"]
    cols = [
        "resource_id", "cloud", "service", "team", "region", "avg_cpu", "avg_mem", "avg_hours",
        "monthly_cost", "recommendation", "estimated_saving", "saving_pct"
    ]
    cols = [c for c in cols if c in df.columns]
    return {
        "count": int(len(df)),
        "estimated_savings_inr": float(df["estimated_saving"].sum()) if "estimated_saving" in df.columns else 0.0,
        "auto_remediation": False,
        "items": _safe_records(df.sort_values("estimated_saving", ascending=False)[cols]),
    }


@router.get("/api/v1/cost-breakdown")
def cost_breakdown(group_by: str = Query("cloud", pattern="^(cloud|service|team|region)$")):
    df = _read("multi_cloud_finops_dataset.csv")
    out = df.groupby(group_by, as_index=False)["cost_inr"].sum().sort_values("cost_inr", ascending=False)
    return _safe_records(out)


@router.post("/api/v1/recommendations/review")
def review_recommendation(payload: RecommendationStatus):
    known = set(_read("rightsizing_recommendations.csv")["resource_id"])
    if payload.resource_id not in known:
        raise HTTPException(status_code=404, detail=f"Unknown resource_id '{payload.resource_id}'. Valid ids: {sorted(known)}")
    record = {
        "resource_id": payload.resource_id,
        "status": payload.status.value,
        "reviewer_note": payload.reviewer_note,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "auto_remediation_executed": False,
    }
    REVIEW_LOG.append(record)
    return {
        **record,
        "message": "Decision recorded for human-in-the-loop review; no cloud resource was modified.",
    }
