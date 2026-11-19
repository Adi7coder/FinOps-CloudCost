from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.benchmark import DISCLAIMER, missing_artifacts
from app.benchmark import router as benchmark_router
from app.config import API_VERSION
from app.demo import store
from app.demo.engine import SimulationError, ensure_initialized
from app.demo.routes import router as demo_router
from app.demo.schemas import HealthResponse


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_initialized()
    yield


app = FastAPI(
    title="AI-Powered FinOps Platform for Cloud Cost Intelligence",
    version=API_VERSION,
    lifespan=lifespan,
    description=(
        "Backend for the FinOps capstone project.\n\n"
        "* **Review-2 benchmark** endpoints return the archived experiment results (simulated benchmark data).\n"
        "* **Interactive startup demo** endpoints (`/api/v1/demo/...`) run a simulated cloud environment for a "
        "fictional company. No real AWS, Azure or GCP account is connected and no real resource is changed.\n\n"
        f"**{DISCLAIMER}**"
    ),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(benchmark_router)
app.include_router(demo_router)


@app.exception_handler(SimulationError)
async def simulation_error_handler(_request: Request, exc: SimulationError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        with store.session() as conn:
            database = "ready" if store.is_initialized(conn) else "not initialized"
    except Exception as exc:
        database = f"error: {exc}"
    return {"status": "ok", "phase": "Capstone final review", "version": API_VERSION,
            "demo_database": database, "missing_benchmark_artifacts": missing_artifacts()}
