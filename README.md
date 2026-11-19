# AI-Powered FinOps Platform for Cloud Cost Intelligence

A capstone project that helps a small company understand and control its cloud spending across AWS,
Microsoft Azure and Google Cloud. It has a FastAPI backend and a Streamlit dashboard, and combines cost
reporting, forecasting, Isolation Forest anomaly detection and rule-based rightsizing recommendations.

> **Demo mode.** The application does not connect to any real AWS, Azure or GCP account. All interactive data
> comes from a simulated environment for a fictional company, and no action in the app changes a real resource.

## 1. Problem

Cloud bills usually arrive at the end of the month. By then it is too late to notice that a test server was left
running, that a background job scaled out by mistake, or that spending is heading over budget. A small team needs
to answer questions like:

- How much have we spent today, this week and this month, and what will the month end at?
- Which provider, service, team or resource is driving the cost?
- Which machines are running but barely used, and how long have they been running?
- Did spending jump unexpectedly, and what changed?
- What could we downsize or stop, how much would it save, and why is it recommended?

## 2. What we developed

The project has two parts that are kept separate on purpose.

1. **Review-2 benchmark (archived experiments).** A 270-day simulated multi-cloud dataset (3,240 records,
   3 providers, 12 services) used to compare forecasting models, evaluate an Isolation Forest detector and test
   the rightsizing rules. The results are stored as CSV files in `data/` and figures in `figures/`.
2. **Interactive startup demo (live simulation).** A fictional startup, *Tasklane Labs Pvt. Ltd.*, runs
   20-22 resources on the three clouds: web and API servers, databases, Kubernetes node pools, staging and
   development machines, queue workers, scheduled jobs, ML training machines, object storage and data transfer.
   The backend simulates their hourly usage and cost. Forecasting, anomaly detection and rightsizing are run on
   this simulated data, so the demo shows its own computed values rather than the benchmark numbers.

## 3. Main features

| Page | What it does |
|---|---|
| Overview | Spend today / this week / month to date, forecast month-end, budget use, potential savings, active anomalies, running and idle resources, daily spend chart, and a "needs attention" list computed from the current state |
| Cost Explorer | Filters by period (today, 7 days, 30 days, month to date, custom), provider, service, team, environment, cost category and resource; trend, breakdowns, service-level change against the previous period, top resources and downloadable line items |
| Resources | Inventory with status, running time, hourly cost, accumulated cost, utilization and last activity; background workers and scheduled jobs (machine uptime compared with useful work); per-resource detail; stop/start in the simulation |
| Anomalies | Isolation Forest detections with observed vs baseline cost, severity, evidence and possible explanations; acknowledge / investigate / dismiss / reopen |
| Forecast & Budget | Month-end projection with an approximate range, cumulative spend against the budget, budget editing, live backtest of Seasonal Naive and Additive Seasonal, and the archived benchmark comparison |
| Optimization | RETAIN / DOWNSIZE / STOP_REVIEW recommendations with reasons and risks; review, approve, reject, snooze; apply an approved action in the simulation and compare proposed with measured savings |
| Model Evaluation | The archived Review-2 results and figures, with where each number comes from |

The sidebar has the simulation controls: advance the clock by 1 hour, 1 day or 7 days, trigger a spending event,
load one of five scenarios, and reset the demo. Every control calls the backend API.

## 4. Technology stack

- Python 3.10, FastAPI, Pydantic, Uvicorn
- pandas, NumPy, scikit-learn (Isolation Forest, Ridge regression)
- Streamlit with Altair charts
- SQLite (standard library `sqlite3`) for the demo state
- pytest for automated tests

No Docker, external database, paid API or cloud credentials are needed.

## 5. Project structure

```
app/
  main.py               FastAPI application (health, docs, routers, error handling)
  benchmark.py          the seven original Review-2 endpoints (archived results)
  init_demo.py          creates or resets the demo database
  ml.py                 imports the model code from scripts/ so it is not duplicated
  demo/
    catalog.py          fictional company, resources, prices, scenarios
    engine.py           hourly usage and cost simulation, clock, stop/start, resize, events
    store.py            SQLite schema and settings
    snapshot.py         loads the usage ledger once per simulation version
    analytics.py        overview, cost explorer, inventory, resource detail
    anomalies.py        Isolation Forest on daily resource records + investigation workflow
    forecasting.py      month-end projection, backtest, budget status
    rightsizing.py      recommendations, decisions, applied actions, measured savings
    routes.py, schemas.py   /api/v1/demo endpoints and Pydantic models
ui/                     Streamlit client (api.py, components.py, format.py, pages/)
dashboard.py            Streamlit entry point
scripts/                Review-2 methodology code and validation scripts
data/, figures/         archived Review-2 datasets, metrics and figures
tests/                  pytest suite
```

## 6. Dataset and demo environment

**Benchmark dataset** (`data/multi_cloud_finops_dataset.csv`): 3,240 daily records from 1 Jan to 27 Sep 2026,
3 providers × 4 services, with 45 injected cost anomalies used only for evaluation. `scripts/generate_benchmark.py`
documents how it was generated (a deterministic reconstruction; the stored CSV remains the reference).

**Demo environment**: the simulation runs from 20 Aug 2026 to a starting clock of 18 Nov 2026, 10:00 IST, and
keeps an hourly ledger (about 68,000 rows in scenario A). Each row stores the resource, cost category, charge
type, billing unit, quantity, unit price, cost, CPU, memory and whether useful work was done in that hour.
Every number in the dashboard is aggregated from this one ledger, so the overview, cost explorer, inventory and
forecast agree with each other (this is checked by the tests).

Billing units follow the type of service:

- compute, database and Kubernetes nodes: running instance-hours × hourly price
- attached disks: GB-month, prorated hourly, still charged when the instance is stopped
- object storage: stored GB-month; data transfer: GB sent out

Prices are simplified on-demand list prices converted at about ₹84 per USD for Mumbai / Central India regions.
Taxes, discounts and reserved pricing are ignored. All times are shown in IST. Random variation comes from a seeded
generator keyed by resource and day, so the same scenario always produces the same numbers, and advancing
24 × 1 hour gives the same records as advancing 1 day.

Scenarios:

| Scenario | Situation |
|---|---|
| A - Normal operations | regular month, projection under budget |
| B - Forgotten development server | `loadtest-runner-01` (m6i.2xlarge) running for 24 days after a two-day test |
| C - Unexpected cost spike | the queue worker group scaled from 2 to 9 instances for 18 hours yesterday |
| D - Budget pressure | an A100 fine-tuning run started six days ago; projection goes over budget |
| E - Optimization review | idle and over-sized resources, including an ETL VM that no longer shuts down |

## 7. Forecasting and anomaly detection

**Forecasting (benchmark).** Seasonal Naive (value from 7 days earlier), Additive Seasonal (linear trend plus
weekly and monthly Fourier terms, fitted with Ridge regression) and an LSTM were compared on a chronological
225/45-day split. Code: `scripts/train_forecasting_models.py`.

**Forecasting (live demo).** On the simulated daily spend, both Seasonal Naive and Additive Seasonal are backtested
on the last 14 complete days, and the model with the lower MAPE is used for the projection (the same selection
rule as in Review-2). Before fitting, the history is adjusted to the current configuration: compute history of
stopped resources is removed, resized resources are re-priced, and days reported as anomalies are replaced by the
resource's typical cost. Resources launched in the last 14 days are added at their current run rate. The projected
month-end = actual month-to-date spend + forecast for the rest of the month. The range is ±1.96 × the standard
deviation of the backtest errors. The LSTM is not run live (it needs PyTorch and 450 training epochs).

**Anomaly detection.** The Isolation Forest code from `scripts/train_anomaly_detection.py` is reused without changes
(300 trees, contamination 45/3240, random_state 42, features `cost_ratio`, `cost_z`, `cpu_avg_pct`,
`memory_avg_pct`, `usage_hours`, `dayofweek`). In the demo it is fitted on daily records of every compute, database
and Kubernetes resource. A record is reported as a cost anomaly when the model flags it and its cost is at least
1.10× the resource's median daily cost; consecutive flagged days are grouped into one incident. In scenario A the
model flags some records but none are cost increases, so nothing is reported; in scenario C exactly one incident
is reported. The "possible explanations" are built from recorded metrics (instance-hours, CPU, active hours) and
are not presented as a confirmed root cause.

## 8. Rightsizing recommendations

The rules and savings fractions are the ones from Review-2 (`scripts/run_rightsizing.py`):

- **STOP_REVIEW**: CPU < 5% and memory < 10% and fewer than 9 active hours per day (saving estimate 85%)
- **DOWNSIZE**: CPU < 20% and memory < 30% and at least 9 active hours per day (saving estimate 45%)
- **RETAIN**: otherwise

In the demo they are applied to the last 7 days of each running resource (or the time since its last start or
resize). The saving estimate uses the resource's monthly cost at its current configuration. Approving a
recommendation only records the decision. "Apply in simulation" then stops, resizes or reschedules the simulated
resource, and the app measures the realized saving from the ledger afterwards, so proposed and realized savings
can be compared. Applied, rejected and snoozed items are not counted in potential savings, which avoids double
counting.

## 9. Installation (Windows, PowerShell)

```powershell
cd FinOps_Review2_Prototype
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

Optional: create the demo database explicitly (the backend also creates it automatically on first start):

```powershell
python -m app.init_demo
```

## 10. Running the application

Two terminals are needed. Start the backend first.

Terminal 1 - backend:

```powershell
.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal 2 - dashboard:

```powershell
.venv\Scripts\Activate.ps1
python -m streamlit run dashboard.py --server.port 8501
```

- Dashboard: http://localhost:8501
- API documentation (Swagger): http://127.0.0.1:8000/docs

`run_backend.bat` and `run_dashboard.bat` do the same from Explorer. Settings can be changed with environment
variables (see `.env.example`): `FINOPS_DB_PATH`, `FINOPS_DEFAULT_SCENARIO`, `FINOPS_API_URL`.

## 11. Operating the demo

1. Start on **Overview** (scenario A) to see spending, the budget and the attention list.
2. Load **scenario B** from the sidebar. The forgotten server appears in the attention list and on **Optimization**
   as STOP_REVIEW. Stop it from **Resources** and the projected month-end on **Forecast & Budget** drops.
3. Load **scenario C** and open **Anomalies** to inspect the spike and acknowledge it. Alternatively, in any scenario
   press *Trigger spending event* and then *+1 day*.
4. Load **scenario D** to see the over-budget warning and the savings that could reduce it. Edit the budget on
   **Forecast & Budget**.
5. Load **scenario E**, approve and apply a recommendation on **Optimization**, press *+1 day*, and compare
   the proposed saving with the realized one.
6. Use **Reset demo** to return to the starting state.

## 12. Tests

```powershell
python -m pytest -q
python scripts\validate_artifacts.py
python scripts\run_demo_checks.py
```

The pytest suite (51 tests) checks the cost formulas and billing units, determinism, stop/start behaviour,
reconciliation between pages, the original rightsizing thresholds, the decision and anomaly workflows,
forecast and budget logic, invalid requests, the seven original endpoints and the archived Review-2 values,
and renders every dashboard page through Streamlit's AppTest. `run_demo_checks.py` runs the artifact validator
and the tests and prints `DEMO READY`. `scripts/reproduce_experiments.py` recomputes the reproducible
benchmark results.

## 13. Experimental results (Review-2 benchmark) and their limits

| Result | Value | Reproduced by the code? |
|---|---|---|
| Dataset | 3,240 records, 270 days, 3 providers, 12 services | yes (stored CSV) |
| Seasonal Naive MAPE | 6.92% | yes, exactly |
| Additive Seasonal MAPE / RMSE | 3.04% / ₹587.70 | no - stored result; the reconstructed code gives 3.04% / ₹606.01 |
| LSTM MAPE | 8.45% | stored result; not retrained |
| Isolation Forest precision / recall / F1 | 0.889 / 0.889 / 0.889 | yes, exactly (TP 40, FP 5, FN 5) |
| Rightsizing reference agreement | 91.67% (11 of 12) | yes, exactly |
| Estimated monthly savings | ₹88,331.69 (21.25% of the portfolio) | yes, exactly |

These numbers come from simulated benchmark data, not real company billing. The savings are estimates, and the
anomaly labels were injected by the generator. Six of the twelve rightsizing utilization profiles in the benchmark
were injected as test scenarios, and one recommendation (AW-S3-APSouth) sits on a threshold boundary.

## 14. Current limitations

- No real cloud billing export or account is connected; all live data is simulated.
- Prices are simplified list prices; reserved instances, discounts, taxes and free tiers are not modelled.
- The demo forecast is for one month ahead and is computed on simulated, fairly regular spending, so its errors
  are lower than real bills would give.
- The rightsizing rules use averages. Short peaks are shown (95th percentile CPU) but not used in the decision.
- There is one simulated company and no user accounts or authentication.
- The LSTM is only available as an archived benchmark result.

## 15. Future improvements

- Import AWS Cost and Usage Reports, Azure Cost Management exports and GCP billing exports into the same ledger.
- Use provider pricing APIs for current rates and reserved-instance or savings-plan comparisons.
- Peak-aware rightsizing and instance family recommendations.
- Alerts by e-mail or chat when a projection crosses the budget or an anomaly is detected.
- User accounts, roles and an audit trail for approvals.
