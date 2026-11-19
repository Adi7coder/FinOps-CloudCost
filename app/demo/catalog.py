from datetime import datetime

COMPANY_NAME = "Tasklane Labs Pvt. Ltd."
COMPANY_DESCRIPTION = "Fictional B2B workflow SaaS startup used for the demo (not a real company or cloud account)"
HISTORY_START = datetime(2026, 8, 20, 0, 0)
BASELINE_CLOCK = datetime(2026, 11, 18, 10, 0)
DEFAULT_MONTHLY_BUDGET = 400000.0
TIMEZONE_LABEL = "IST"

REGIONS = {"AWS": "ap-south-1 (Mumbai)", "Azure": "Central India", "GCP": "asia-south1 (Mumbai)"}
DISK_PRICE_PER_GB_MONTH = {"AWS": 6.70, "Azure": 5.80, "GCP": 8.40}
UNIT_BY_CATEGORY = {"Compute": "instance-hour", "Database": "instance-hour", "Kubernetes": "node-hour",
                    "Storage": "GB-month", "Network": "GB"}
USAGE_KINDS = ("web", "service", "database", "worker", "job", "dev", "idle", "batch", "continuous")

PRICING_ASSUMPTIONS = [
    "Rates are simplified on-demand list prices converted at about INR 84 per USD (Mumbai / Central India regions).",
    "Compute, database and Kubernetes nodes are billed per running instance-hour.",
    "Attached disks are billed per GB-month (prorated hourly) and keep billing while an instance is stopped.",
    "Object storage is billed per GB-month on the stored volume; data transfer is billed per GB sent out.",
    "Taxes, discounts, reserved instances and free tiers are ignored.",
]


def resource(resource_id, name, provider, service, category, kind, environment, team, resource_class, purpose,
             unit_price, instances=1, max_instances=None, disk_gb=0.0, disk_price=None, status="running",
             launched_days_ago=None, **profile):
    return {
        "resource_id": resource_id, "name": name, "provider": provider, "service": service,
        "category": category, "kind": kind, "environment": environment, "team": team,
        "region": REGIONS[provider], "resource_class": resource_class, "purpose": purpose,
        "unit": UNIT_BY_CATEGORY[category], "unit_price": unit_price, "instances": instances,
        "max_instances": max_instances, "disk_gb": disk_gb,
        "disk_price": DISK_PRICE_PER_GB_MONTH[provider] if disk_price is None else disk_price,
        "status": status, "launched_days_ago": launched_days_ago, "profile": profile,
    }


BASE_RESOURCES = [
    resource("asg-web-prod", "web-frontend-prod", "AWS", "EC2", "Compute", "web", "production", "Platform",
             "m6i.large (auto scaling 2-5)", "Serves the customer web application behind the load balancer",
             8.70, instances=2, max_instances=5, disk_gb=60, cpu_low=28, cpu_high=66, mem_low=45, mem_high=60),
    resource("i-0a7c3e91f42b6d815", "api-gateway-prod", "AWS", "EC2", "Compute", "service", "production",
             "Backend", "c6i.xlarge x3", "REST API used by the web and mobile clients",
             16.10, instances=3, disk_gb=90, cpu=42, mem=55),
    resource("db-orders-prod", "orders-db-prod", "AWS", "RDS", "Database", "database", "production", "Backend",
             "db.r6g.large Multi-AZ (PostgreSQL)", "Primary transactional database for orders and invoices",
             45.20, disk_gb=200, disk_price=9.60, cpu=38, mem=68),
    resource("asg-queue-workers", "queue-workers-prod", "AWS", "EC2", "Compute", "worker", "production",
             "Backend", "c6i.large x2", "Background workers that render invoice PDFs and send e-mails from SQS",
             8.05, instances=2, disk_gb=40, start=7, end=23, p_active=0.92, cpu=55, mem=46, idle_cpu=3,
             idle_mem=28),
    resource("i-03be52d9c6a1f7e40", "report-batch-runner", "AWS", "EC2", "Compute", "job", "production", "Data",
             "c6i.2xlarge", "Scheduled job that builds nightly customer usage reports (01:00-03:00 IST)",
             32.20, disk_gb=50, status="scheduled", start=1, end=3, cpu=70, mem=55, idle_cpu=1.2, idle_mem=5),
    resource("i-0d91f6a24c8e3b571", "staging-api", "AWS", "EC2", "Compute", "dev", "staging", "Backend",
             "t3.xlarge", "Staging copy of the API used for QA before releases",
             15.30, disk_gb=50, start=9, end=21, p_weekday=0.92, p_weekend=0.48, cpu=16, mem=32, idle_cpu=2.5,
             idle_mem=18),
    resource("s3-tasklane-media", "tasklane-media", "AWS", "S3", "Storage", "storage", "production", "Platform",
             "S3 Standard", "User uploads, invoice PDFs and static assets",
             1.93, gb=3200, growth=0.0012),
    resource("aws-egress-aps1", "aws-data-transfer-out", "AWS", "Data Transfer", "Network", "transfer",
             "production", "Platform", "Internet egress", "Traffic sent from AWS to users on the internet",
             7.56, gb_per_hour=5.2),
    resource("aks-platform-np1", "aks-platform-nodepool", "Azure", "AKS", "Kubernetes", "service", "production",
             "Platform", "Standard_D4s_v5 x3", "Kubernetes node pool for the notification and auth services",
             17.10, instances=3, disk_gb=384, cpu=46, mem=61),
    resource("sqldb-analytics-gp4", "analytics-sql-db", "Azure", "SQL Database", "Database", "database",
             "production", "Data", "General Purpose Gen5 4 vCore", "Reporting database used by internal dashboards",
             61.50, cpu=11, mem=25),
    resource("st-tasklane-backups", "tasklane-backups", "Azure", "Blob Storage", "Storage", "storage",
             "production", "Data", "Blob Storage (Cool tier)", "Nightly database backups and audit log archive",
             0.84, gb=7600, growth=0.0015),
    resource("vm-dev-frontend-01", "dev-frontend-vm", "Azure", "Virtual Machines", "Compute", "dev",
             "development", "Platform", "B2ms", "Shared development VM for the frontend team",
             7.00, disk_gb=64, start=10, end=19, p_weekday=0.80, p_weekend=0.08, cpu=22, mem=38, idle_cpu=1.8,
             idle_mem=14),
    resource("vm-etl-nightly", "etl-nightly-vm", "Azure", "Virtual Machines", "Compute", "job", "production",
             "Data", "F8s_v2", "Scheduled ETL job that loads the analytics database (02:00-04:00 IST)",
             28.50, disk_gb=128, status="scheduled", start=2, end=4, cpu=38, mem=48, idle_cpu=1.2, idle_mem=5),
    resource("gke-ml-serving-pool", "gke-ml-serving", "GCP", "GKE", "Kubernetes", "service", "production",
             "AI-ML", "n2-standard-4 x2", "Serves the document classification model behind the API",
             17.20, instances=2, disk_gb=200, cpu=51, mem=64),
    resource("csql-events-ha", "events-db", "GCP", "Cloud SQL", "Database", "database", "production", "Backend",
             "PostgreSQL 2 vCPU / 8 GB HA", "Event store for audit trails and webhooks",
             21.80, disk_gb=100, disk_price=14.30, cpu=34, mem=58),
    resource("gce-ml-train-t4", "ml-training-t4", "GCP", "Compute Engine", "Compute", "batch", "development",
             "AI-ML", "n1-standard-8 + 1x T4", "Model training runs every Tuesday and Friday (08:00-22:00 IST)",
             62.40, disk_gb=200, status="scheduled", days=[1, 4], start=8, end=22, cpu=82, mem=70),
    resource("gcs-tasklane-datalake", "tasklane-datalake", "GCP", "Cloud Storage", "Storage", "storage",
             "production", "Data", "Cloud Storage Standard", "Raw event data and training datasets",
             1.68, gb=11000, growth=0.0018),
    resource("gcp-egress-as1", "gcp-network-egress", "GCP", "Network Egress", "Network", "transfer",
             "production", "AI-ML", "Internet egress", "Model responses and dataset downloads",
             10.10, gb_per_hour=1.6),
    resource("gce-analytics-nb-01", "analytics-notebook", "GCP", "Compute Engine", "Compute", "dev",
             "development", "Data", "e2-standard-8", "Jupyter notebook server for the data team",
             22.70, disk_gb=100, start=9, end=21, p_weekday=0.95, p_weekend=0.50, cpu=14, mem=28, idle_cpu=2,
             idle_mem=16),
    resource("gce-dev-sandbox-02", "dev-api-sandbox", "GCP", "Compute Engine", "Compute", "dev", "development",
             "Backend", "e2-standard-2", "Personal sandbox for API experiments",
             5.70, disk_gb=30, start=10, end=19, p_weekday=0.75, p_weekend=0.05, cpu=20, mem=35, idle_cpu=1.8,
             idle_mem=13),
]

LOADTEST_RUNNER = resource(
    "i-0f4e8a2c71d9b3065", "loadtest-runner-01", "AWS", "EC2", "Compute", "idle", "development", "Backend",
    "m6i.2xlarge", "Launched for a load test of the billing API; nobody stopped it afterwards",
    34.80, disk_gb=100, launched_days_ago=24, busy_days=2, cpu=58, mem=52, idle_cpu=1.6, idle_mem=5.5)

LLM_FINETUNE = resource(
    "gce-llm-finetune-a100", "llm-finetune-a100", "GCP", "Compute Engine", "Compute", "continuous",
    "development", "AI-ML", "a2-highgpu-1g (1x A100)", "Fine-tuning run for the support-ticket classifier",
    309.00, disk_gb=500, launched_days_ago=6, cpu=79, mem=74)

SCENARIOS = {
    "normal": {
        "title": "A - Normal operations",
        "summary": "Regular month: production services, a few development machines and scheduled jobs.",
        "extra_resources": [], "events": [], "always_on": {},
    },
    "forgotten_server": {
        "title": "B - Forgotten development server",
        "summary": "A load-test machine has been left running for 24 days after a two-day test.",
        "extra_resources": [LOADTEST_RUNNER], "events": [], "always_on": {},
    },
    "cost_spike": {
        "title": "C - Unexpected cost spike",
        "summary": "Yesterday the queue worker group scaled to 9 instances for 18 hours.",
        "extra_resources": [],
        "events": [{"resource_id": "asg-queue-workers", "start_hours_ago": 29, "duration_hours": 18,
                    "factor": 4.5, "label": "Queue worker group scaled from 2 to 9 instances"}],
        "always_on": {},
    },
    "budget_pressure": {
        "title": "D - Budget pressure",
        "summary": "A GPU fine-tuning run started six days ago and the load-test machine is still running.",
        "extra_resources": [LOADTEST_RUNNER, LLM_FINETUNE], "events": [], "always_on": {},
    },
    "optimization_review": {
        "title": "E - Optimization review",
        "summary": "Several idle or over-sized resources, including an ETL VM that no longer shuts down.",
        "extra_resources": [LOADTEST_RUNNER], "events": [], "always_on": {"vm-etl-nightly": 10},
    },
}

SPENDING_EVENTS = {
    "worker_scale_out": {"resource_id": "asg-queue-workers", "duration_hours": 18, "factor": 4.5,
                         "label": "Queue worker autoscaling misconfiguration (scaled from 2 to 9 instances)"},
}
