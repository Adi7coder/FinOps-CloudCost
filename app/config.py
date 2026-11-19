import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
SCRIPTS_DIR = BASE_DIR / "scripts"

API_VERSION = "1.0.0"


def database_path() -> Path:
    return Path(os.environ.get("FINOPS_DB_PATH", BASE_DIR / "runtime" / "finops_demo.db"))


def default_scenario() -> str:
    return os.environ.get("FINOPS_DEFAULT_SCENARIO", "normal")
