import sys

from app.config import SCRIPTS_DIR

# The live demo reuses the Review-2 methodology code in scripts/ instead of keeping a second copy.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from preprocess_features import FEATURES  # noqa: E402
from run_rightsizing import SAVINGS_FRACTION, recommend  # noqa: E402
from train_anomaly_detection import (  # noqa: E402
    CONTAMINATION,
    N_ESTIMATORS,
    RANDOM_STATE,
    score_records,
    train_isolation_forest,
)
from train_forecasting_models import RIDGE_ALPHA, SEASONAL_LAG, build_seasonal_features  # noqa: E402
from train_forecasting_models import evaluate as evaluate_forecast  # noqa: E402

__all__ = [
    "FEATURES", "SAVINGS_FRACTION", "recommend", "CONTAMINATION", "N_ESTIMATORS", "RANDOM_STATE",
    "score_records", "train_isolation_forest", "RIDGE_ALPHA", "SEASONAL_LAG", "build_seasonal_features",
    "evaluate_forecast",
]
