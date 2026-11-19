
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from preprocess_features import DATA, FEATURES, build_anomaly_features, load_preprocessed

N_ESTIMATORS = 300
CONTAMINATION = 45 / 3240        # expected anomaly share of the benchmark
RANDOM_STATE = 42


def train_isolation_forest(features: pd.DataFrame) -> IsolationForest:
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    model.fit(features[FEATURES])
    return model


def score_records(model: IsolationForest, features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    raw_prediction = model.predict(features[FEATURES])          
    out["iforest_pred"] = (raw_prediction == -1).astype(int)
    out["anomaly_score"] = -model.score_samples(features[FEATURES])
    return out


def evaluate(scored: pd.DataFrame) -> dict:
    y_true = scored["anomaly_label"]
    y_pred = scored["iforest_pred"]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "true_anomalies": int(y_true.sum()),
        "flagged": int(y_pred.sum()),
    }


def compare_with_canonical(scored: pd.DataFrame, metrics: dict):
    canonical_scored = pd.read_csv(DATA / "anomaly_scored_dataset.csv")
    canonical_metrics = pd.read_csv(DATA / "anomaly_metrics.csv").iloc[0]

    same_predictions = (scored["iforest_pred"].values == canonical_scored["iforest_pred"].values).all()
    same_scores = np.allclose(scored["anomaly_score"].values, canonical_scored["anomaly_score"].values)
    print(f"predictions identical to canonical: {same_predictions}")
    print(f"anomaly scores identical to canonical: {same_scores}")
    for name in ["precision", "recall", "f1"]:
        status = "MATCH" if abs(metrics[name] - canonical_metrics[name]) < 1e-9 else "DIFF"
        print(f"{name:10s} recomputed {metrics[name]:.16f}  canonical {canonical_metrics[name]:.16f}  {status}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the Isolation Forest anomaly detector.")
    parser.add_argument("--output", type=Path, help="optional CSV path for the scored dataset")
    args = parser.parse_args()

    features = build_anomaly_features(load_preprocessed())
    model = train_isolation_forest(features)
    scored = score_records(model, features)
    metrics = evaluate(scored)

    print(f"Features: {FEATURES}")
    print(f"IsolationForest(n_estimators={N_ESTIMATORS}, contamination=45/3240, random_state={RANDOM_STATE})\n")
    print("Confusion matrix (rows = actual, columns = predicted)")
    print(f"               pred Normal  pred Anomaly")
    print(f"actual Normal  {metrics['TN']:>11d}  {metrics['FP']:>12d}")
    print(f"actual Anomaly {metrics['FN']:>11d}  {metrics['TP']:>12d}\n")
    print(f"true anomalies {metrics['true_anomalies']}, flagged {metrics['flagged']}")
    compare_with_canonical(scored, metrics)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = scored.copy()
        output["date"] = output["date"].dt.strftime("%Y-%m-%d")
        output.to_csv(args.output, index=False)
        print(f"\nScored dataset written to {args.output}")


if __name__ == "__main__":
    main()
