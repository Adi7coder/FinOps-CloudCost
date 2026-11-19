
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from preprocess_features import DATA, build_daily_forecast_series, load_preprocessed

TRAIN_DAYS = 225
HOLDOUT_DAYS = 45
SEASONAL_LAG = 7                 # weekly seasonality
CANONICAL_METRICS = DATA / "forecast_metrics.csv"


#data preparation

def chronological_split(series: np.ndarray):
    assert len(series) == TRAIN_DAYS + HOLDOUT_DAYS, f"expected 270 daily values, got {len(series)}"
    return series[:TRAIN_DAYS], series[TRAIN_DAYS:]


#evaluation

def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = actual - predicted
    return {
        "MAPE_pct": float(100 * np.mean(np.abs(error / actual))),
        "RMSE_INR": float(np.sqrt(np.mean(error ** 2))),
        "MAE_INR": float(np.mean(np.abs(error))),
    }


#Seasonal Naive
def seasonal_naive_forecast(series: np.ndarray) -> np.ndarray:
    holdout_index = np.arange(TRAIN_DAYS, len(series))
    return series[holdout_index - SEASONAL_LAG]



#ASridge
RIDGE_ALPHA = 1.0
WEEKLY_PERIOD = 7
MONTHLY_PERIOD = 30.5
WEEKLY_HARMONICS = 2
MONTHLY_HARMONICS = 1
USE_QUADRATIC_TREND = False


def build_seasonal_features(n_days: int) -> np.ndarray:
    t = np.arange(n_days, dtype=float)
    trend = t / n_days                                   
    columns = [trend]
    if USE_QUADRATIC_TREND:
        columns.append(trend ** 2)

    for k in range(1, WEEKLY_HARMONICS + 1):
        columns.append(np.sin(2 * np.pi * k * t / WEEKLY_PERIOD))
        columns.append(np.cos(2 * np.pi * k * t / WEEKLY_PERIOD))

    for k in range(1, MONTHLY_HARMONICS + 1):
        columns.append(np.sin(2 * np.pi * k * t / MONTHLY_PERIOD))
        columns.append(np.cos(2 * np.pi * k * t / MONTHLY_PERIOD))

    return np.column_stack(columns)


def additive_seasonal_forecast(series: np.ndarray) -> np.ndarray:
    X = build_seasonal_features(len(series))
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X[:TRAIN_DAYS], series[:TRAIN_DAYS])
    return model.predict(X[TRAIN_DAYS:])


# LSTM

SEQUENCE_LENGTH = 28
INPUT_SIZE = 1
HIDDEN_SIZE = 32
NUM_LAYERS = 1
EPOCHS = 450
LEARNING_RATE = 0.01
GRADIENT_CLIP = 1.0
LSTM_SEED = 42


def make_sequences(scaled: np.ndarray, end: int):
    X, y = [], []
    for target_day in range(SEQUENCE_LENGTH, end):
        X.append(scaled[target_day - SEQUENCE_LENGTH:target_day])
        y.append(scaled[target_day])
    return np.array(X)[..., None], np.array(y)[:, None]


def train_lstm_forecast(series: np.ndarray, epochs: int = EPOCHS) -> np.ndarray:
    import torch
    from torch import nn

    class CostLSTM(nn.Module):

        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
            self.output = nn.Linear(HIDDEN_SIZE, 1)

        def forward(self, x):
            sequence_output, _ = self.lstm(x)
            return self.output(sequence_output[:, -1, :])  

    torch.manual_seed(LSTM_SEED)
    np.random.seed(LSTM_SEED)

    train_min, train_max = series[:TRAIN_DAYS].min(), series[:TRAIN_DAYS].max()
    scaled = (series - train_min) / (train_max - train_min)

    X_train, y_train = make_sequences(scaled, TRAIN_DAYS)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    model = CostLSTM()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

    X_holdout = []
    for target_day in range(TRAIN_DAYS, len(series)):
        X_holdout.append(scaled[target_day - SEQUENCE_LENGTH:target_day])
    X_holdout = torch.tensor(np.array(X_holdout)[..., None], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predicted_scaled = model(X_holdout).numpy().ravel()
    return predicted_scaled * (train_max - train_min) + train_min


#driver
def run_all(train_lstm: bool = False, lstm_epochs: int = EPOCHS) -> pd.DataFrame:
    daily = build_daily_forecast_series(load_preprocessed()).values
    _, holdout = chronological_split(daily)

    results = [
        {"model": "Seasonal Naive", **evaluate(holdout, seasonal_naive_forecast(daily))},
        {"model": "Additive Seasonal", **evaluate(holdout, additive_seasonal_forecast(daily))},
    ]
    if train_lstm:
        results.append({"model": "LSTM", **evaluate(holdout, train_lstm_forecast(daily, lstm_epochs))})
    return pd.DataFrame(results)


def print_comparison(recomputed: pd.DataFrame):
    canonical = pd.read_csv(CANONICAL_METRICS).set_index("model")
    print(f"{'model':18s}{'metric':10s}{'recomputed':>16s}{'canonical':>16s}  status")
    for _, row in recomputed.iterrows():
        for metric in ["MAPE_pct", "RMSE_INR", "MAE_INR"]:
            stored = canonical.loc[row["model"], metric]
            status = "MATCH" if abs(row[metric] - stored) < 1e-6 else "differs (canonical value is reported)"
            print(f"{row['model']:18s}{metric:10s}{row[metric]:>16.6f}{stored:>16.6f}  {status}")
    if "LSTM" not in set(recomputed["model"]):
        lstm = canonical.loc["LSTM"]
        print(f"{'LSTM':18s}not retrained; canonical MAPE {lstm['MAPE_pct']:.6f}%, RMSE {lstm['RMSE_INR']:.6f}, "
              f"MAE {lstm['MAE_INR']:.6f} (use --train-lstm to train)")
    best = canonical["MAPE_pct"].idxmin()
    print(f"\nCanonical best model (lowest holdout MAPE): {best}")


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate the Review-2 forecasting models.")
    parser.add_argument("--train-lstm", action="store_true", help="train the LSTM (requires PyTorch)")
    parser.add_argument("--lstm-epochs", type=int, default=EPOCHS)
    parser.add_argument("--output", type=Path, help="optional CSV path for the recomputed metrics")
    args = parser.parse_args()

    print(f"Chronological split: {TRAIN_DAYS} train days / {HOLDOUT_DAYS} holdout days (no shuffling)\n")
    recomputed = run_all(args.train_lstm, args.lstm_epochs)
    print_comparison(recomputed)

    if args.output:
        if args.output.resolve() == CANONICAL_METRICS.resolve():
            sys.exit("Refusing to overwrite the canonical data/forecast_metrics.csv.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        recomputed.to_csv(args.output, index=False)
        print(f"\nRecomputed metrics written to {args.output}")


if __name__ == "__main__":
    main()
