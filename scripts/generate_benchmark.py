
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
CANONICAL = DATA / "multi_cloud_finops_dataset.csv"
DEFAULT_OUTPUT = DATA / "generated_multi_cloud_finops_dataset.csv"

SEED = 42
START_DATE = "2026-01-01"
DAYS = 270
COLUMNS = ["record_id", "date", "cloud", "service", "team", "region", "cost_inr", "cpu_avg_pct", "memory_avg_pct",
           "usage_hours", "anomaly_label", "anomaly_type"]
NUMERIC = ["cost_inr", "cpu_avg_pct", "memory_avg_pct", "usage_hours"]

TREND_PER_DAY = 0.0011
WEEKLY_AMP, WEEKDAY_ADJ, WEEKEND_ADJ = 0.08, 0.05, -0.07
MONTHLY_AMP, MONTHLY_PERIOD = 0.06, 30.5
COST_NOISE_SD = 0.035
SPIKE_COST = (1.75, 2.65)
SPIKE_CPU = (1.25, 1.70)
SPIKE_MEM = (1.10, 1.45)
IDLE_CPU, IDLE_MEM, IDLE_HOURS = (1.0, 4.0), (2.0, 8.0), (2.0, 8.0)


SERVICE_PROFILES = [
    ("AWS", "EC2", "AI-ML", "IN-South", 1476.90, 40.41, 10.27, 46.16, 10.98, 0.27, 21.89, 2.36),
    ("AWS", "RDS", "Backend", "IN-Central", 1053.25, 51.63, 12.48, 61.93, 12.00, 0.42, 21.87, 2.16),
    ("AWS", "S3", "Data", "AP-South", 359.51, 5.65, 9.89, 7.68, 8.90, 0.04, 23.90, 2.31),
    ("AWS", "EKS", "Platform", "IN-South", 877.25, 46.08, 12.43, 53.16, 11.26, 0.35, 21.63, 2.19),
    ("Azure", "VirtualMachines", "Backend", "AP-South", 1187.62, 39.27, 11.31, 44.81, 11.23, 0.21, 21.96, 2.39),
    ("Azure", "SQLDatabase", "Data", "IN-South", 1002.34, 48.24, 11.74, 60.83, 12.21, 0.41, 22.12, 2.31),
    ("Azure", "BlobStorage", "Platform", "IN-Central", 330.52, 4.53, 10.76, 6.67, 8.59, -0.06, 23.90, 2.12),
    ("Azure", "AKS", "AI-ML", "AP-South", 840.17, 44.45, 11.73, 51.72, 11.27, 0.36, 22.16, 2.25),
    ("GCP", "ComputeEngine", "Data", "IN-Central", 1211.55, 40.24, 10.74, 45.35, 10.35, 0.28, 21.78, 1.99),
    ("GCP", "CloudSQL", "Platform", "AP-South", 980.84, 47.79, 12.90, 59.24, 11.89, 0.40, 22.03, 2.06),
    ("GCP", "CloudStorage", "AI-ML", "IN-South", 340.82, 4.75, 10.41, 5.80, 9.35, -0.03, 23.89, 2.36),
    ("GCP", "GKE", "Backend", "IN-Central", 859.30, 46.06, 12.59, 53.25, 11.92, 0.31, 21.86, 2.32),
]
CLOUDS = ["AWS", "Azure", "GCP"]
STORAGE_SERVICE_INDEX = 2 


def is_cost_spike(d: int, service_index: int, cloud_index: int) -> bool:
    return ((d * 37 + service_index * 11 + cloud_index * 17) % 97 == 0
            or (d + 13 * service_index + 7 * cloud_index) % 211 == 0)


def is_idle_episode(d: int, service_index: int, cloud_index: int) -> bool:
    return service_index != STORAGE_SERVICE_INDEX and (d + 9 * service_index + 4 * cloud_index) % 83 == 0


def cost_multiplier(d: int, weekday: bool) -> float:
    trend = 1.0 + TREND_PER_DAY * d
    weekly = 1.0 + WEEKLY_AMP * np.sin(2 * np.pi * d / 7) + (WEEKDAY_ADJ if weekday else WEEKEND_ADJ)
    monthly = 1.0 + MONTHLY_AMP * np.sin(2 * np.pi * d / MONTHLY_PERIOD)
    return trend * weekly * monthly


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=DAYS, freq="D")
    rows = []
    for d, date in enumerate(dates):
        weekday = date.dayofweek < 5
        seasonal = cost_multiplier(d, weekday)
        for k, (cloud, service, team, region, base, cpu_mu, cpu_sd, mem_mu, mem_sd, rho, h_mu, h_sd) in enumerate(SERVICE_PROFILES):
            cloud_index, service_index = CLOUDS.index(cloud), k % 4
            spike = is_cost_spike(d, service_index, cloud_index)
            idle = is_idle_episode(d, service_index, cloud_index)

            cost = base * seasonal * rng.normal(1.0, COST_NOISE_SD)
            z_cpu, z_ind = rng.standard_normal(2)
            cpu = np.clip(cpu_mu + cpu_sd * z_cpu, 1.0, 99.0)
            mem = np.clip(mem_mu + mem_sd * (rho * z_cpu + np.sqrt(1 - rho ** 2) * z_ind), 1.0, 99.0)
            hours = np.clip(rng.normal(h_mu, h_sd), 0.0, 24.0)

            spike_cost, spike_cpu, spike_mem = rng.uniform(*SPIKE_COST), rng.uniform(*SPIKE_CPU), rng.uniform(*SPIKE_MEM)
            if spike:
                cost *= spike_cost
                cpu = min(cpu * spike_cpu, 99.0)
                mem = min(mem * spike_mem, 99.0)

            idle_cpu, idle_mem, idle_hours = rng.uniform(*IDLE_CPU), rng.uniform(*IDLE_MEM), rng.uniform(*IDLE_HOURS)
            if idle:  # idle utilization overrides spike utilization; cost is unaffected
                cpu, mem, hours = idle_cpu, idle_mem, idle_hours

            rows.append({
                "record_id": len(rows), "date": date.strftime("%Y-%m-%d"), "cloud": cloud, "service": service,
                "team": team, "region": region, "cost_inr": round(cost, 2), "cpu_avg_pct": round(cpu, 2),
                "memory_avg_pct": round(mem, 2), "usage_hours": round(hours, 2),
                "anomaly_label": int(spike), "anomaly_type": "cost_spike" if spike else "normal",
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def compare(gen: pd.DataFrame, ref: pd.DataFrame) -> bool:
    print("\n== Comparison with canonical data/multi_cloud_finops_dataset.csv")
    print(f"{'':28s}{'generated':>12s}{'canonical':>12s}")
    for label, fn in [("rows", len), ("dates", lambda x: x["date"].nunique()), ("clouds", lambda x: x["cloud"].nunique()),
                      ("services", lambda x: x["service"].nunique()), ("anomalies", lambda x: int(x["anomaly_label"].sum()))]:
        print(f"{label:28s}{fn(gen):>12}{fn(ref):>12}")
    if len(gen) != len(ref):
        print("\nEXACT BENCHMARK REPRODUCTION: NOT YET EXACT (row counts differ)")
        return False

    def flag(ok):
        return "MATCH" if ok else "DIFF"
    categorical = ["record_id", "date", "cloud", "service", "team", "region", "anomaly_type"]
    cat_ok = all((gen[c].values == ref[c].values).all() for c in categorical)
    print(f"\ncategorical columns {categorical}: {flag(cat_ok)}")
    gen_pos, ref_pos = set(gen.index[gen["anomaly_label"] == 1]), set(ref.index[ref["anomaly_label"] == 1])
    print(f"anomaly positions: {flag(gen_pos == ref_pos)} ({len(gen_pos & ref_pos)}/{len(ref_pos)} matched, "
          f"{len(gen_pos - ref_pos)} extra)")
    gen_idle, ref_idle = set(gen.index[gen["usage_hours"] < 9]), set(ref.index[ref["usage_hours"] < 9])
    print(f"idle-episode positions (usage_hours < 9): {flag(gen_idle == ref_idle)} ({len(gen_idle & ref_idle)}/{len(ref_idle)} matched)")

    print(f"\n{'numeric field':16s}{'MAE':>10s}{'max abs err':>13s}{'exact rows':>12s}{'gen mean':>10s}{'ref mean':>10s}")
    for c in NUMERIC:
        err = (gen[c] - ref[c]).abs()
        print(f"{c:16s}{err.mean():>10.3f}{err.max():>13.3f}{int((err < 0.005).sum()):>12d}{gen[c].mean():>10.2f}{ref[c].mean():>10.2f}")

    print(f"\n{'per-service mean':16s}{'cost gen/ref':>20s}{'cpu gen/ref':>16s}{'mem gen/ref':>16s}{'hours gen/ref':>16s}")
    g, r = gen.groupby("service", sort=False)[NUMERIC].mean(), ref.groupby("service", sort=False)[NUMERIC].mean()
    for s in r.index:
        print(f"{s:16s}{g.loc[s, 'cost_inr']:>10.1f}/{r.loc[s, 'cost_inr']:<9.1f}{g.loc[s, 'cpu_avg_pct']:>8.1f}/{r.loc[s, 'cpu_avg_pct']:<7.1f}"
              f"{g.loc[s, 'memory_avg_pct']:>8.1f}/{r.loc[s, 'memory_avg_pct']:<7.1f}{g.loc[s, 'usage_hours']:>8.2f}/{r.loc[s, 'usage_hours']:<7.2f}")
    daily_g, daily_r = gen.groupby("date")["cost_inr"].sum(), ref.groupby("date")["cost_inr"].sum()
    print(f"daily total cost correlation (shared trend/seasonality): {np.corrcoef(daily_g, daily_r)[0, 1]:.3f}")

    same = (gen[COLUMNS].astype(str).values == ref[COLUMNS].astype(str).values).all(axis=1)
    print(f"\nexactly matching rows: {int(same.sum())}/{len(ref)} ({100 * same.mean():.2f}%)")
    exact = bool(same.all())
    if exact:
        print("EXACT BENCHMARK REPRODUCTION: PASS")
    else:
        differing = [c for c in COLUMNS if not (gen[c].astype(str).values == ref[c].astype(str).values).all()]
        print("EXACT BENCHMARK REPRODUCTION: NOT YET EXACT")
        print(f"Differing fields: {differing}. Structure, mappings and anomaly/idle positions are reproduced; the random "
              "draws of the original generator could not be recovered, so per-row values differ. This output is a "
              "deterministic benchmark reconstruction, not the original historical generator. The canonical CSV remains "
              "the source of truth for all Review-2 results.")
    return exact


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true",
                        help="also replace data/multi_cloud_finops_dataset.csv (timestamped backup first)")
    parser.add_argument("--force", action="store_true",
                        help="with --overwrite: replace the canonical CSV even if reproduction is not exact")
    parser.add_argument("--strict", action="store_true", help="exit 1 unless reproduction is exact")
    args = parser.parse_args()

    gen = generate(args.seed)
    output = args.output.resolve()
    if output == CANONICAL.resolve():
        sys.exit("Refusing to write to the canonical CSV via --output; use --overwrite instead.")
    gen.to_csv(output, index=False)
    print(f"Generated records: {len(gen)}")
    print(f"Days: {gen['date'].nunique()}")
    print(f"Clouds: {gen['cloud'].nunique()}")
    print(f"Services: {gen['service'].nunique()}")
    print(f"Injected anomalies: {int(gen['anomaly_label'].sum())}")
    print(f"Output: {output.relative_to(BASE) if output.is_relative_to(BASE) else output}")

    exact = compare(gen, pd.read_csv(CANONICAL)) if CANONICAL.exists() else False

    if args.overwrite:
        if not exact and not args.force:
            print("\n--overwrite refused: the reconstruction is not exact, and replacing the canonical CSV would change "
                  "downstream Review-2 results. Add --force only if that is really intended.")
            return 1
        if CANONICAL.exists():
            backup = CANONICAL.with_name(f"{CANONICAL.stem}.backup_{datetime.now():%Y%m%d_%H%M%S}.csv")
            shutil.copy2(CANONICAL, backup)
            print(f"\nBackup written: {backup.relative_to(BASE)}")
        gen.to_csv(CANONICAL, index=False)
        print(f"Canonical CSV replaced: {CANONICAL.relative_to(BASE)}")
    return 1 if args.strict and not exact else 0


if __name__ == "__main__":
    sys.exit(main())
