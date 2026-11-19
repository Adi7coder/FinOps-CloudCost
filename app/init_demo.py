import argparse
import sys

from app.config import database_path
from app.demo import catalog, engine, store


def main():
    parser = argparse.ArgumentParser(description="Create or reset the local SQLite database for the startup demo.")
    parser.add_argument("--scenario", default="normal", choices=sorted(catalog.SCENARIOS))
    args = parser.parse_args()

    with store.session() as conn:
        engine.load_scenario(conn, args.scenario)
        rows = conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0]
        resources = conn.execute("SELECT COUNT(*) FROM resources").fetchone()[0]
        clock = store.get_setting(conn, "clock")
    print(f"Demo database ready: {database_path()}")
    print(f"Scenario: {catalog.SCENARIOS[args.scenario]['title']}")
    print(f"{resources} simulated resources, {rows:,} hourly usage records, simulation clock {clock} IST")
    return 0


if __name__ == "__main__":
    sys.exit(main())
