from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import timedelta

from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


def _run_step(module_name: str, *args: str) -> None:
    command = [sys.executable, "-m", module_name, *args]
    log.info("Running %s", " ".join(command))
    completed = subprocess.run(command, cwd=get_settings().base_dir)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill ingestors, aggregates, features, and predictions for a date range"
    )
    add_date_range_args(parser)
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip raw ingestors and only refresh aggregates, features, and predictions",
    )
    parser.add_argument(
        "--skip-starters",
        action="store_true",
        help="Skip starter and pitcher Statcast backfill when only hitter outcomes need updating",
    )
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)
    start_raw = start_date.isoformat()
    end_raw = end_date.isoformat()

    if not args.skip_ingest:
        ingest_steps: list[tuple[str, bool]] = [
            ("src.ingestors.games", True),
            ("src.ingestors.boxscores", True),
            ("src.ingestors.player_batting", True),
            ("src.ingestors.lineups_backfill", True),
            ("src.ingestors.matchup_splits", True),
            ("src.ingestors.lineups", False),
            ("src.ingestors.weather", True),
            ("src.ingestors.market_totals", True),
        ]
        if not args.skip_starters:
            ingest_steps.extend(
                [
                    ("src.ingestors.starters", True),
                    ("src.ingestors.pitcher_statcast", True),
                ]
            )
        for module_name, accepts_date_range in ingest_steps:
            if accepts_date_range:
                _run_step(module_name, "--start-date", start_raw, "--end-date", end_raw)
            else:
                _run_step(module_name)

    for module_name in (
        "src.transforms.offense_daily",
        "src.transforms.bullpens_daily",
        "src.transforms.freeze_markets",
        "src.features.totals_builder",
        "src.features.first5_totals_builder",
        "src.features.inning1_nrfi_builder",
        "src.features.hits_builder",
        "src.features.hr_builder",
        "src.features.total_bases_builder",
        "src.features.strikeouts_builder",
    ):
        _run_step(module_name, "--start-date", start_raw, "--end-date", end_raw)

    current_date = start_date
    while current_date <= end_date:
        current_raw = current_date.isoformat()
        _run_step("src.models.predict_totals", "--target-date", current_raw)
        _run_step("src.models.predict_first5_totals", "--target-date", current_raw)
        _run_step("src.models.predict_inning1_nrfi", "--target-date", current_raw)
        _run_step("src.models.predict_hits", "--target-date", current_raw)
        _run_step("src.models.predict_hr", "--target-date", current_raw)
        _run_step("src.models.predict_total_bases", "--target-date", current_raw)
        _run_step("src.models.predict_strikeouts", "--target-date", current_raw)
        current_date += timedelta(days=1)

    _run_step("src.transforms.product_surfaces", "--start-date", start_raw, "--end-date", end_raw)

    log.info("Range pipeline complete for %s to %s", start_raw, end_raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())