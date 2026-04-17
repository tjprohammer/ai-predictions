"""
Canonical module/step sequences for Update Center actions and /api/pipeline/run.

Tests should import `build_update_job_sequence` (and related builders) from here
instead of hard-coding expected module lists so they stay aligned with production.
"""

from __future__ import annotations

from datetime import date, timedelta

ModuleStep = tuple[str, list[str]]

RESULTS_BACKFILL_DAYS = 5

# Single source of truth for dashboard actions: labels and request validation both use these keys.
UPDATE_ACTION_LABELS: dict[str, str] = {
    "refresh_everything": "Refresh Everything",
    "prepare_slate": "Prepare Slate",
    "import_manual_inputs": "Update Lineups & Markets",
    "update_lineups_only": "Update Lineups",
    "update_markets_only": "Update Markets",
    "refresh_results": "Refresh Daily Results",
    "rebuild_predictions": "Rebuild Predictions",
    "grade_predictions": "Grade Predictions",
    "retrain_models": "Retrain Models",
}
UPDATE_JOB_ACTION_KEYS: frozenset[str] = frozenset(UPDATE_ACTION_LABELS.keys())


def _results_backfill_range(target_date: str) -> tuple[str, str]:
    """Return (start_date, end_date) covering recent days needing results backfill."""
    end = date.fromisoformat(target_date)
    start = end - timedelta(days=RESULTS_BACKFILL_DAYS)
    return start.isoformat(), end.isoformat()


def build_publish_target_date_sequence(
    target_date: str,
    *,
    refresh_aggregates: bool,
    rebuild_features: bool,
    include_market_freeze: bool,
) -> list[ModuleStep]:
    sequence: list[ModuleStep] = []
    if refresh_aggregates:
        sequence.extend(
            [
                ("src.transforms.offense_daily", []),
                ("src.transforms.bullpens_daily", []),
            ]
        )
    if rebuild_features:
        if include_market_freeze:
            sequence.append(("src.transforms.freeze_markets", ["--target-date", target_date]))
        sequence.extend(
            [
                ("src.features.totals_builder", ["--target-date", target_date]),
                ("src.features.first5_totals_builder", ["--target-date", target_date]),
                ("src.features.inning1_nrfi_builder", ["--target-date", target_date]),
                ("src.features.hits_builder", ["--target-date", target_date]),
                ("src.features.hr_builder", ["--target-date", target_date]),
                ("src.features.total_bases_builder", ["--target-date", target_date]),
                ("src.features.strikeouts_builder", ["--target-date", target_date]),
            ]
        )
    sequence.extend(
        [
            ("src.models.predict_totals", ["--target-date", target_date]),
            ("src.models.predict_first5_totals", ["--target-date", target_date]),
            ("src.models.predict_inning1_nrfi", ["--target-date", target_date]),
            ("src.models.predict_hits", ["--target-date", target_date]),
            ("src.models.predict_hr", ["--target-date", target_date]),
            ("src.models.predict_total_bases", ["--target-date", target_date]),
            ("src.models.predict_strikeouts", ["--target-date", target_date]),
            ("src.transforms.product_surfaces", ["--target-date", target_date]),
        ]
    )
    return sequence


def build_pipeline_run_sequence(
    target_date: str, refresh_aggregates: bool, rebuild_features: bool
) -> list[ModuleStep]:
    return build_publish_target_date_sequence(
        target_date,
        refresh_aggregates=refresh_aggregates,
        rebuild_features=rebuild_features,
        include_market_freeze=True,
    )


def build_results_refresh_sequence(target_date: str) -> list[ModuleStep]:
    backfill_start, backfill_end = _results_backfill_range(target_date)
    return [
        ("src.ingestors.games", ["--start-date", backfill_start, "--end-date", backfill_end]),
        ("src.ingestors.boxscores", ["--start-date", backfill_start, "--end-date", backfill_end]),
        ("src.ingestors.player_batting", ["--start-date", backfill_start, "--end-date", backfill_end]),
        (
            "src.ingestors.lineups_backfill",
            ["--start-date", backfill_start, "--end-date", backfill_end],
        ),
        ("src.ingestors.matchup_splits", ["--start-date", backfill_start, "--end-date", backfill_end]),
        ("src.ingestors.weather", ["--start-date", backfill_start, "--end-date", backfill_end, "--mode", "observed"]),
        ("src.transforms.offense_daily", ["--start-date", backfill_start, "--end-date", backfill_end]),
        ("src.transforms.bullpens_daily", ["--start-date", backfill_start, "--end-date", backfill_end]),
        ("src.transforms.product_surfaces", ["--target-date", target_date]),
    ]


def _manual_import_publish_sequence(target_date: str) -> list[ModuleStep]:
    """Rebuild features + predictions + surfaces after manual CSV / market ingest."""
    return build_publish_target_date_sequence(
        target_date,
        refresh_aggregates=False,
        rebuild_features=True,
        include_market_freeze=False,
    )


def build_update_job_sequence(action: str, target_date: str) -> list[ModuleStep]:
    if action == "refresh_everything":
        backfill_start, backfill_end = _results_backfill_range(target_date)
        return [
            ("src.ingestors.games", ["--target-date", target_date]),
            ("src.ingestors.starters", ["--target-date", target_date]),
            ("src.ingestors.prepare_slate_inputs", ["--target-date", target_date]),
            ("src.ingestors.lineups", ["--target-date", target_date]),
            ("src.ingestors.player_status", ["--target-date", target_date]),
            ("src.ingestors.market_totals", ["--target-date", target_date]),
            ("src.ingestors.weather", ["--target-date", target_date]),
            ("src.ingestors.umpire", ["--target-date", target_date]),
            ("src.transforms.freeze_markets", ["--target-date", target_date]),
            ("src.ingestors.validator", ["--target-date", target_date]),
            ("src.ingestors.games", ["--start-date", backfill_start, "--end-date", backfill_end]),
            ("src.ingestors.boxscores", ["--start-date", backfill_start, "--end-date", backfill_end]),
            ("src.ingestors.player_batting", ["--start-date", backfill_start, "--end-date", backfill_end]),
            (
                "src.ingestors.lineups_backfill",
                ["--start-date", backfill_start, "--end-date", backfill_end],
            ),
            ("src.ingestors.matchup_splits", ["--start-date", backfill_start, "--end-date", backfill_end]),
            ("src.ingestors.weather", ["--start-date", backfill_start, "--end-date", backfill_end, "--mode", "observed"]),
            ("src.transforms.offense_daily", ["--start-date", backfill_start, "--end-date", backfill_end]),
            ("src.transforms.bullpens_daily", ["--start-date", backfill_start, "--end-date", backfill_end]),
            ("src.features.totals_builder", ["--target-date", target_date]),
            ("src.features.first5_totals_builder", ["--target-date", target_date]),
            ("src.features.inning1_nrfi_builder", ["--target-date", target_date]),
            ("src.features.hits_builder", ["--target-date", target_date]),
            ("src.features.hr_builder", ["--target-date", target_date]),
            ("src.features.total_bases_builder", ["--target-date", target_date]),
            ("src.features.strikeouts_builder", ["--target-date", target_date]),
            ("src.models.predict_totals", ["--target-date", target_date]),
            ("src.models.predict_first5_totals", ["--target-date", target_date]),
            ("src.models.predict_inning1_nrfi", ["--target-date", target_date]),
            ("src.models.predict_hits", ["--target-date", target_date]),
            ("src.models.predict_hr", ["--target-date", target_date]),
            ("src.models.predict_total_bases", ["--target-date", target_date]),
            ("src.models.predict_strikeouts", ["--target-date", target_date]),
            ("src.transforms.product_surfaces", ["--target-date", target_date]),
        ]
    if action == "prepare_slate":
        return [
            ("src.ingestors.games", ["--target-date", target_date]),
            ("src.ingestors.starters", ["--target-date", target_date]),
            ("src.ingestors.prepare_slate_inputs", ["--target-date", target_date]),
            ("src.ingestors.lineups", ["--target-date", target_date]),
            ("src.ingestors.player_status", ["--target-date", target_date]),
            ("src.ingestors.market_totals", ["--target-date", target_date]),
            ("src.ingestors.weather", ["--target-date", target_date]),
            ("src.ingestors.umpire", ["--target-date", target_date]),
            ("src.ingestors.matchup_splits", ["--target-date", target_date]),
            ("src.transforms.freeze_markets", ["--target-date", target_date]),
            ("src.ingestors.validator", ["--target-date", target_date]),
            *build_publish_target_date_sequence(
                target_date,
                refresh_aggregates=False,
                rebuild_features=True,
                include_market_freeze=False,
            ),
        ]
    if action == "import_manual_inputs":
        return [
            ("src.ingestors.lineups", ["--target-date", target_date]),
            ("src.ingestors.player_status", ["--target-date", target_date]),
            ("src.ingestors.market_totals", ["--target-date", target_date]),
            ("src.ingestors.umpire", ["--target-date", target_date]),
            ("src.transforms.freeze_markets", ["--target-date", target_date]),
            ("src.ingestors.validator", ["--target-date", target_date]),
            *_manual_import_publish_sequence(target_date),
        ]
    if action == "update_lineups_only":
        return [
            ("src.ingestors.lineups", ["--target-date", target_date]),
            ("src.ingestors.player_status", ["--target-date", target_date]),
            ("src.ingestors.umpire", ["--target-date", target_date]),
            ("src.transforms.freeze_markets", ["--target-date", target_date]),
            ("src.ingestors.validator", ["--target-date", target_date]),
            *_manual_import_publish_sequence(target_date),
        ]
    if action == "update_markets_only":
        return [
            ("src.ingestors.market_totals", ["--target-date", target_date]),
            ("src.ingestors.umpire", ["--target-date", target_date]),
            ("src.transforms.freeze_markets", ["--target-date", target_date]),
            ("src.ingestors.validator", ["--target-date", target_date]),
            *_manual_import_publish_sequence(target_date),
        ]
    if action == "refresh_results":
        return build_results_refresh_sequence(target_date)
    if action == "grade_predictions":
        return build_results_refresh_sequence(target_date)
    if action == "retrain_models":
        return [
            ("src.models.train_totals", []),
            ("src.models.train_first5_totals", []),
            ("src.models.train_inning1_nrfi", []),
            ("src.models.train_hits", []),
            ("src.models.train_hr", []),
            ("src.models.train_strikeouts", []),
            *build_publish_target_date_sequence(
                target_date,
                refresh_aggregates=False,
                rebuild_features=False,
                include_market_freeze=False,
            ),
        ]
    return build_pipeline_run_sequence(target_date, refresh_aggregates=False, rebuild_features=True)


def label_for_update_action(action: str) -> str:
    try:
        return UPDATE_ACTION_LABELS[action]
    except KeyError as exc:
        raise KeyError(f"Unknown update action {action!r}") from exc
