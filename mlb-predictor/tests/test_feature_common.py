from src.features.common import build_hitter_priors, build_pitcher_priors, bullpen_snapshot, coerce_utc_timestamp_series, hitter_snapshot, latest_market_snapshot, pitcher_snapshot
from datetime import date, datetime, timezone
import sqlite3

import pandas as pd
import pytest

from types import SimpleNamespace

from src.features import common as feature_common
from src.features.common import build_hitter_priors, build_pitcher_priors, hitter_snapshot, latest_market_snapshot, pitcher_snapshot
from src.utils import db as db_utils


def test_build_hitter_priors_handles_empty_frame_with_object_dates():
    frame = pd.DataFrame(columns=["player_id", "game_date", "hits", "xba", "xwoba", "hard_hit_pct", "strikeouts", "plate_appearances"])

    result = build_hitter_priors(frame, 2025)

    assert result.empty


def test_build_pitcher_priors_coerces_string_game_dates():
    frame = pd.DataFrame(
        [
            {"pitcher_id": 1, "game_date": "2025-04-01", "xwoba_against": 0.31, "csw_pct": 0.28, "avg_fb_velo": 94.1},
            {"pitcher_id": 1, "game_date": "2026-04-01", "xwoba_against": 0.35, "csw_pct": 0.25, "avg_fb_velo": 93.4},
        ]
    )

    result = build_pitcher_priors(frame, 2025)

    assert float(result.loc[1, "prior_xwoba"]) == 0.31
    assert float(result.loc[1, "prior_csw"]) == 0.28


def test_hitter_snapshot_handles_empty_history_frame():
    frame = pd.DataFrame(columns=["player_id", "game_date", "hits", "xba", "xwoba", "hard_hit_pct", "strikeouts", "plate_appearances"])

    result = hitter_snapshot(1, date(2026, 4, 2), frame, pd.DataFrame(), 120)

    assert result["hit_rate_7"] is None
    assert result["streak_len_capped"] == 0


def test_pitcher_snapshot_handles_string_game_dates():
    frame = pd.DataFrame(
        [
            {"pitcher_id": 7, "game_date": "2026-03-28", "xwoba_against": 0.29, "csw_pct": 0.31, "avg_fb_velo": 95.0},
        ]
    )

    result = pitcher_snapshot(7, date(2026, 4, 2), frame, pd.DataFrame(), 10)

    assert result["xwoba"] == 0.29
    assert result["days_rest"] == 5


def test_bullpen_snapshot_surfaces_recent_run_prevention_metrics():
    frame = pd.DataFrame(
        [
            {"team": "DET", "game_date": "2026-03-29", "innings_pitched": 2.0, "pitches_thrown": 31, "runs_allowed": 1, "earned_runs": 1, "hits_allowed": 3},
            {"team": "DET", "game_date": "2026-03-31", "innings_pitched": 1.2, "pitches_thrown": 27, "runs_allowed": 0, "earned_runs": 0, "hits_allowed": 1},
            {"team": "DET", "game_date": "2026-04-01", "innings_pitched": 3.0, "pitches_thrown": 42, "runs_allowed": 2, "earned_runs": 2, "hits_allowed": 4},
        ]
    )

    result = bullpen_snapshot("DET", date(2026, 4, 2), frame)

    assert result["pitches_last3"] == 100
    assert result["runs_allowed_last3"] == 3
    assert result["earned_runs_last3"] == 3
    assert result["hits_allowed_last3"] == 8
    assert result["era_last3"] == pytest.approx(4.05, rel=1e-2)


def test_latest_market_snapshot_returns_selected_sportsbook():
    markets = pd.DataFrame(
        [
            {
                "game_id": 7,
                "sportsbook": "DraftKings",
                "line_value": 8.5,
                "over_price": -110,
                "under_price": -110,
                "snapshot_ts": pd.Timestamp("2026-04-02T16:00:00Z"),
            },
            {
                "game_id": 7,
                "sportsbook": "FanDuel",
                "line_value": 9.0,
                "over_price": -108,
                "under_price": -112,
                "snapshot_ts": pd.Timestamp("2026-04-02T17:00:00Z"),
            },
        ]
    )

    result = latest_market_snapshot(7, datetime(2026, 4, 2, 18, 0, tzinfo=timezone.utc), markets)

    assert result["market_total"] == 9.0
    assert result["market_sportsbook"] == "FanDuel"


def test_write_feature_snapshot_replaces_overlapping_ranges(monkeypatch, tmp_path):
    monkeypatch.setattr(feature_common, "get_settings", lambda: SimpleNamespace(feature_dir=tmp_path))
    lane_dir = tmp_path / "strikeouts"
    lane_dir.mkdir(parents=True, exist_ok=True)

    overlapping = lane_dir / "2026-04-03_2026-04-04.parquet"
    non_overlapping = lane_dir / "2026-04-01_2026-04-02.parquet"
    pd.DataFrame([{"game_id": 1}]).to_parquet(overlapping, index=False)
    pd.DataFrame([{"game_id": 2}]).to_parquet(non_overlapping, index=False)

    output_path = feature_common.write_feature_snapshot(
        pd.DataFrame([{"game_id": 3}]),
        "strikeouts",
        date(2026, 4, 4),
        date(2026, 4, 4),
    )

    assert output_path.exists()
    assert not overlapping.exists()
    assert non_overlapping.exists()


def test_coerce_utc_timestamp_series_handles_mixed_iso_formats():
    series = pd.Series([
        "2026-04-01 20:23:48.174274+00:00",
        "2026-04-01 20:23:48+0000",
    ])

    result = coerce_utc_timestamp_series(series)

    assert result.notna().all()
    assert str(result.iloc[0].tz) == "UTC"


def _configure_temp_sqlite(monkeypatch, tmp_path):
    database_path = tmp_path / "feature-persist.sqlite3"
    settings = SimpleNamespace(database_url=f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setattr(db_utils, "get_settings", lambda: settings)
    db_utils.get_engine.cache_clear()
    return database_path


def test_persist_totals_features_replaces_selected_date_rows(monkeypatch, tmp_path):
    database_path = _configure_temp_sqlite(monkeypatch, tmp_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE game_features_totals (
                feature_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id BIGINT,
                game_date DATE,
                home_team TEXT,
                away_team TEXT,
                prediction_ts TEXT,
                game_start_ts TEXT,
                line_snapshot_ts TEXT,
                feature_cutoff_ts TEXT,
                feature_version TEXT,
                feature_payload TEXT,
                actual_total_runs SMALLINT,
                created_at TEXT,
                updated_at TEXT,
                market_sportsbook TEXT,
                UNIQUE(game_id, feature_cutoff_ts, feature_version)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO game_features_totals (
                game_id, game_date, home_team, away_team, prediction_ts, game_start_ts,
                line_snapshot_ts, feature_cutoff_ts, feature_version, feature_payload,
                actual_total_runs, market_sportsbook
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                824460,
                "2026-04-04",
                "CLE",
                "CHC",
                "2026-04-04T19:00:00+00:00",
                "2026-04-04T23:15:00+00:00",
                "2026-04-04T19:00:00+00:00",
                "2026-04-04T23:15:00+00:00",
                "v1_totals_core",
                "{}",
                None,
                "FanDuel",
            ),
        )

    feature_common.persist_totals_features(
        pd.DataFrame(
            [
                {
                    "game_id": 824461,
                    "game_date": date(2026, 4, 4),
                    "home_team": "NYY",
                    "away_team": "BOS",
                    "prediction_ts": datetime(2026, 4, 4, 20, 0, tzinfo=timezone.utc),
                    "game_start_ts": datetime(2026, 4, 4, 23, 5, tzinfo=timezone.utc),
                    "line_snapshot_ts": datetime(2026, 4, 4, 19, 30, tzinfo=timezone.utc),
                    "feature_cutoff_ts": datetime(2026, 4, 4, 23, 5, tzinfo=timezone.utc),
                    "feature_version": "v1_totals_core",
                    "market_sportsbook": "FanDuel",
                    "market_total": 8.5,
                    "market_over_price": -110,
                    "market_under_price": -110,
                    "line_movement": 0.0,
                    "actual_total_runs": None,
                }
            ]
        ),
        date(2026, 4, 4),
        date(2026, 4, 4),
    )

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT game_id, game_date FROM game_features_totals ORDER BY game_id"
        ).fetchall()

    db_utils.get_engine.cache_clear()
    assert rows == [(824461, "2026-04-04")]


@pytest.mark.parametrize(
    ("table_name", "persist_fn", "id_column", "row_key"),
    [
        ("player_features_hits", feature_common.persist_hits_features, "player_id", 700001),
        ("game_features_pitcher_strikeouts", feature_common.persist_strikeout_features, "pitcher_id", 800001),
    ],
)
def test_persist_entity_features_replaces_selected_date_rows(
    monkeypatch,
    tmp_path,
    table_name,
    persist_fn,
    id_column,
    row_key,
):
    database_path = _configure_temp_sqlite(monkeypatch, tmp_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            f"""
            CREATE TABLE {table_name} (
                feature_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id BIGINT,
                game_date DATE,
                {id_column} BIGINT,
                team TEXT,
                opponent TEXT,
                prediction_ts TEXT,
                game_start_ts TEXT,
                line_snapshot_ts TEXT,
                feature_cutoff_ts TEXT,
                feature_version TEXT,
                feature_payload TEXT,
                {'got_hit BOOLEAN' if id_column == 'player_id' else 'actual_strikeouts SMALLINT'},
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(game_id, {id_column}, feature_cutoff_ts, feature_version)
            )
            """
        )
        connection.execute(
            f"""
            INSERT INTO {table_name} (
                game_id, game_date, {id_column}, team, opponent, prediction_ts, game_start_ts,
                line_snapshot_ts, feature_cutoff_ts, feature_version, feature_payload,
                {'got_hit' if id_column == 'player_id' else 'actual_strikeouts'}
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                824460,
                "2026-04-04",
                row_key,
                "CLE",
                "CHC",
                "2026-04-04T19:00:00+00:00",
                "2026-04-04T23:15:00+00:00",
                "2026-04-04T19:00:00+00:00",
                "2026-04-04T23:15:00+00:00",
                "v1_core",
                "{}",
                None,
            ),
        )

    base_row = {
        "game_id": 824461,
        "game_date": date(2026, 4, 4),
        id_column: row_key + 1,
        "team": "NYY",
        "opponent": "BOS",
        "prediction_ts": datetime(2026, 4, 4, 20, 0, tzinfo=timezone.utc),
        "game_start_ts": datetime(2026, 4, 4, 23, 5, tzinfo=timezone.utc),
        "line_snapshot_ts": datetime(2026, 4, 4, 19, 30, tzinfo=timezone.utc),
        "feature_cutoff_ts": datetime(2026, 4, 4, 23, 5, tzinfo=timezone.utc),
        "feature_version": "v1_core",
    }
    if id_column == "player_id":
        base_row["got_hit"] = None
        base_row["projected_plate_appearances"] = 4.2
    else:
        base_row["actual_strikeouts"] = None
        base_row["baseline_strikeouts"] = 5.1

    persist_fn(
        pd.DataFrame([base_row]),
        date(2026, 4, 4),
        date(2026, 4, 4),
    )

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            f"SELECT game_id, game_date, {id_column} FROM {table_name} ORDER BY game_id, {id_column}"
        ).fetchall()

    db_utils.get_engine.cache_clear()
    assert rows == [(824461, "2026-04-04", row_key + 1)]