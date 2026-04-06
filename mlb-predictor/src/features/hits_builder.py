from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from src.features.common import (
    build_hitter_priors,
    build_pitcher_priors,
    build_team_priors,
    bullpen_snapshot,
    coerce_utc_timestamp_series,
    compute_board_state,
    compute_freshness_score,
    compute_starter_certainty,
    count_missing_fallbacks,
    default_cutoff,
    hitter_snapshot,
    infer_lineup_from_history,
    latest_market_snapshot,
    latest_weather_snapshot,
    offense_snapshot,
    park_snapshot,
    persist_hits_features,
    pitcher_snapshot,
    projected_plate_appearances,
    write_feature_snapshot,
)
from src.features.contracts import HITS_CERTAINTY_KEY_FIELDS, HITS_FEATURE_COLUMNS, HITS_META_COLUMNS, HITS_TARGET_COLUMN, validate_columns
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


def _load_frames(start_date, end_date, settings):
    history_start = f"{settings.prior_season}-01-01"
    return {
        "games": query_df(
            """
            SELECT game_id, game_date, game_start_ts, season, home_team, away_team
            FROM games
            WHERE game_date BETWEEN :start_date AND :end_date
            """,
            {"start_date": start_date, "end_date": end_date},
        ),
        "lineups": query_df(
            """
            SELECT *
            FROM lineups
            WHERE game_date BETWEEN :start_date AND :end_date
            """,
            {"start_date": start_date, "end_date": end_date},
        ),
        "players": query_df(
            """
            SELECT player_id, full_name, team_abbr
            FROM dim_players
            """
        ),
        "player_batting": query_df(
            """
            SELECT *
            FROM player_game_batting
            WHERE game_date >= :history_start AND game_date <= :end_date
            """,
            {"history_start": history_start, "end_date": end_date},
        ),
        "pitcher_starts": query_df(
            """
            SELECT *
            FROM pitcher_starts
            WHERE game_date >= :history_start AND game_date <= :end_date
            """,
            {"history_start": history_start, "end_date": end_date},
        ),
        "bullpens": query_df(
            """
            SELECT *
            FROM bullpens_daily
            WHERE game_date >= :history_start AND game_date <= :end_date
            """,
            {"history_start": history_start, "end_date": end_date},
        ),
        "team_offense": query_df(
            """
            SELECT *
            FROM team_offense_daily
            WHERE game_date >= :history_start AND game_date <= :end_date
            """,
            {"history_start": history_start, "end_date": end_date},
        ),
        "weather": query_df(
            """
            SELECT *
            FROM game_weather
            WHERE game_date BETWEEN :start_date AND :end_date
            """,
            {"start_date": start_date, "end_date": end_date},
        ),
        "markets": query_df(
            """
            SELECT *
            FROM game_markets
            WHERE game_date BETWEEN :start_date AND :end_date
              AND market_type = 'total'
            """,
            {"start_date": start_date, "end_date": end_date},
        ),
        "parks": query_df("SELECT * FROM park_factors"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build historical 1+ hit feature rows")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)
    settings = get_settings()
    frames = _load_frames(start_date, end_date, settings)

    games = frames["games"]
    if games.empty:
        log.info("Games missing for hits feature build")
        return 0

    for frame_name in ("player_batting", "pitcher_starts", "bullpens", "team_offense"):
        if not frames[frame_name].empty:
            frames[frame_name]["game_date"] = pd.to_datetime(frames[frame_name]["game_date"])
    for frame_name in ("lineups", "weather", "markets"):
        if not frames[frame_name].empty and "snapshot_ts" in frames[frame_name].columns:
            frames[frame_name]["snapshot_ts"] = coerce_utc_timestamp_series(frames[frame_name]["snapshot_ts"])
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    games["game_start_ts"] = pd.to_datetime(games["game_start_ts"], utc=True)

    hitter_priors = build_hitter_priors(frames["player_batting"], settings.prior_season)
    pitcher_priors = build_pitcher_priors(frames["pitcher_starts"], settings.prior_season)
    team_priors = build_team_priors(frames["team_offense"], settings.prior_season)
    prediction_ts = datetime.now(timezone.utc)
    feature_version = f"{settings.model_version_prefix}_hits_core"

    rows = []
    for game in games.itertuples(index=False):
        cutoff_ts = default_cutoff(game.game_date, game.game_start_ts)
        game_lineups = frames["lineups"][
            (frames["lineups"]["game_id"] == game.game_id) & (frames["lineups"]["snapshot_ts"] <= cutoff_ts)
        ].copy()

        latest_lineup_frames = []
        for lineup_team in (game.home_team, game.away_team):
            team_lineups = game_lineups[game_lineups["team"] == lineup_team].copy()
            if not team_lineups.empty:
                latest_snapshot = team_lineups["snapshot_ts"].max()
                latest_lineup_frames.append(team_lineups[team_lineups["snapshot_ts"] == latest_snapshot].copy())
                continue
            inferred = infer_lineup_from_history(
                lineup_team,
                game.game_date,
                frames["player_batting"],
                frames["players"],
            )
            if not inferred.empty:
                latest_lineup_frames.append(inferred)

        if not latest_lineup_frames:
            continue
        latest_lineups = pd.concat(latest_lineup_frames, ignore_index=True)
        starters = frames["pitcher_starts"][frames["pitcher_starts"]["game_id"] == game.game_id].copy()
        home_starter_id = None
        away_starter_id = None
        if not starters.empty:
            home_rows = starters[starters["side"] == "home"]
            away_rows = starters[starters["side"] == "away"]
            home_starter_id = int(home_rows.iloc[0]["pitcher_id"]) if not home_rows.empty else None
            away_starter_id = int(away_rows.iloc[0]["pitcher_id"]) if not away_rows.empty else None
            home_starter_probable = bool(home_rows.iloc[0]["is_probable"]) if not home_rows.empty else None
            away_starter_probable = bool(away_rows.iloc[0]["is_probable"]) if not away_rows.empty else None
        else:
            home_starter_probable = None
            away_starter_probable = None

        market = latest_market_snapshot(game.game_id, cutoff_ts, frames["markets"])
        weather = latest_weather_snapshot(game.game_id, cutoff_ts, frames["weather"])
        season = int(game.game_date.year) if pd.isna(game.season) else int(game.season)
        park = park_snapshot(game.home_team, season, frames["parks"], settings.prior_season)
        home_offense = offense_snapshot(
            game.home_team,
            game.game_date,
            frames["team_offense"],
            team_priors,
            full_weight_games=settings.team_full_weight_games,
            prior_blend_mode=settings.prior_blend_mode,
            prior_weight_multiplier=settings.prior_weight_multiplier,
        )
        away_offense = offense_snapshot(
            game.away_team,
            game.game_date,
            frames["team_offense"],
            team_priors,
            full_weight_games=settings.team_full_weight_games,
            prior_blend_mode=settings.prior_blend_mode,
            prior_weight_multiplier=settings.prior_weight_multiplier,
        )
        home_bullpen = bullpen_snapshot(game.home_team, game.game_date, frames["bullpens"])
        away_bullpen = bullpen_snapshot(game.away_team, game.game_date, frames["bullpens"])
        home_pitcher = pitcher_snapshot(
            home_starter_id,
            game.game_date,
            frames["pitcher_starts"],
            pitcher_priors,
            full_weight_starts=settings.pitcher_full_weight_starts,
            prior_blend_mode=settings.prior_blend_mode,
            prior_weight_multiplier=settings.prior_weight_multiplier,
        )
        away_pitcher = pitcher_snapshot(
            away_starter_id,
            game.game_date,
            frames["pitcher_starts"],
            pitcher_priors,
            full_weight_starts=settings.pitcher_full_weight_starts,
            prior_blend_mode=settings.prior_blend_mode,
            prior_weight_multiplier=settings.prior_weight_multiplier,
        )

        actual_hits = frames["player_batting"][frames["player_batting"]["game_id"] == game.game_id][["player_id", "hits"]].copy()
        actual_map = {int(row.player_id): int(row.hits) for row in actual_hits.itertuples(index=False)}

        game_start = game.game_start_ts.to_pydatetime() if pd.notna(game.game_start_ts) else None
        home_starter_cert = compute_starter_certainty(home_starter_id, home_starter_probable)
        away_starter_cert = compute_starter_certainty(away_starter_id, away_starter_probable)
        team_lineup_certainties = {}
        for lineup_team_name in (game.home_team, game.away_team):
            team_df = latest_lineups[latest_lineups["team"] == lineup_team_name]
            if not team_df.empty:
                confirmed = team_df["is_confirmed"].fillna(False).sum()
                team_lineup_certainties[lineup_team_name] = float(confirmed) / len(team_df)
            else:
                team_lineup_certainties[lineup_team_name] = 0.0
        weather_freshness = compute_freshness_score(
            weather["weather_snapshot_ts"], game_start, decay_hours=(3, 12, 24, 48),
        )
        market_freshness = compute_freshness_score(
            market["line_snapshot_ts"], game_start, decay_hours=(1, 6, 12, 24),
        )

        for lineup_row in latest_lineups.itertuples(index=False):
            lineup_team = lineup_row.team
            opponent = game.away_team if lineup_team == game.home_team else game.home_team
            opponent_starter = away_pitcher if lineup_team == game.home_team else home_pitcher
            opponent_bullpen = away_bullpen if lineup_team == game.home_team else home_bullpen
            team_environment = home_offense["runs_rate"] if lineup_team == game.home_team else away_offense["runs_rate"]
            hitter = hitter_snapshot(
                int(lineup_row.player_id),
                game.game_date,
                frames["player_batting"],
                hitter_priors,
                settings.min_pa_full_weight,
                prior_blend_mode=settings.prior_blend_mode,
                prior_weight_multiplier=settings.prior_weight_multiplier,
            )
            actual_player_hits = actual_map.get(int(lineup_row.player_id)) if actual_map else None
            rows.append(
                {
                    "game_id": int(game.game_id),
                    "game_date": game.game_date,
                    "player_id": int(lineup_row.player_id),
                    "team": lineup_team,
                    "opponent": opponent,
                    "prediction_ts": prediction_ts,
                    "game_start_ts": game.game_start_ts.to_pydatetime() if pd.notna(game.game_start_ts) else None,
                    "line_snapshot_ts": market["line_snapshot_ts"],
                    "feature_cutoff_ts": cutoff_ts,
                    "feature_version": feature_version,
                    "player_name": lineup_row.player_name,
                    "home_away": "H" if lineup_team == game.home_team else "A",
                    "lineup_slot": int(lineup_row.lineup_slot) if pd.notna(lineup_row.lineup_slot) else None,
                    "is_confirmed_lineup": bool(lineup_row.is_confirmed),
                    "projected_plate_appearances": projected_plate_appearances(lineup_row.lineup_slot),
                    "hit_rate_7": hitter["hit_rate_7"],
                    "hit_rate_14": hitter["hit_rate_14"],
                    "hit_rate_30": hitter["hit_rate_30"],
                    "hit_rate_blended": hitter["hit_rate_blended"],
                    "xba_14": hitter["xba_14"],
                    "xwoba_14": hitter["xwoba_14"],
                    "hard_hit_pct_14": hitter["hard_hit_pct_14"],
                    "k_pct_14": hitter["k_pct_14"],
                    "season_prior_hit_rate": hitter["season_prior_hit_rate"],
                    "season_prior_xba": hitter["season_prior_xba"],
                    "season_prior_xwoba": hitter["season_prior_xwoba"],
                    "opposing_starter_xwoba": opponent_starter["xwoba"],
                    "opposing_starter_csw": opponent_starter["csw"],
                    "opposing_bullpen_pitches_last3": opponent_bullpen["pitches_last3"],
                    "opposing_bullpen_innings_last3": opponent_bullpen["innings_last3"],
                    "venue_run_factor": park["run_factor"],
                    "park_hr_factor": park["hr_factor"],
                    "temperature_f": weather["temperature_f"],
                    "wind_speed_mph": weather["wind_speed_mph"],
                    "team_run_environment": team_environment,
                    "streak_len_capped": hitter["streak_len_capped"],
                    "got_hit": None if actual_player_hits is None else actual_player_hits > 0,
                }
            )
            row = rows[-1]
            row["starter_certainty_score"] = away_starter_cert if lineup_team == game.home_team else home_starter_cert
            row["lineup_certainty_score"] = team_lineup_certainties[lineup_team]
            row["weather_freshness_score"] = weather_freshness
            row["market_freshness_score"] = market_freshness
            row["bullpen_completeness_score"] = opponent_bullpen["completeness_3"]
            missing = count_missing_fallbacks(row, HITS_CERTAINTY_KEY_FIELDS)
            row["missing_fallback_count"] = missing
            row["board_state"] = compute_board_state(missing, threshold_minimal=2)

    feature_frame = pd.DataFrame(rows)
    if feature_frame.empty:
        log.info("No hits feature rows were built")
        return 0
    validate_columns(feature_frame, HITS_META_COLUMNS + HITS_FEATURE_COLUMNS + [HITS_TARGET_COLUMN], "hits")
    output_path = write_feature_snapshot(feature_frame, "hits", start_date, end_date)
    persisted = persist_hits_features(feature_frame, start_date, end_date)
    log.info("Built %s hit rows -> %s and persisted %s DB rows", len(feature_frame), output_path, persisted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())