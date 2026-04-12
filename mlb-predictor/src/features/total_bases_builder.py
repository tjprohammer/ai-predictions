"""Build per-batter total bases feature rows used to train / predict total bases.

Mirrors hits_builder.py — same game loop, lineup resolution, and pitcher snapshots.
Extends with TB-specific features (tb_rate_*, iso_14, hr_rate_14) and the
batter_total_bases prop market line.
"""
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
    pitcher_snapshot,
    projected_plate_appearances,
    write_feature_snapshot,
)
from src.features.contracts import (
    TOTAL_BASES_CERTAINTY_KEY_FIELDS,
    TOTAL_BASES_FEATURE_COLUMNS,
    TOTAL_BASES_META_COLUMNS,
    TOTAL_BASES_TARGET_COLUMN,
    validate_columns,
)
from src.features.totals_builder import _umpire_run_value
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df, table_exists, upsert_rows
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)

_TB_META = TOTAL_BASES_META_COLUMNS


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
            """SELECT * FROM lineups WHERE game_date BETWEEN :start_date AND :end_date""",
            {"start_date": start_date, "end_date": end_date},
        ),
        "players": query_df("SELECT player_id, full_name, team_abbr FROM dim_players"),
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
            SELECT * FROM bullpens_daily
            WHERE game_date >= :history_start AND game_date <= :end_date
            """,
            {"history_start": history_start, "end_date": end_date},
        ),
        "team_offense": query_df(
            """
            SELECT * FROM team_offense_daily
            WHERE game_date >= :history_start AND game_date <= :end_date
            """,
            {"history_start": history_start, "end_date": end_date},
        ),
        "weather": query_df(
            """
            SELECT * FROM game_weather WHERE game_date BETWEEN :start_date AND :end_date
            """,
            {"start_date": start_date, "end_date": end_date},
        ),
        "game_markets": query_df(
            """
            SELECT * FROM game_markets
            WHERE game_date BETWEEN :start_date AND :end_date AND market_type = 'total'
            """,
            {"start_date": start_date, "end_date": end_date},
        ),
        "tb_prop_markets": query_df(
            """
            SELECT * FROM player_prop_markets
            WHERE game_date BETWEEN :start_date AND :end_date
              AND market_type = 'batter_total_bases'
            """,
            {"start_date": start_date, "end_date": end_date},
        ),
        "parks": query_df("SELECT * FROM park_factors"),
        "umpire_assignments": query_df(
            """
            SELECT ua.game_id, ua.game_date, ua.umpire_name, ua.umpire_id
            FROM umpire_assignments ua
            INNER JOIN (
                SELECT game_id, MAX(snapshot_ts) AS max_ts
                FROM umpire_assignments GROUP BY game_id
            ) latest ON ua.game_id = latest.game_id AND ua.snapshot_ts = latest.max_ts
            """
        ) if table_exists("umpire_assignments") else pd.DataFrame(),
        "games_history": query_df(
            """
            SELECT game_id, game_date, total_runs FROM games
            WHERE game_date >= :history_start AND game_date <= :end_date
              AND total_runs IS NOT NULL
            """,
            {"history_start": history_start, "end_date": end_date},
        ),
    }



def _hitter_tb_snapshot(player_id: int, before_date, player_batting: pd.DataFrame) -> dict:
    """Compute total-bases rolling rates for a hitter from raw game log."""
    pb = player_batting[player_batting["player_id"] == player_id].copy()
    if pb.empty:
        return {"tb_rate_7": None, "tb_rate_14": None, "tb_rate_30": None, "iso_14": None, "hr_rate_14": None}
    if "game_date" in pb.columns:
        pb["game_date"] = pd.to_datetime(pb["game_date"]).dt.date
    past = pb[pb["game_date"] < before_date].sort_values("game_date", ascending=False)
    if past.empty:
        return {"tb_rate_7": None, "tb_rate_14": None, "tb_rate_30": None, "iso_14": None, "hr_rate_14": None}
    for col in ("total_bases", "hits", "doubles", "triples", "home_runs", "plate_appearances"):
        past[col] = pd.to_numeric(past.get(col), errors="coerce").fillna(0)
    def _rate(window: pd.DataFrame) -> float | None:
        g = len(window)
        return round(float(window["total_bases"].sum()) / g, 4) if g > 0 else None
    p7, p14, p30 = past.head(7), past.head(14), past.head(30)
    pa_14 = p14["plate_appearances"].sum()
    iso_14 = round(float(p14["total_bases"].sum() - p14["hits"].sum()) / pa_14, 4) if pa_14 > 0 else None
    hr_14 = round(float(p14["home_runs"].sum()) / pa_14, 4) if pa_14 > 0 else None
    return {"tb_rate_7": _rate(p7), "tb_rate_14": _rate(p14), "tb_rate_30": _rate(p30),
            "iso_14": iso_14, "hr_rate_14": hr_14}


def _latest_tb_prop_market(game_id: int, player_id: int, cutoff_ts, tb_prop_markets: pd.DataFrame) -> dict:
    """Return the freshest batter_total_bases market for this player."""
    empty = {"market_tb_line": None, "market_tb_over_price": None, "market_tb_under_price": None}
    if tb_prop_markets.empty:
        return empty
    gm = tb_prop_markets[
        (tb_prop_markets["game_id"] == game_id) & (tb_prop_markets["player_id"] == player_id)
        & (tb_prop_markets["snapshot_ts"] <= cutoff_ts)
    ]
    if gm.empty:
        return empty
    latest = gm.loc[gm["snapshot_ts"].idxmax()]
    return {
        "market_tb_line": float(latest.get("line_value")) if pd.notna(latest.get("line_value")) else None,
        "market_tb_over_price": int(latest.get("over_price")) if pd.notna(latest.get("over_price")) else None,
        "market_tb_under_price": int(latest.get("under_price")) if pd.notna(latest.get("under_price")) else None,
    }


def _persist_tb_features(frame: pd.DataFrame, start_date, end_date) -> int:
    meta_set = set(_TB_META)
    rows_to_upsert = []
    for record in frame.to_dict(orient="records"):
        payload = {k: v for k, v in record.items() if k not in meta_set and k != TOTAL_BASES_TARGET_COLUMN}
        row = {col: record.get(col) for col in _TB_META}
        row["feature_payload"] = payload
        row[TOTAL_BASES_TARGET_COLUMN] = record.get(TOTAL_BASES_TARGET_COLUMN)
        rows_to_upsert.append(row)
    return upsert_rows(
        "game_features_total_bases",
        rows_to_upsert,
        conflict_columns=["game_id", "player_id", "feature_cutoff_ts", "feature_version"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build batter total-bases feature rows")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)
    settings = get_settings()
    frames = _load_frames(start_date, end_date, settings)

    games = frames["games"]
    if games.empty:
        log.info("No games for total bases feature build")
        return 0

    for fn in ("player_batting", "pitcher_starts", "bullpens", "team_offense", "games_history"):
        if not frames[fn].empty:
            frames[fn]["game_date"] = pd.to_datetime(frames[fn]["game_date"])

    # Compute total_bases from hit components (not stored as a column in player_game_batting)
    pb = frames["player_batting"]
    if not pb.empty and "total_bases" not in pb.columns:
        frames["player_batting"]["total_bases"] = (
            pb.get("singles", pd.Series(0, index=pb.index)).fillna(0)
            + 2 * pb.get("doubles", pd.Series(0, index=pb.index)).fillna(0)
            + 3 * pb.get("triples", pd.Series(0, index=pb.index)).fillna(0)
            + 4 * pb.get("home_runs", pd.Series(0, index=pb.index)).fillna(0)
        ).astype(int)
    for fn in ("lineups", "weather", "game_markets", "tb_prop_markets"):
        if not frames[fn].empty and "snapshot_ts" in frames[fn].columns:
            frames[fn]["snapshot_ts"] = coerce_utc_timestamp_series(frames[fn]["snapshot_ts"])
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    games["game_start_ts"] = pd.to_datetime(games["game_start_ts"], utc=True)

    hitter_priors = build_hitter_priors(frames["player_batting"], settings.prior_season)
    pitcher_priors = build_pitcher_priors(frames["pitcher_starts"], settings.prior_season)
    team_priors = build_team_priors(frames["team_offense"], settings.prior_season)
    prediction_ts = datetime.now(timezone.utc)
    feature_version = f"{settings.model_version_prefix}_total_bases_core"

    rows = []
    for game in games.itertuples(index=False):
        cutoff_ts = default_cutoff(game.game_date, game.game_start_ts)
        game_lineups = frames["lineups"][
            (frames["lineups"]["game_id"] == game.game_id) & (frames["lineups"]["snapshot_ts"] <= cutoff_ts)
        ].copy()
        latest_lineup_frames = []
        for lu_team in (game.home_team, game.away_team):
            tlu = game_lineups[game_lineups["team"] == lu_team].copy()
            if not tlu.empty:
                snap = tlu["snapshot_ts"].max()
                latest_lineup_frames.append(tlu[tlu["snapshot_ts"] == snap].copy())
            else:
                inf = infer_lineup_from_history(lu_team, game.game_date, frames["player_batting"], frames["players"])
                if not inf.empty:
                    latest_lineup_frames.append(inf)
        if not latest_lineup_frames:
            continue
        latest_lineups = pd.concat(latest_lineup_frames, ignore_index=True)
        starters = frames["pitcher_starts"][frames["pitcher_starts"]["game_id"] == game.game_id].copy()
        home_starter_id = away_starter_id = None
        home_starter_probable = away_starter_probable = None
        if not starters.empty:
            hr_s = starters[starters["side"] == "home"]; ar_s = starters[starters["side"] == "away"]
            home_starter_id = int(hr_s.iloc[0]["pitcher_id"]) if not hr_s.empty else None
            away_starter_id = int(ar_s.iloc[0]["pitcher_id"]) if not ar_s.empty else None
            home_starter_probable = bool(hr_s.iloc[0]["is_probable"]) if not hr_s.empty else None
            away_starter_probable = bool(ar_s.iloc[0]["is_probable"]) if not ar_s.empty else None
        market = latest_market_snapshot(game.game_id, cutoff_ts, frames["game_markets"])
        weather = latest_weather_snapshot(game.game_id, cutoff_ts, frames["weather"])
        season = int(game.game_date.year) if pd.isna(game.season) else int(game.season)
        park = park_snapshot(game.home_team, season, frames["parks"], settings.prior_season)
        home_offense = offense_snapshot(game.home_team, game.game_date, frames["team_offense"], team_priors,
                                        full_weight_games=settings.team_full_weight_games,
                                        prior_blend_mode=settings.prior_blend_mode,
                                        prior_weight_multiplier=settings.prior_weight_multiplier)
        away_offense = offense_snapshot(game.away_team, game.game_date, frames["team_offense"], team_priors,
                                        full_weight_games=settings.team_full_weight_games,
                                        prior_blend_mode=settings.prior_blend_mode,
                                        prior_weight_multiplier=settings.prior_weight_multiplier)
        home_bullpen = bullpen_snapshot(game.home_team, game.game_date, frames["bullpens"])
        away_bullpen = bullpen_snapshot(game.away_team, game.game_date, frames["bullpens"])
        home_pitcher = pitcher_snapshot(home_starter_id, game.game_date, frames["pitcher_starts"], pitcher_priors,
                                        full_weight_starts=settings.pitcher_full_weight_starts,
                                        prior_blend_mode=settings.prior_blend_mode,
                                        prior_weight_multiplier=settings.prior_weight_multiplier)
        away_pitcher = pitcher_snapshot(away_starter_id, game.game_date, frames["pitcher_starts"], pitcher_priors,
                                        full_weight_starts=settings.pitcher_full_weight_starts,
                                        prior_blend_mode=settings.prior_blend_mode,
                                        prior_weight_multiplier=settings.prior_weight_multiplier)
        actual_tb_map: dict[int, int] = {}
        actuals = frames["player_batting"][frames["player_batting"]["game_id"] == game.game_id][["player_id", "total_bases"]].copy()
        for r in actuals.dropna(subset=["total_bases"]).itertuples(index=False):
            actual_tb_map[int(r.player_id)] = int(r.total_bases)
        game_start = game.game_start_ts.to_pydatetime() if pd.notna(game.game_start_ts) else None
        home_sc = compute_starter_certainty(home_starter_id, home_starter_probable)
        away_sc = compute_starter_certainty(away_starter_id, away_starter_probable)
        lineup_certs: dict[str, float] = {}
        for lt in (game.home_team, game.away_team):
            td = latest_lineups[latest_lineups["team"] == lt]
            lineup_certs[lt] = float(td["is_confirmed"].fillna(False).sum()) / len(td) if not td.empty else 0.0
        weather_freshness = compute_freshness_score(weather["weather_snapshot_ts"], game_start)
        market_freshness = compute_freshness_score(market["line_snapshot_ts"], game_start)
        ump_row = frames["umpire_assignments"][frames["umpire_assignments"]["game_id"] == int(game.game_id)] \
            if not frames["umpire_assignments"].empty else pd.DataFrame()
        ump_name = str(ump_row.iloc[0]["umpire_name"]) if not ump_row.empty else None
        ump_run_val = _umpire_run_value(ump_name, game.game_date, frames["umpire_assignments"], frames["games_history"])

        _PITCHER_POSITIONS = {"P", "SP", "RP", "CP", "MR", "LHP", "RHP"}
        for lr in latest_lineups.itertuples(index=False):
            fp = str(getattr(lr, "field_position", "") or "").upper().strip()
            if fp in _PITCHER_POSITIONS:
                continue
            lu_team = lr.team
            opponent = game.away_team if lu_team == game.home_team else game.home_team
            opp_pitcher = away_pitcher if lu_team == game.home_team else home_pitcher
            opp_bullpen = away_bullpen if lu_team == game.home_team else home_bullpen
            team_env = home_offense["runs_rate"] if lu_team == game.home_team else away_offense["runs_rate"]
            hitter = hitter_snapshot(int(lr.player_id), game.game_date, frames["player_batting"], hitter_priors,
                                     settings.min_pa_full_weight, prior_blend_mode=settings.prior_blend_mode,
                                     prior_weight_multiplier=settings.prior_weight_multiplier)
            tb_stats = _hitter_tb_snapshot(int(lr.player_id), game.game_date, frames["player_batting"])
            tb_market = _latest_tb_prop_market(int(game.game_id), int(lr.player_id), cutoff_ts, frames["tb_prop_markets"])
            row = {
                "game_id": int(game.game_id), "game_date": game.game_date,
                "player_id": int(lr.player_id), "team": lu_team, "opponent": opponent,
                "prediction_ts": prediction_ts, "game_start_ts": game_start,
                "line_snapshot_ts": market["line_snapshot_ts"], "feature_cutoff_ts": cutoff_ts,
                "feature_version": feature_version, "player_name": lr.player_name,
                "home_away": "H" if lu_team == game.home_team else "A",
                "lineup_slot": int(lr.lineup_slot) if pd.notna(lr.lineup_slot) else None,
                "is_confirmed_lineup": bool(lr.is_confirmed),
                "projected_plate_appearances": projected_plate_appearances(lr.lineup_slot),
                "hit_rate_7": hitter["hit_rate_7"], "hit_rate_14": hitter["hit_rate_14"],
                "hit_rate_30": hitter["hit_rate_30"], "hit_rate_blended": hitter["hit_rate_blended"],
                "xba_14": hitter["xba_14"], "xwoba_14": hitter["xwoba_14"],
                "hard_hit_pct_14": hitter["hard_hit_pct_14"], "k_pct_14": hitter["k_pct_14"],
                "season_prior_hit_rate": hitter["season_prior_hit_rate"],
                "season_prior_xwoba": hitter["season_prior_xwoba"],
                "tb_rate_7": tb_stats["tb_rate_7"], "tb_rate_14": tb_stats["tb_rate_14"],
                "tb_rate_30": tb_stats["tb_rate_30"], "iso_14": tb_stats["iso_14"],
                "hr_rate_14": tb_stats["hr_rate_14"],
                "opposing_starter_xwoba": opp_pitcher["xwoba"],
                "opposing_starter_hard_hit_pct": opp_pitcher["hard_hit_pct"],
                "opposing_starter_k_pct": opp_pitcher.get("k_pct"),
                "opposing_bullpen_pitches_last3": opp_bullpen["pitches_last3"],
                "venue_run_factor": park["run_factor"], "park_hr_factor": park["hr_factor"],
                "temperature_f": weather["temperature_f"], "wind_speed_mph": weather["wind_speed_mph"],
                "team_run_environment": team_env,
                "game_total_line": market.get("market_total"),
                "ump_run_value": ump_run_val,
                "market_tb_line": tb_market["market_tb_line"],
                "market_tb_over_price": tb_market["market_tb_over_price"],
                "market_tb_under_price": tb_market["market_tb_under_price"],
                "starter_certainty_score": away_sc if lu_team == game.home_team else home_sc,
                "lineup_certainty_score": lineup_certs[lu_team],
                "weather_freshness_score": weather_freshness,
                "market_freshness_score": market_freshness,
                TOTAL_BASES_TARGET_COLUMN: actual_tb_map.get(int(lr.player_id)),
            }
            row["missing_fallback_count"] = count_missing_fallbacks(row, TOTAL_BASES_CERTAINTY_KEY_FIELDS)
            row["board_state"] = compute_board_state(row["missing_fallback_count"], threshold_minimal=2)
            rows.append(row)

    feature_frame = pd.DataFrame(rows)
    if feature_frame.empty:
        log.info("No total bases feature rows were built")
        return 0
    validate_columns(feature_frame, TOTAL_BASES_META_COLUMNS + TOTAL_BASES_FEATURE_COLUMNS + [TOTAL_BASES_TARGET_COLUMN], "total_bases")
    output_path = write_feature_snapshot(feature_frame, "total_bases", start_date, end_date)
    persisted = _persist_tb_features(feature_frame, start_date, end_date)
    log.info("Built %s total-bases rows → %s, persisted %s DB rows", len(feature_frame), output_path, persisted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
