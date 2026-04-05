from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import delete_for_date_range, query_df, table_exists, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)


def _is_final_game_status(status: Any) -> bool:
    normalized = str(status or "").strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in ("final", "completed", "game over", "closed"))


def _hit_streak(values: pd.Series) -> list[int]:
    streak = 0
    results = []
    for value in values.astype(int).tolist():
        if value > 0:
            streak += 1
        else:
            streak = 0
        results.append(streak)
    return results


def _rolling_rate(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def _rolling_avg(numerator: pd.Series, denominator: pd.Series, window: int) -> pd.Series:
    num = numerator.rolling(window, min_periods=1).sum()
    den = denominator.rolling(window, min_periods=1).sum().replace(0, np.nan)
    return num / den


def _concat_group_results(frame: pd.DataFrame, group_column: str, decorator) -> pd.DataFrame:
    groups = [decorator(group) for _, group in frame.groupby(group_column, sort=False)]
    if not groups:
        return pd.DataFrame(columns=frame.columns)
    return pd.concat(groups, ignore_index=False)


def _build_player_trends(start_date, end_date) -> int:
    history_start = pd.Timestamp(start_date) - pd.Timedelta(days=21)
    frame = query_df(
        """
        SELECT game_id, game_date, player_id, team, opponent, lineup_slot, hits, at_bats, plate_appearances, xwoba, hard_hit_pct
        FROM player_game_batting
        WHERE game_date BETWEEN :history_start AND :end_date
        ORDER BY player_id, game_date, game_id
        """,
        {"history_start": history_start.date(), "end_date": end_date},
    )
    if frame.empty:
        return 0
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    frame["had_hit"] = (pd.to_numeric(frame["hits"], errors="coerce").fillna(0) > 0).astype(int)

    def _decorate(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(["game_date", "game_id"]).copy()
        hits = pd.to_numeric(group["hits"], errors="coerce")
        at_bats = pd.to_numeric(group["at_bats"], errors="coerce")
        xwoba = pd.to_numeric(group["xwoba"], errors="coerce")
        hard_hit_pct = pd.to_numeric(group["hard_hit_pct"], errors="coerce")
        group["rolling_hit_rate_3"] = _rolling_rate(group["had_hit"], 3)
        group["rolling_hit_rate_5"] = _rolling_rate(group["had_hit"], 5)
        group["rolling_hit_rate_7"] = _rolling_rate(group["had_hit"], 7)
        group["rolling_hit_rate_14"] = _rolling_rate(group["had_hit"], 14)
        group["rolling_batting_avg_7"] = _rolling_avg(hits.fillna(0), at_bats.fillna(0), 7)
        group["rolling_batting_avg_14"] = _rolling_avg(hits.fillna(0), at_bats.fillna(0), 14)
        group["rolling_xwoba_7"] = xwoba.rolling(7, min_periods=1).mean()
        group["rolling_xwoba_14"] = xwoba.rolling(14, min_periods=1).mean()
        group["rolling_hard_hit_pct_7"] = hard_hit_pct.rolling(7, min_periods=1).mean()
        group["rolling_hard_hit_pct_14"] = hard_hit_pct.rolling(14, min_periods=1).mean()
        group["hit_streak"] = _hit_streak(group["had_hit"])
        return group

    trend_frame = _concat_group_results(frame, "player_id", _decorate)
    trend_frame = trend_frame[(trend_frame["game_date"].dt.date >= start_date) & (trend_frame["game_date"].dt.date <= end_date)].copy()
    rows = []
    for record in trend_frame.to_dict(orient="records"):
        rows.append(
            {
                "game_date": pd.to_datetime(record["game_date"]).date(),
                "player_id": int(record["player_id"]),
                "team": record["team"],
                "opponent": record.get("opponent"),
                "lineup_slot": record.get("lineup_slot"),
                "hits": record.get("hits"),
                "at_bats": record.get("at_bats"),
                "plate_appearances": record.get("plate_appearances"),
                "had_hit": bool(record.get("had_hit")),
                "hit_streak": int(record.get("hit_streak") or 0),
                "rolling_hit_rate_3": record.get("rolling_hit_rate_3"),
                "rolling_hit_rate_5": record.get("rolling_hit_rate_5"),
                "rolling_hit_rate_7": record.get("rolling_hit_rate_7"),
                "rolling_hit_rate_14": record.get("rolling_hit_rate_14"),
                "rolling_batting_avg_7": record.get("rolling_batting_avg_7"),
                "rolling_batting_avg_14": record.get("rolling_batting_avg_14"),
                "rolling_xwoba_7": record.get("rolling_xwoba_7"),
                "rolling_xwoba_14": record.get("rolling_xwoba_14"),
                "rolling_hard_hit_pct_7": record.get("rolling_hard_hit_pct_7"),
                "rolling_hard_hit_pct_14": record.get("rolling_hard_hit_pct_14"),
            }
        )
    delete_for_date_range("player_trend_daily", start_date, end_date)
    return upsert_rows("player_trend_daily", rows, ["game_date", "player_id"])


def _build_pitcher_trends(start_date, end_date) -> int:
    history_start = pd.Timestamp(start_date) - pd.Timedelta(days=21)
    frame = query_df(
        """
        SELECT
            ps.game_id,
            ps.game_date,
            ps.pitcher_id,
            ps.team,
            ps.side,
            ps.is_probable,
            ps.ip,
            ps.strikeouts,
            ps.walks,
            ps.batters_faced,
            ps.pitch_count,
            ps.whiff_pct,
            ps.csw_pct,
            ps.xwoba_against,
            ps.days_rest,
            CASE WHEN ps.team = g.home_team THEN g.away_team ELSE g.home_team END AS opponent
        FROM pitcher_starts ps
        LEFT JOIN games g
            ON g.game_id = ps.game_id
           AND g.game_date = ps.game_date
        WHERE ps.game_date BETWEEN :history_start AND :end_date
        ORDER BY ps.pitcher_id, ps.game_date, ps.game_id
        """,
        {"history_start": history_start.date(), "end_date": end_date},
    )
    if frame.empty:
        return 0
    frame["game_date"] = pd.to_datetime(frame["game_date"])

    def _decorate(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values(["game_date", "game_id"]).copy()
        group["rolling_strikeouts_3"] = pd.to_numeric(group["strikeouts"], errors="coerce").rolling(3, min_periods=1).mean()
        group["rolling_strikeouts_5"] = pd.to_numeric(group["strikeouts"], errors="coerce").rolling(5, min_periods=1).mean()
        group["rolling_ip_3"] = pd.to_numeric(group["ip"], errors="coerce").rolling(3, min_periods=1).mean()
        group["rolling_ip_5"] = pd.to_numeric(group["ip"], errors="coerce").rolling(5, min_periods=1).mean()
        group["rolling_pitch_count_3"] = pd.to_numeric(group["pitch_count"], errors="coerce").rolling(3, min_periods=1).mean()
        group["rolling_whiff_pct_3"] = pd.to_numeric(group["whiff_pct"], errors="coerce").rolling(3, min_periods=1).mean()
        group["rolling_csw_pct_3"] = pd.to_numeric(group["csw_pct"], errors="coerce").rolling(3, min_periods=1).mean()
        strikeout_sum = pd.to_numeric(group["strikeouts"], errors="coerce").rolling(5, min_periods=1).sum()
        batters_faced = pd.to_numeric(group["batters_faced"], errors="coerce")
        if batters_faced.dropna().empty:
            usage_sum = pd.to_numeric(group["pitch_count"], errors="coerce").rolling(5, min_periods=1).sum().replace(0, np.nan)
        else:
            usage_sum = batters_faced.rolling(5, min_periods=1).sum().replace(0, np.nan)
        group["rolling_k_per_batter_5"] = strikeout_sum / usage_sum
        return group

    trend_frame = _concat_group_results(frame, "pitcher_id", _decorate)
    trend_frame = trend_frame[(trend_frame["game_date"].dt.date >= start_date) & (trend_frame["game_date"].dt.date <= end_date)].copy()
    rows = []
    for record in trend_frame.to_dict(orient="records"):
        rows.append(
            {
                "game_date": pd.to_datetime(record["game_date"]).date(),
                "pitcher_id": int(record["pitcher_id"]),
                "team": record["team"],
                "opponent": record.get("opponent"),
                "is_probable": bool(record.get("is_probable")) if record.get("is_probable") is not None else None,
                "innings_pitched": record.get("ip"),
                "strikeouts": record.get("strikeouts"),
                "walks": record.get("walks"),
                "pitches_thrown": record.get("pitch_count"),
                "whiff_pct": record.get("whiff_pct"),
                "csw_pct": record.get("csw_pct"),
                "xwoba_against": record.get("xwoba_against"),
                "days_rest": record.get("days_rest"),
                "rolling_strikeouts_3": record.get("rolling_strikeouts_3"),
                "rolling_strikeouts_5": record.get("rolling_strikeouts_5"),
                "rolling_ip_3": record.get("rolling_ip_3"),
                "rolling_ip_5": record.get("rolling_ip_5"),
                "rolling_pitch_count_3": record.get("rolling_pitch_count_3"),
                "rolling_whiff_pct_3": record.get("rolling_whiff_pct_3"),
                "rolling_csw_pct_3": record.get("rolling_csw_pct_3"),
                "rolling_k_per_batter_5": record.get("rolling_k_per_batter_5"),
            }
        )
    delete_for_date_range("pitcher_trend_daily", start_date, end_date)
    return upsert_rows("pitcher_trend_daily", rows, ["game_date", "pitcher_id"])


def _composite_entity_id(game_id: Any, subject_id: Any) -> int:
    return int(game_id) * 1000000 + int(subject_id)


def _latest_rows(query: str, params: dict[str, Any]) -> pd.DataFrame:
    frame = query_df(query, params)
    return frame if not frame.empty else pd.DataFrame()


def _fetch_weather_by_game(start_date, end_date) -> dict[int, dict[str, dict[str, Any]]]:
    """Return {game_id: {"forecast": {...}, "observed": {...}}} for the date range."""
    if not table_exists("game_weather"):
        return {}
    frame = query_df(
        """
        SELECT game_id, weather_type, temperature_f, wind_speed_mph, humidity_pct,
               ROW_NUMBER() OVER (PARTITION BY game_id, weather_type ORDER BY snapshot_ts DESC) AS rn
        FROM game_weather
        WHERE game_date BETWEEN :start_date AND :end_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        return {}
    frame = frame[frame["rn"] == 1]
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for _, row in frame.iterrows():
        game_id = int(row["game_id"])
        wt = str(row.get("weather_type") or "forecast")
        # Treat both "archive" and "observed" as observed for grading purposes
        key = "observed" if wt in ("archive", "observed") else "forecast"
        result.setdefault(game_id, {})[key] = {
            "temperature_f": row.get("temperature_f"),
            "wind_speed_mph": row.get("wind_speed_mph"),
            "humidity_pct": row.get("humidity_pct"),
        }
    return result


def _weather_outcome_fields(game_id: int, weather_map: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Return weather columns for a prediction outcome row."""
    game_weather = weather_map.get(game_id, {})
    forecast = game_weather.get("forecast", {})
    observed = game_weather.get("observed", {})
    ft = forecast.get("temperature_f")
    ot = observed.get("temperature_f")
    return {
        "forecast_temperature_f": ft,
        "forecast_wind_speed_mph": forecast.get("wind_speed_mph"),
        "forecast_humidity_pct": forecast.get("humidity_pct"),
        "observed_temperature_f": ot,
        "observed_wind_speed_mph": observed.get("wind_speed_mph"),
        "observed_humidity_pct": observed.get("humidity_pct"),
        "weather_delta_temperature_f": round(float(ot) - float(ft), 1) if ft is not None and ot is not None else None,
    }


def _market_source_priority(value: object) -> int:
    raw = str(value or "").lower()
    if "manual" in raw or "csv" in raw:
        return 2
    if "odds" in raw or "covers" in raw:
        return 1
    return 0


def _select_closing_market_rows(
    frame: pd.DataFrame,
    *,
    preferred_sportsbook_by_game: dict[int, str] | None = None,
) -> dict[int, dict[str, Any]]:
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], errors="coerce", utc=True)
    frame["game_start_ts"] = pd.to_datetime(frame["game_start_ts"], errors="coerce", utc=True)
    frame = frame[frame["snapshot_ts"].notna()].copy()
    frame = frame[frame["game_start_ts"].isna() | (frame["snapshot_ts"] <= frame["game_start_ts"])].copy()
    if frame.empty:
        return {}
    frame["price_complete"] = frame[["over_price", "under_price"]].notna().all(axis=1).astype(int)
    frame["source_priority"] = frame["source_name"].map(_market_source_priority)

    result: dict[int, dict[str, Any]] = {}
    for game_id, group in frame.groupby("game_id"):
        preferred_sportsbook = None
        same_sportsbook = None
        if preferred_sportsbook_by_game is not None:
            raw_preferred = preferred_sportsbook_by_game.get(int(game_id))
            if raw_preferred is not None and pd.notna(raw_preferred) and str(raw_preferred).strip():
                preferred_sportsbook = str(raw_preferred)
                same_sportsbook = False
        chosen_group = group
        if preferred_sportsbook is not None:
            sportsbook_group = group[group["sportsbook"].eq(preferred_sportsbook)].copy()
            if not sportsbook_group.empty:
                chosen_group = sportsbook_group
                same_sportsbook = True
        chosen = chosen_group.sort_values(
            ["snapshot_ts", "price_complete", "source_priority", "sportsbook"],
            ascending=[False, False, False, True],
        ).iloc[0]
        result[int(game_id)] = {
            "sportsbook": chosen.get("sportsbook"),
            "line_value": chosen.get("line_value"),
            "snapshot_ts": chosen.get("snapshot_ts"),
            "same_sportsbook": same_sportsbook,
        }
    return result


def _fetch_closing_market_by_game(
    start_date,
    end_date,
    *,
    market_type: str = "total",
    preferred_sportsbook_by_game: dict[int, str] | None = None,
) -> dict[int, dict[str, Any]]:
    """Return latest pre-start market snapshot per game for CLV review."""
    result: dict[int, dict[str, Any]] = {}
    if table_exists("market_snapshot_versions"):
        frame = query_df(
            """
            SELECT msv.game_id, msv.sportsbook, msv.line_value, msv.snapshot_ts, msv.source_name, msv.over_price, msv.under_price,
                   g.game_start_ts
            FROM market_snapshot_versions msv
            JOIN games g ON g.game_id = msv.game_id AND CAST(g.game_date AS TEXT) = msv.game_date
            WHERE msv.game_date BETWEEN :start_date AND :end_date
              AND msv.market_type = :market_type
              AND msv.line_value IS NOT NULL
            """,
            {"start_date": str(start_date), "end_date": str(end_date), "market_type": market_type},
        )
        result = _select_closing_market_rows(frame, preferred_sportsbook_by_game=preferred_sportsbook_by_game)

    if not table_exists("game_markets"):
        return result

    fallback_frame = query_df(
        """
        SELECT gm.game_id, gm.sportsbook, gm.line_value, gm.snapshot_ts, gm.source_name, gm.over_price, gm.under_price,
               g.game_start_ts
        FROM game_markets gm
        JOIN games g ON g.game_id = gm.game_id AND g.game_date = gm.game_date
        WHERE gm.game_date BETWEEN :start_date AND :end_date
          AND gm.market_type = :market_type
          AND gm.line_value IS NOT NULL
        """,
        {"start_date": start_date, "end_date": end_date, "market_type": market_type},
    )
    fallback_result = _select_closing_market_rows(
        fallback_frame,
        preferred_sportsbook_by_game=preferred_sportsbook_by_game,
    )
    for game_id, record in fallback_result.items():
        result.setdefault(game_id, record)
    return result


def _build_prediction_outcomes(start_date, end_date) -> int:
    rows: list[dict[str, Any]] = []
    weather_map = _fetch_weather_by_game(start_date, end_date)

    totals = _latest_rows(
        """
        WITH ranked AS (
            SELECT p.*, ROW_NUMBER() OVER (
                PARTITION BY p.game_date, p.game_id
                ORDER BY p.prediction_ts DESC, p.created_at DESC, p.model_version DESC, p.model_name DESC
            ) AS row_rank
            FROM predictions_totals p
            WHERE p.game_date BETWEEN :start_date AND :end_date
        )
        SELECT r.*, g.total_runs, g.away_team, g.home_team
               , g.status AS game_status
        FROM ranked r
        LEFT JOIN games g ON g.game_id = r.game_id AND g.game_date = r.game_date
        WHERE r.row_rank = 1
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    preferred_totals_books = {
        int(record["game_id"]): str(record["market_sportsbook"])
        for record in totals.to_dict(orient="records")
        if record.get("market_sportsbook") is not None and pd.notna(record.get("market_sportsbook"))
    }
    totals_closing_market = _fetch_closing_market_by_game(
        start_date,
        end_date,
        market_type="total",
        preferred_sportsbook_by_game=preferred_totals_books,
    )
    for record in totals.to_dict(orient="records"):
        is_final_game = _is_final_game_status(record.get("game_status"))
        actual_total = record.get("total_runs") if is_final_game else None
        market_total = record.get("market_total")
        predicted_total = record.get("predicted_total_runs")
        graded = actual_total is not None
        actual_side = None
        success = None
        beat_market = None
        brier_score = None
        probability = record.get("over_probability")
        opposite_probability = record.get("under_probability")
        recommended_side = None
        closing_market = totals_closing_market.get(int(record["game_id"]), {})
        closing_line = closing_market.get("line_value")
        clv_line_delta = None
        clv_side_value = None
        beat_closing_line = None
        if market_total is not None and predicted_total is not None:
            recommended_side = "over" if predicted_total >= market_total else "under"
        if market_total is not None and closing_line is not None:
            clv_line_delta = float(closing_line) - float(market_total)
            if recommended_side == "over":
                clv_side_value = clv_line_delta
            elif recommended_side == "under":
                clv_side_value = -clv_line_delta
            if clv_side_value is not None:
                beat_closing_line = clv_side_value > 0
        if graded and market_total is not None:
            if actual_total > market_total:
                actual_side = "over"
            elif actual_total < market_total:
                actual_side = "under"
            else:
                actual_side = "push"
            if actual_side != "push" and recommended_side is not None:
                success = recommended_side == actual_side
            if predicted_total is not None:
                beat_market = abs(float(predicted_total) - float(actual_total)) < abs(float(market_total) - float(actual_total))
            if probability is not None and actual_side != "push":
                actual_over = 1.0 if actual_side == "over" else 0.0
                brier_score = (float(probability) - actual_over) ** 2
        rows.append(
            {
                "game_date": pd.to_datetime(record["game_date"]).date(),
                "market": "totals",
                "entity_type": "game",
                "entity_id": int(record["game_id"]),
                "game_id": int(record["game_id"]),
                "player_id": None,
                "pitcher_id": None,
                "team": None,
                "opponent": None,
                "model_name": record["model_name"],
                "model_version": record["model_version"],
                "prediction_ts": record["prediction_ts"],
                "predicted_value": predicted_total,
                "actual_value": actual_total,
                "market_line": market_total,
                "entry_market_sportsbook": record.get("market_sportsbook"),
                "entry_market_snapshot_ts": record.get("market_snapshot_ts"),
                "closing_market_sportsbook": closing_market.get("sportsbook"),
                "closing_market_line": closing_line,
                "closing_market_snapshot_ts": closing_market.get("snapshot_ts"),
                "closing_market_same_sportsbook": closing_market.get("same_sportsbook"),
                "clv_line_delta": clv_line_delta,
                "clv_side_value": clv_side_value,
                "beat_closing_line": beat_closing_line,
                "probability": probability,
                "opposite_probability": opposite_probability,
                "recommended_side": recommended_side,
                "actual_side": actual_side,
                "graded": graded,
                "success": success,
                "beat_market": beat_market,
                "absolute_error": None if not graded or predicted_total is None else abs(float(predicted_total) - float(actual_total)),
                "squared_error": None if not graded or predicted_total is None else (float(predicted_total) - float(actual_total)) ** 2,
                "brier_score": brier_score,
                "meta_payload": {"away_team": record.get("away_team"), "home_team": record.get("home_team")},
                **_weather_outcome_fields(int(record["game_id"]), weather_map),
            }
        )

    hits = _latest_rows(
        """
        WITH ranked AS (
            SELECT p.*, ROW_NUMBER() OVER (
                PARTITION BY p.game_date, p.game_id, p.player_id
                ORDER BY p.prediction_ts DESC, p.created_at DESC, p.model_version DESC, p.model_name DESC
            ) AS row_rank
            FROM predictions_player_hits p
            WHERE p.game_date BETWEEN :start_date AND :end_date
        )
        SELECT r.*, f.opponent, actual.hits AS actual_hits, g.status AS game_status
        FROM ranked r
        LEFT JOIN player_features_hits f
            ON f.game_id = r.game_id
           AND f.player_id = r.player_id
           AND f.game_date = r.game_date
        LEFT JOIN player_game_batting actual
            ON actual.game_id = r.game_id
           AND actual.player_id = r.player_id
        LEFT JOIN games g
            ON g.game_id = r.game_id
           AND g.game_date = r.game_date
        WHERE r.row_rank = 1
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    for record in hits.to_dict(orient="records"):
        is_final_game = _is_final_game_status(record.get("game_status"))
        actual_hits = record.get("actual_hits") if is_final_game else None
        actual_hit = None if actual_hits is None else (1.0 if float(actual_hits) > 0 else 0.0)
        probability = record.get("predicted_hit_probability")
        rows.append(
            {
                "game_date": pd.to_datetime(record["game_date"]).date(),
                "market": "hits",
                "entity_type": "player",
                "entity_id": _composite_entity_id(record["game_id"], record["player_id"]),
                "game_id": int(record["game_id"]),
                "player_id": int(record["player_id"]),
                "pitcher_id": None,
                "team": record["team"],
                "opponent": record.get("opponent"),
                "model_name": record["model_name"],
                "model_version": record["model_version"],
                "prediction_ts": record["prediction_ts"],
                "predicted_value": probability,
                "actual_value": actual_hit,
                "market_line": None,
                "probability": probability,
                "opposite_probability": None,
                "recommended_side": "yes",
                "actual_side": None if actual_hit is None else ("yes" if actual_hit > 0 else "no"),
                "graded": actual_hit is not None,
                "success": None if actual_hit is None else bool(actual_hit > 0),
                "beat_market": None,
                "absolute_error": None if actual_hit is None or probability is None else abs(float(probability) - float(actual_hit)),
                "squared_error": None if actual_hit is None or probability is None else (float(probability) - float(actual_hit)) ** 2,
                "brier_score": None if actual_hit is None or probability is None else (float(probability) - float(actual_hit)) ** 2,
                "meta_payload": {"fair_price": record.get("fair_price"), "market_price": record.get("market_price"), "edge": record.get("edge")},
                **_weather_outcome_fields(int(record["game_id"]), weather_map),
            }
        )

    strikeouts = _latest_rows(
        """
        WITH ranked AS (
            SELECT p.*, ROW_NUMBER() OVER (
                PARTITION BY p.game_date, p.game_id, p.pitcher_id
                ORDER BY p.prediction_ts DESC, p.created_at DESC, p.model_version DESC, p.model_name DESC
            ) AS row_rank
            FROM predictions_pitcher_strikeouts p
            WHERE p.game_date BETWEEN :start_date AND :end_date
        )
        SELECT
            r.*,
            CASE WHEN ps.team = g.home_team THEN g.away_team ELSE g.home_team END AS opponent,
            ps.strikeouts AS actual_strikeouts,
            g.status AS game_status
        FROM ranked r
        LEFT JOIN pitcher_starts ps
            ON ps.game_id = r.game_id
           AND ps.pitcher_id = r.pitcher_id
           AND ps.game_date = r.game_date
        LEFT JOIN games g
            ON g.game_id = r.game_id
           AND g.game_date = r.game_date
        WHERE r.row_rank = 1
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    for record in strikeouts.to_dict(orient="records"):
        is_final_game = _is_final_game_status(record.get("game_status"))
        actual_value = record.get("actual_strikeouts") if is_final_game else None
        market_line = record.get("market_line")
        predicted_value = record.get("predicted_strikeouts")
        graded = actual_value is not None
        recommended_side = None
        actual_side = None
        success = None
        probability = record.get("over_probability")
        opposite_probability = record.get("under_probability")
        if market_line is not None and predicted_value is not None:
            recommended_side = "over" if predicted_value >= market_line else "under"
        if graded and market_line is not None:
            if actual_value > market_line:
                actual_side = "over"
            elif actual_value < market_line:
                actual_side = "under"
            else:
                actual_side = "push"
            if actual_side != "push" and recommended_side is not None:
                success = recommended_side == actual_side
        rows.append(
            {
                "game_date": pd.to_datetime(record["game_date"]).date(),
                "market": "pitcher_strikeouts",
                "entity_type": "pitcher",
                "entity_id": _composite_entity_id(record["game_id"], record["pitcher_id"]),
                "game_id": int(record["game_id"]),
                "player_id": None,
                "pitcher_id": int(record["pitcher_id"]),
                "team": record["team"],
                "opponent": record.get("opponent"),
                "model_name": record["model_name"],
                "model_version": record["model_version"],
                "prediction_ts": record["prediction_ts"],
                "predicted_value": predicted_value,
                "actual_value": actual_value,
                "market_line": market_line,
                "probability": probability,
                "opposite_probability": opposite_probability,
                "recommended_side": recommended_side,
                "actual_side": actual_side,
                "graded": graded,
                "success": success,
                "beat_market": None if not graded or market_line is None else abs(float(predicted_value) - float(actual_value)) < abs(float(market_line) - float(actual_value)),
                "absolute_error": None if not graded or predicted_value is None else abs(float(predicted_value) - float(actual_value)),
                "squared_error": None if not graded or predicted_value is None else (float(predicted_value) - float(actual_value)) ** 2,
                "brier_score": None if probability is None or actual_side in {None, "push"} else (float(probability) - (1.0 if actual_side == "over" else 0.0)) ** 2,
                "meta_payload": {"edge": record.get("edge")},
                **_weather_outcome_fields(int(record["game_id"]), weather_map),
            }
        )

    for row in rows:
        row.setdefault("entry_market_sportsbook", None)
        row.setdefault("entry_market_snapshot_ts", None)
        row.setdefault("closing_market_sportsbook", None)
        row.setdefault("closing_market_line", None)
        row.setdefault("closing_market_snapshot_ts", None)
        row.setdefault("closing_market_same_sportsbook", None)
        row.setdefault("clv_line_delta", None)
        row.setdefault("clv_side_value", None)
        row.setdefault("beat_closing_line", None)

    delete_for_date_range("prediction_outcomes_daily", start_date, end_date, date_column="game_date")
    return upsert_rows(
        "prediction_outcomes_daily",
        rows,
        ["game_date", "market", "entity_type", "entity_id", "model_name", "model_version", "prediction_ts"],
    )


def _build_model_scorecards(start_date, end_date) -> int:
    frame = query_df(
        """
        SELECT *
        FROM prediction_outcomes_daily
        WHERE game_date BETWEEN :start_date AND :end_date
        ORDER BY game_date, market, model_name, model_version
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        return 0
    rows = []
    for (game_date, market, model_name, model_version), group in frame.groupby(["game_date", "market", "model_name", "model_version"]):
        graded = group[group["graded"].eq(True)]
        success_sample = graded[graded["success"].notna()]
        beat_market_sample = graded[graded["beat_market"].notna()]
        clv_sample = group[group["clv_side_value"].notna()] if "clv_side_value" in group.columns else group.iloc[0:0]
        rows.append(
            {
                "score_date": pd.to_datetime(game_date).date(),
                "market": market,
                "model_name": model_name,
                "model_version": model_version,
                "predictions_count": int(len(group)),
                "graded_count": int(len(graded)),
                "success_count": int(success_sample["success"].eq(True).sum()),
                "pending_count": int(len(group) - len(graded)),
                "success_rate": None if success_sample.empty else float(success_sample["success"].mean()),
                "avg_absolute_error": None if graded["absolute_error"].dropna().empty else float(graded["absolute_error"].dropna().mean()),
                "avg_bias": None if graded[["predicted_value", "actual_value"]].dropna().empty else float((graded["predicted_value"].astype(float) - graded["actual_value"].astype(float)).mean()),
                "brier_score": None if graded["brier_score"].dropna().empty else float(graded["brier_score"].dropna().mean()),
                "beat_market_rate": None if beat_market_sample.empty else float(beat_market_sample["beat_market"].mean()),
                "clv_count": int(len(clv_sample)),
                "avg_clv_side_value": None if clv_sample.empty else float(clv_sample["clv_side_value"].astype(float).mean()),
                "positive_clv_rate": None if clv_sample.empty else float(clv_sample["beat_closing_line"].eq(True).mean()),
            }
        )
    delete_for_date_range("model_scorecards_daily", start_date, end_date, date_column="score_date")
    return upsert_rows("model_scorecards_daily", rows, ["score_date", "market", "model_name", "model_version"])


CALIBRATION_BINS = [
    (0.00, 0.20, "0-20%"),
    (0.20, 0.35, "20-35%"),
    (0.35, 0.50, "35-50%"),
    (0.50, 0.65, "50-65%"),
    (0.65, 0.80, "65-80%"),
    (0.80, 1.01, "80-100%"),
]


def _build_calibration_bins(start_date, end_date) -> int:
    """Build calibration bins from graded prediction outcomes."""
    if not table_exists("prediction_calibration_bins"):
        return 0
    frame = query_df(
        """
        SELECT game_date, market, model_name, model_version, probability, success, brier_score
        FROM prediction_outcomes_daily
        WHERE game_date BETWEEN :start_date AND :end_date
          AND graded = TRUE
          AND probability IS NOT NULL
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        return 0

    rows: list[dict[str, Any]] = []
    for (score_date, market, model_name, model_version), group in frame.groupby(
        ["game_date", "market", "model_name", "model_version"]
    ):
        for lower, upper, label in CALIBRATION_BINS:
            mask = (group["probability"] >= lower) & (group["probability"] < upper)
            bin_df = group[mask]
            if bin_df.empty:
                continue
            count = len(bin_df)
            success_series = bin_df["success"].dropna()
            actual_hit_rate = float(success_series.mean()) if len(success_series) > 0 else None
            mean_prob = float(bin_df["probability"].mean())
            brier_sum = float(bin_df["brier_score"].dropna().sum()) if bin_df["brier_score"].notna().any() else None
            rows.append({
                "score_date": pd.to_datetime(score_date).date() if not isinstance(score_date, str) else score_date,
                "market": market,
                "model_name": model_name,
                "model_version": model_version,
                "bin_lower": lower,
                "bin_upper": upper,
                "bin_label": label,
                "count": count,
                "actual_hit_rate": round(actual_hit_rate, 4) if actual_hit_rate is not None else None,
                "mean_predicted_prob": round(mean_prob, 4),
                "brier_score_sum": round(brier_sum, 6) if brier_sum is not None else None,
            })
    if not rows:
        return 0
    return upsert_rows("prediction_calibration_bins", rows, ["score_date", "market", "model_name", "bin_label"])


def _build_recommendation_history(start_date, end_date) -> int:
    """Write recommendation history from prediction outcomes."""
    if not table_exists("recommendation_history"):
        return 0
    frame = query_df(
        """
        SELECT game_date, game_id, market, entity_type, entity_id, player_id, team,
               model_name, model_version,
               recommended_side, probability, market_line, predicted_value,
               actual_value, actual_side, graded, success,
               entry_market_sportsbook, entry_market_snapshot_ts,
               closing_market_sportsbook, closing_market_line,
               closing_market_same_sportsbook,
               clv_line_delta, clv_side_value, beat_closing_line,
               meta_payload
        FROM prediction_outcomes_daily
        WHERE game_date BETWEEN :start_date AND :end_date
          AND recommended_side IS NOT NULL
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        return 0

    rows: list[dict[str, Any]] = []
    for _, record in frame.iterrows():
        meta = record.get("meta_payload") or {}
        if isinstance(meta, str):
            import json as _json
            try:
                meta = _json.loads(meta)
            except Exception:
                meta = {}
        predicted = record.get("predicted_value")
        market_line = record.get("market_line")
        edge = None
        if predicted is not None and market_line is not None:
            try:
                edge = round(abs(float(predicted) - float(market_line)), 4)
            except (ValueError, TypeError):
                pass
        rows.append({
            "game_date": pd.to_datetime(record["game_date"]).date() if not isinstance(record["game_date"], str) else record["game_date"],
            "game_id": int(record["game_id"]),
            "market": record["market"],
            "entity_type": record.get("entity_type", "game"),
            "entity_id": int(record["entity_id"]) if pd.notna(record.get("entity_id")) else None,
            "player_id": int(record["player_id"]) if pd.notna(record.get("player_id")) else None,
            "team": record.get("team"),
            "away_team": meta.get("away_team"),
            "home_team": meta.get("home_team"),
            "model_name": record["model_name"],
            "model_version": record.get("model_version"),
            "recommended_side": record["recommended_side"],
            "probability": record.get("probability"),
            "market_line": market_line,
            "entry_market_sportsbook": record.get("entry_market_sportsbook"),
            "entry_market_snapshot_ts": record.get("entry_market_snapshot_ts"),
            "closing_market_sportsbook": record.get("closing_market_sportsbook"),
            "closing_market_line": record.get("closing_market_line"),
            "closing_market_same_sportsbook": record.get("closing_market_same_sportsbook"),
            "clv_line_delta": record.get("clv_line_delta"),
            "clv_side_value": record.get("clv_side_value"),
            "beat_closing_line": record.get("beat_closing_line"),
            "predicted_value": predicted,
            "actual_value": record.get("actual_value"),
            "actual_side": record.get("actual_side"),
            "graded": bool(record.get("graded")),
            "success": record.get("success"),
            "edge": edge,
        })
    if not rows:
        return 0
    return upsert_rows("recommendation_history", rows, ["game_id", "market", "entity_id", "model_name"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build trend tables, prediction outcomes, and model scorecards")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    player_rows = _build_player_trends(start_date, end_date)
    pitcher_rows = _build_pitcher_trends(start_date, end_date)
    outcome_rows = _build_prediction_outcomes(start_date, end_date)
    scorecard_rows = _build_model_scorecards(start_date, end_date)
    calibration_rows = _build_calibration_bins(start_date, end_date)
    recommendation_rows = _build_recommendation_history(start_date, end_date)
    log.info(
        "Built product surfaces for %s to %s -> player trends %s, pitcher trends %s, "
        "outcomes %s, scorecards %s, calibration bins %s, recommendations %s",
        start_date,
        end_date,
        player_rows,
        pitcher_rows,
        outcome_rows,
        scorecard_rows,
        calibration_rows,
        recommendation_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())