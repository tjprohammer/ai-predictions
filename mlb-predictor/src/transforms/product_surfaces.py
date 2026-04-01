from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import delete_for_date_range, query_df, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)


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
        ORDER BY pitcher_id, game_date, game_id
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


def _build_prediction_outcomes(start_date, end_date) -> int:
    rows: list[dict[str, Any]] = []

    totals = _latest_rows(
        """
        WITH ranked AS (
            SELECT p.*, ROW_NUMBER() OVER (PARTITION BY p.game_date, p.game_id, p.model_name, p.model_version ORDER BY p.prediction_ts DESC) AS row_rank
            FROM predictions_totals p
            WHERE p.game_date BETWEEN :start_date AND :end_date
        )
        SELECT r.*, g.total_runs, g.away_team, g.home_team
        FROM ranked r
        LEFT JOIN games g ON g.game_id = r.game_id AND g.game_date = r.game_date
        WHERE r.row_rank = 1
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    for record in totals.to_dict(orient="records"):
        actual_total = record.get("total_runs")
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
        if market_total is not None and predicted_total is not None:
            recommended_side = "over" if predicted_total >= market_total else "under"
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
                "game_date": record["game_date"],
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
            }
        )

    hits = _latest_rows(
        """
        WITH ranked AS (
            SELECT p.*, ROW_NUMBER() OVER (PARTITION BY p.game_date, p.game_id, p.player_id, p.model_name, p.model_version ORDER BY p.prediction_ts DESC) AS row_rank
            FROM predictions_player_hits p
            WHERE p.game_date BETWEEN :start_date AND :end_date
        )
        SELECT r.*, f.opponent, actual.hits AS actual_hits
        FROM ranked r
        LEFT JOIN player_features_hits f
            ON f.game_id = r.game_id
           AND f.player_id = r.player_id
           AND f.game_date = r.game_date
        LEFT JOIN player_game_batting actual
            ON actual.game_id = r.game_id
           AND actual.player_id = r.player_id
        WHERE r.row_rank = 1
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    for record in hits.to_dict(orient="records"):
        actual_hit = None if record.get("actual_hits") is None else (1.0 if float(record.get("actual_hits")) > 0 else 0.0)
        probability = record.get("predicted_hit_probability")
        rows.append(
            {
                "game_date": record["game_date"],
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
            }
        )

    strikeouts = _latest_rows(
        """
        WITH ranked AS (
            SELECT p.*, ROW_NUMBER() OVER (PARTITION BY p.game_date, p.game_id, p.pitcher_id, p.model_name, p.model_version ORDER BY p.prediction_ts DESC) AS row_rank
            FROM predictions_pitcher_strikeouts p
            WHERE p.game_date BETWEEN :start_date AND :end_date
        )
        SELECT
            r.*,
            CASE WHEN ps.team = g.home_team THEN g.away_team ELSE g.home_team END AS opponent,
            ps.strikeouts AS actual_strikeouts
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
        actual_value = record.get("actual_strikeouts")
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
                "game_date": record["game_date"],
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
            }
        )

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
        rows.append(
            {
                "score_date": game_date,
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
            }
        )
    delete_for_date_range("model_scorecards_daily", start_date, end_date, date_column="score_date")
    return upsert_rows("model_scorecards_daily", rows, ["score_date", "market", "model_name", "model_version"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build trend tables, prediction outcomes, and model scorecards")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    player_rows = _build_player_trends(start_date, end_date)
    pitcher_rows = _build_pitcher_trends(start_date, end_date)
    outcome_rows = _build_prediction_outcomes(start_date, end_date)
    scorecard_rows = _build_model_scorecards(start_date, end_date)
    log.info(
        "Built product surfaces for %s to %s -> player trends %s, pitcher trends %s, outcomes %s, scorecards %s",
        start_date,
        end_date,
        player_rows,
        pitcher_rows,
        outcome_rows,
        scorecard_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())