"""Career / game-log batter vs pitcher stats (shared by API and HR slugger ranking)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.utils.db import query_df, table_exists


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    cleaned = frame.copy().astype(object)
    cleaned = cleaned.where(pd.notnull(cleaned), None)
    return cleaned.to_dict(orient="records")


def fetch_batter_vs_pitcher_map(
    target_date: date,
    matchups: list[tuple[int, int]],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Return career BvP stats for each (batter_id, pitcher_id) pair.

    First checks matchup_splits (StatMuse-sourced career data), then
    computes from game logs as a fallback for any pairs not found.
    """
    if not matchups:
        return {}

    result: dict[tuple[int, int], dict[str, Any]] = {}

    if table_exists("matchup_splits"):
        params: dict[str, Any] = {}
        pair_clauses = []
        for idx, (batter_id, pitcher_id) in enumerate(matchups):
            params[f"b_{idx}"] = int(batter_id)
            params[f"p_{idx}"] = int(pitcher_id)
            pair_clauses.append(f"(ms.player_id = :b_{idx} AND ms.opponent_id = :p_{idx})")
        if pair_clauses:
            where_clause = " OR ".join(pair_clauses)
            try:
                frame = query_df(
                    f"""
                    SELECT ms.player_id AS batter_id,
                           ms.opponent_id AS pitcher_id,
                           ms.at_bats, ms.hits, ms.home_runs,
                           ms.walks, ms.strikeouts, ms.rbi,
                           ms.doubles, ms.triples,
                           ms.batting_avg, ms.obp, ms.slg, ms.ops
                    FROM matchup_splits ms
                    WHERE ms.split_type = 'bvp'
                      AND ms.season = 0
                      AND ({where_clause})
                    """,
                    params,
                )
            except Exception:
                frame = pd.DataFrame()
            if not frame.empty:
                for row in _frame_records(frame):
                    key = (int(row["batter_id"]), int(row["pitcher_id"]))
                    at_bats = int(row.get("at_bats") or 0)
                    if at_bats == 0:
                        continue
                    result[key] = {
                        "at_bats": at_bats,
                        "hits": int(row.get("hits") or 0),
                        "home_runs": int(row.get("home_runs") or 0),
                        "walks": int(row.get("walks") or 0),
                        "strikeouts": int(row.get("strikeouts") or 0),
                        "rbi": int(row.get("rbi") or 0),
                        "doubles": int(row.get("doubles") or 0),
                        "triples": int(row.get("triples") or 0),
                        "batting_avg": row.get("batting_avg"),
                        "obp": row.get("obp"),
                        "slg": row.get("slg"),
                        "ops": row.get("ops"),
                        "source": "career",
                    }

    missing = [pair for pair in matchups if pair not in result]
    if missing and table_exists("player_game_batting") and table_exists("pitcher_starts"):
        params2: dict[str, Any] = {"target_date": target_date}
        pair_clauses2 = []
        for idx, (batter_id, pitcher_id) in enumerate(missing):
            params2[f"mb_{idx}"] = int(batter_id)
            params2[f"mp_{idx}"] = int(pitcher_id)
            pair_clauses2.append(f"(b.player_id = :mb_{idx} AND ps.pitcher_id = :mp_{idx})")
        where_clause2 = " OR ".join(pair_clauses2)
        try:
            frame2 = query_df(
                f"""
                SELECT
                    b.player_id AS batter_id,
                    ps.pitcher_id,
                    SUM(b.hits) AS hits,
                    SUM(b.at_bats) AS at_bats,
                    SUM(b.home_runs) AS home_runs,
                    SUM(b.walks) AS walks,
                    SUM(b.strikeouts) AS strikeouts,
                    SUM(b.rbi) AS rbi,
                    SUM(b.doubles) AS doubles,
                    SUM(b.triples) AS triples,
                    CAST(SUM(b.hits) AS REAL) / NULLIF(SUM(b.at_bats), 0) AS batting_avg
                FROM player_game_batting b
                INNER JOIN pitcher_starts ps
                    ON ps.game_id = b.game_id
                   AND ps.team = b.opponent
                WHERE b.game_date < :target_date
                  AND ({where_clause2})
                GROUP BY b.player_id, ps.pitcher_id
                """,
                params2,
            )
        except Exception:
            frame2 = pd.DataFrame()
        if not frame2.empty:
            for row in _frame_records(frame2):
                key = (int(row["batter_id"]), int(row["pitcher_id"]))
                at_bats = int(row.get("at_bats") or 0)
                if at_bats == 0:
                    continue
                result[key] = {
                    "at_bats": at_bats,
                    "hits": int(row.get("hits") or 0),
                    "home_runs": int(row.get("home_runs") or 0),
                    "walks": int(row.get("walks") or 0),
                    "strikeouts": int(row.get("strikeouts") or 0),
                    "rbi": int(row.get("rbi") or 0),
                    "doubles": int(row.get("doubles") or 0),
                    "triples": int(row.get("triples") or 0),
                    "batting_avg": row.get("batting_avg"),
                    "obp": None,
                    "slg": None,
                    "ops": None,
                    "source": "game_logs",
                }

    return result
