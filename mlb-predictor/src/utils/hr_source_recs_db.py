"""Load HR prediction rows shaped for slugger cards / Daily Results (shared SQL)."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd

from src.utils.bvp_lookup import fetch_batter_vs_pitcher_map
from src.utils.db import query_df, table_exists


def load_hr_source_recs_for_date(target_date: date, min_probability: float = 0.0) -> list[dict[str, Any]]:
    """Latest HR prediction per batter; same shape as ``build_player_hr_board_card`` / slugger selection."""
    if not table_exists("predictions_player_hr"):
        return []
    try:
        slugger_frame = query_df(
        """
        WITH ranked AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY p.game_id, p.player_id
                    ORDER BY p.prediction_ts DESC
                ) AS row_rank
            FROM predictions_player_hr p
            WHERE p.game_date = :target_date
        )
        SELECT
            r.game_id,
            r.player_id,
            r.team,
            r.predicted_hr_probability,
            r.fair_price,
            r.market_price,
            r.edge,
            r.reasoning_json,
            g.away_team,
            g.home_team,
            g.game_start_ts,
            g.status AS game_status,
            g.game_date AS game_date,
            COALESCE(dp.full_name, CAST(r.player_id AS TEXT)) AS player_name,
            CASE
                WHEN r.team = g.home_team THEN g.away_team
                WHEN r.team = g.away_team THEN g.home_team
                ELSE NULL
            END AS opponent,
            bat.home_runs AS actual_home_runs,
            pf.hr_per_pa_blended,
            pf.projected_plate_appearances,
            pf.xwoba_14,
            pf.hard_hit_pct_14,
            pf.streak_len_capped,
            pf.season_prior_hr_per_pa,
            pf.hr_game_rate_30,
            pf.park_hr_factor,
            pf.opposing_starter_hr_per_9,
            pf.opposing_starter_barrel_pct,
            ph.hit_rate_7,
            ph.hit_rate_14,
            ph.hit_rate_blended,
            ps_opp.pitcher_id AS opposing_pitcher_id
        FROM ranked r
        INNER JOIN games g ON g.game_id = r.game_id AND g.game_date = r.game_date
        LEFT JOIN dim_players dp ON dp.player_id = r.player_id
        LEFT JOIN player_game_batting bat
            ON bat.game_id = r.game_id
           AND bat.player_id = r.player_id
           AND bat.game_date = r.game_date
        LEFT JOIN (
            SELECT
                ps_inner.game_id,
                ps_inner.team,
                ps_inner.pitcher_id,
                ROW_NUMBER() OVER (
                    PARTITION BY ps_inner.game_id, ps_inner.team
                    ORDER BY
                        CASE
                            WHEN ps_inner.ip IS NOT NULL
                              OR ps_inner.pitch_count IS NOT NULL
                              OR ps_inner.strikeouts IS NOT NULL THEN 0
                            ELSE 1
                        END,
                        CASE WHEN COALESCE(ps_inner.is_probable, FALSE) THEN 1 ELSE 0 END,
                        ps_inner.pitcher_id
                ) AS pst_rn
            FROM pitcher_starts ps_inner
            WHERE ps_inner.game_date = :target_date
        ) ps_opp ON ps_opp.game_id = r.game_id
            AND ps_opp.team = CASE
                WHEN r.team = g.home_team THEN g.away_team
                WHEN r.team = g.away_team THEN g.home_team
                ELSE NULL
            END
            AND ps_opp.pst_rn = 1
        LEFT JOIN (
            SELECT
                fr.game_id,
                fr.player_id,
                CAST(NULLIF(fr.feature_payload ->> 'hr_per_pa_blended', '') AS DOUBLE PRECISION) AS hr_per_pa_blended,
                CAST(NULLIF(fr.feature_payload ->> 'projected_plate_appearances', '') AS DOUBLE PRECISION) AS projected_plate_appearances,
                CAST(NULLIF(fr.feature_payload ->> 'xwoba_14', '') AS DOUBLE PRECISION) AS xwoba_14,
                CAST(NULLIF(fr.feature_payload ->> 'hard_hit_pct_14', '') AS DOUBLE PRECISION) AS hard_hit_pct_14,
                CAST(NULLIF(fr.feature_payload ->> 'streak_len_capped', '') AS DOUBLE PRECISION) AS streak_len_capped,
                CAST(NULLIF(fr.feature_payload ->> 'season_prior_hr_per_pa', '') AS DOUBLE PRECISION) AS season_prior_hr_per_pa,
                CAST(NULLIF(fr.feature_payload ->> 'hr_game_rate_30', '') AS DOUBLE PRECISION) AS hr_game_rate_30,
                CAST(NULLIF(fr.feature_payload ->> 'park_hr_factor', '') AS DOUBLE PRECISION) AS park_hr_factor,
                CAST(NULLIF(fr.feature_payload ->> 'opposing_starter_hr_per_9', '') AS DOUBLE PRECISION) AS opposing_starter_hr_per_9,
                CAST(NULLIF(fr.feature_payload ->> 'opposing_starter_barrel_pct', '') AS DOUBLE PRECISION) AS opposing_starter_barrel_pct
            FROM (
                SELECT
                    f.game_id,
                    f.player_id,
                    f.feature_payload,
                    ROW_NUMBER() OVER (
                        PARTITION BY f.game_id, f.player_id
                        ORDER BY f.prediction_ts DESC
                    ) AS frn
                FROM player_features_hr f
                WHERE f.game_date = :target_date
            ) fr
            WHERE fr.frn = 1
        ) pf ON pf.game_id = r.game_id AND pf.player_id = r.player_id
        LEFT JOIN (
            SELECT
                hf.game_id,
                hf.player_id,
                CAST(NULLIF(hf.feature_payload ->> 'hit_rate_7', '') AS DOUBLE PRECISION) AS hit_rate_7,
                CAST(NULLIF(hf.feature_payload ->> 'hit_rate_14', '') AS DOUBLE PRECISION) AS hit_rate_14,
                CAST(NULLIF(hf.feature_payload ->> 'hit_rate_blended', '') AS DOUBLE PRECISION) AS hit_rate_blended
            FROM (
                SELECT
                    h.game_id,
                    h.player_id,
                    h.feature_payload,
                    ROW_NUMBER() OVER (
                        PARTITION BY h.game_id, h.player_id
                        ORDER BY h.prediction_ts DESC
                    ) AS hfn
                FROM player_features_hits h
                WHERE h.game_date = :target_date
            ) hf
            WHERE hf.hfn = 1
        ) ph ON ph.game_id = r.game_id AND ph.player_id = r.player_id
        WHERE r.row_rank = 1
          AND r.predicted_hr_probability >= :min_probability
        """,
            {"target_date": target_date, "min_probability": min_probability},
        )
    except Exception:
        return []
    if slugger_frame.empty:
        return []
    cleaned = slugger_frame.copy().astype(object)
    cleaned = cleaned.where(pd.notnull(cleaned), None)
    out: list[dict[str, Any]] = []
    for srow in cleaned.to_dict(orient="records"):
        reasoning_raw = srow.get("reasoning_json") or "[]"
        try:
            hr_reasoning = json.loads(reasoning_raw) if isinstance(reasoning_raw, str) else []
        except json.JSONDecodeError:
            hr_reasoning = []
        out.append(
            {
                "game_id": srow.get("game_id"),
                "player_id": srow.get("player_id"),
                "player_name": srow.get("player_name"),
                "team": srow.get("team"),
                "away_team": srow.get("away_team"),
                "home_team": srow.get("home_team"),
                "predicted_hr_probability": srow.get("predicted_hr_probability"),
                "fair_price": srow.get("fair_price"),
                "market_price": srow.get("market_price"),
                "edge": srow.get("edge"),
                "is_confirmed_lineup": False,
                "has_lineup_snapshot": True,
                "hr_reasoning": hr_reasoning,
                "game_start_ts": srow.get("game_start_ts"),
                "game_status": srow.get("game_status"),
                "game_date": srow.get("game_date"),
                "opponent": srow.get("opponent"),
                "actual_home_runs": srow.get("actual_home_runs"),
                "hr_per_pa_blended": srow.get("hr_per_pa_blended"),
                "projected_plate_appearances": srow.get("projected_plate_appearances"),
                "xwoba_14": srow.get("xwoba_14"),
                "hard_hit_pct_14": srow.get("hard_hit_pct_14"),
                "streak_len_capped": srow.get("streak_len_capped"),
                "season_prior_hr_per_pa": srow.get("season_prior_hr_per_pa"),
                "hr_game_rate_30": srow.get("hr_game_rate_30"),
                "hit_rate_7": srow.get("hit_rate_7"),
                "hit_rate_14": srow.get("hit_rate_14"),
                "hit_rate_blended": srow.get("hit_rate_blended"),
                "park_hr_factor": srow.get("park_hr_factor"),
                "opposing_starter_hr_per_9": srow.get("opposing_starter_hr_per_9"),
                "opposing_starter_barrel_pct": srow.get("opposing_starter_barrel_pct"),
                "opposing_pitcher_id": srow.get("opposing_pitcher_id"),
                "bvp_ab": None,
                "bvp_ops": None,
                "bvp_home_runs": None,
            }
        )
    pairs: list[tuple[int, int]] = []
    for row in out:
        bid = row.get("player_id")
        pid = row.get("opposing_pitcher_id")
        if bid is not None and pid is not None:
            pairs.append((int(bid), int(pid)))
    if pairs:
        bvp_map = fetch_batter_vs_pitcher_map(target_date, pairs)
        for row in out:
            bid = row.get("player_id")
            pid = row.get("opposing_pitcher_id")
            if bid is None or pid is None:
                continue
            st = bvp_map.get((int(bid), int(pid)))
            if not st:
                continue
            ab = int(st.get("at_bats") or 0)
            row["bvp_ab"] = ab
            row["bvp_home_runs"] = int(st.get("home_runs") or 0)
            ops = st.get("ops")
            if ops is not None:
                try:
                    row["bvp_ops"] = float(ops)
                except (TypeError, ValueError):
                    row["bvp_ops"] = None
            if row["bvp_ops"] is None:
                obp = st.get("obp")
                slg = st.get("slg")
                if obp is not None and slg is not None:
                    try:
                        row["bvp_ops"] = float(obp) + float(slg)
                    except (TypeError, ValueError):
                        pass
    return out
