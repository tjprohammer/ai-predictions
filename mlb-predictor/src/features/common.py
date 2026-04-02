from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.contracts import (
    FIRST5_TOTALS_META_COLUMNS,
    FIRST5_TOTALS_TARGET_COLUMN,
    HITS_META_COLUMNS,
    HITS_TARGET_COLUMN,
    TOTALS_META_COLUMNS,
    TOTALS_TARGET_COLUMN,
)
from src.features.priors import blend_with_prior
from src.utils.db import upsert_rows
from src.utils.settings import get_settings


def _safe_mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def safe_rate(numerator: float, denominator: float) -> float | None:
    if denominator in (0, None) or pd.isna(denominator):
        return None
    return float(numerator) / float(denominator)


def weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float | None:
    if value_col not in frame.columns or weight_col not in frame.columns:
        return None
    valid = frame[[value_col, weight_col]].dropna()
    if valid.empty:
        return None
    weights = valid[weight_col].astype(float)
    if float(weights.sum()) <= 0:
        return float(valid[value_col].astype(float).mean())
    return float(np.average(valid[value_col].astype(float), weights=weights))


def baseball_ip_to_outs(value: float | int | None) -> int:
    if value is None or pd.isna(value):
        return 0
    whole = int(float(value))
    tenths = int(round((float(value) - whole) * 10))
    return whole * 3 + tenths


def outs_to_baseball_ip(outs: int) -> float:
    whole, remainder = divmod(int(outs), 3)
    return float(f"{whole}.{remainder}")


def build_team_priors(team_offense: pd.DataFrame, prior_season: int) -> pd.DataFrame:
    prior = team_offense[team_offense["season"] == prior_season].copy()
    if prior.empty:
        return pd.DataFrame()
    grouped = prior.groupby("team", as_index=True).agg(
        prior_runs_rate=("runs", "mean"),
        prior_hits_rate=("hits", "mean"),
        prior_xwoba=("xwoba", "mean"),
        prior_iso=("iso", "mean"),
        prior_bb_pct=("bb_pct", "mean"),
        prior_k_pct=("k_pct", "mean"),
    )
    return grouped


def offense_snapshot(
    team: str,
    game_date: date,
    team_offense: pd.DataFrame,
    team_priors: pd.DataFrame,
    full_weight_games: int = 30,
    prior_blend_mode: str = "standard",
    prior_weight_multiplier: float = 1.0,
) -> dict[str, float | None]:
    history = team_offense.copy()
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
    history = history[(history["team"] == team) & (history["game_date"].dt.date < game_date)].copy()
    history = history.sort_values("game_date").tail(full_weight_games)
    sample_size = len(history)
    prior = team_priors.loc[team] if team in team_priors.index else {}
    return {
        "runs_rate": blend_with_prior(
            _safe_mean(history["runs"]),
            prior.get("prior_runs_rate") if isinstance(prior, pd.Series) else None,
            sample_size,
            full_weight_games,
            mode=prior_blend_mode,
            prior_weight_multiplier=prior_weight_multiplier,
        ),
        "hits_rate": blend_with_prior(
            _safe_mean(history["hits"]),
            prior.get("prior_hits_rate") if isinstance(prior, pd.Series) else None,
            sample_size,
            full_weight_games,
            mode=prior_blend_mode,
            prior_weight_multiplier=prior_weight_multiplier,
        ),
        "xwoba": blend_with_prior(
            _safe_mean(history["xwoba"]),
            prior.get("prior_xwoba") if isinstance(prior, pd.Series) else None,
            sample_size,
            full_weight_games,
            mode=prior_blend_mode,
            prior_weight_multiplier=prior_weight_multiplier,
        ),
        "iso": blend_with_prior(
            _safe_mean(history["iso"]),
            prior.get("prior_iso") if isinstance(prior, pd.Series) else None,
            sample_size,
            full_weight_games,
            mode=prior_blend_mode,
            prior_weight_multiplier=prior_weight_multiplier,
        ),
        "bb_pct": blend_with_prior(
            _safe_mean(history["bb_pct"]),
            prior.get("prior_bb_pct") if isinstance(prior, pd.Series) else None,
            sample_size,
            full_weight_games,
            mode=prior_blend_mode,
            prior_weight_multiplier=prior_weight_multiplier,
        ),
        "k_pct": blend_with_prior(
            _safe_mean(history["k_pct"]),
            prior.get("prior_k_pct") if isinstance(prior, pd.Series) else None,
            sample_size,
            full_weight_games,
            mode=prior_blend_mode,
            prior_weight_multiplier=prior_weight_multiplier,
        ),
    }


def build_pitcher_priors(pitcher_starts: pd.DataFrame, prior_season: int) -> pd.DataFrame:
    prior = pitcher_starts[pitcher_starts["game_date"].dt.year == prior_season].copy()
    if prior.empty:
        return pd.DataFrame()
    return prior.groupby("pitcher_id", as_index=True).agg(
        prior_xwoba=("xwoba_against", "mean"),
        prior_csw=("csw_pct", "mean"),
        prior_avg_fb_velo=("avg_fb_velo", "mean"),
    )


def pitcher_snapshot(
    pitcher_id: int | None,
    game_date: date,
    pitcher_starts: pd.DataFrame,
    pitcher_priors: pd.DataFrame,
    full_weight_starts: int = 10,
    prior_blend_mode: str = "standard",
    prior_weight_multiplier: float = 1.0,
) -> dict[str, float | None]:
    if pitcher_id is None:
        return {"xwoba": None, "csw": None, "avg_fb_velo": None, "days_rest": None}
    history = pitcher_starts[
        (pitcher_starts["pitcher_id"] == pitcher_id) & (pitcher_starts["game_date"].dt.date < game_date)
    ].copy()
    history = history.sort_values("game_date")
    sample = history.tail(full_weight_starts)
    prior = pitcher_priors.loc[pitcher_id] if pitcher_id in pitcher_priors.index else {}
    last_game_date = history["game_date"].max() if not history.empty else None
    last_game_date = last_game_date.date() if pd.notna(last_game_date) else None
    sample_size = len(sample)
    return {
        "xwoba": blend_with_prior(
            _safe_mean(sample["xwoba_against"]),
            prior.get("prior_xwoba") if isinstance(prior, pd.Series) else None,
            sample_size,
            full_weight_starts,
            mode=prior_blend_mode,
            prior_weight_multiplier=prior_weight_multiplier,
        ),
        "csw": blend_with_prior(
            _safe_mean(sample["csw_pct"]),
            prior.get("prior_csw") if isinstance(prior, pd.Series) else None,
            sample_size,
            full_weight_starts,
            mode=prior_blend_mode,
            prior_weight_multiplier=prior_weight_multiplier,
        ),
        "avg_fb_velo": blend_with_prior(
            _safe_mean(sample["avg_fb_velo"]),
            prior.get("prior_avg_fb_velo") if isinstance(prior, pd.Series) else None,
            sample_size,
            full_weight_starts,
            mode=prior_blend_mode,
            prior_weight_multiplier=prior_weight_multiplier,
        ),
        "days_rest": None if last_game_date is None else (game_date - last_game_date).days,
    }


def build_hitter_priors(player_batting: pd.DataFrame, prior_season: int) -> pd.DataFrame:
    prior = player_batting[player_batting["game_date"].dt.year == prior_season].copy()
    if prior.empty:
        return pd.DataFrame()
    prior["had_hit"] = (prior["hits"] > 0).astype(float)
    return prior.groupby("player_id", as_index=True).agg(
        prior_hit_rate=("had_hit", "mean"),
        prior_xba=("xba", "mean"),
        prior_xwoba=("xwoba", "mean"),
        prior_hard_hit_pct=("hard_hit_pct", "mean"),
        prior_k_pct=("strikeouts", lambda value: float(value.sum()) / max(float(prior.loc[value.index, "plate_appearances"].sum()), 1.0)),
    )


def capped_hit_streak(history: pd.DataFrame) -> int:
    if history.empty:
        return 0
    streak = 0
    for had_hit in history.sort_values("game_date", ascending=False)["hits"].fillna(0).astype(int).tolist():
        if had_hit > 0:
            streak += 1
            if streak >= 5:
                return 5
        else:
            break
    return streak


def _assign_lineup_slots(slot_estimates: list[float | None], max_players: int) -> list[int]:
    available_slots = list(range(1, max_players + 1))
    assigned: list[int] = []
    for estimate in slot_estimates:
        if estimate is not None and not pd.isna(estimate):
            preferred = sorted(available_slots, key=lambda slot: (abs(slot - float(estimate)), slot))[0]
        else:
            preferred = available_slots[0]
        assigned.append(preferred)
        available_slots.remove(preferred)
    return assigned


def infer_lineup_from_history(
    team: str,
    game_date: date,
    player_batting: pd.DataFrame,
    players: pd.DataFrame | None = None,
    lookback_days: int = 21,
    max_players: int = 9,
) -> pd.DataFrame:
    columns = ["team", "player_id", "player_name", "lineup_slot", "is_confirmed"]
    if player_batting.empty:
        return pd.DataFrame(columns=columns)

    history = player_batting.copy()
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
    history = history[(history["team"] == team) & (history["game_date"].dt.date < game_date)].copy()
    if history.empty:
        return pd.DataFrame(columns=columns)

    lookback_start = pd.Timestamp(game_date - timedelta(days=lookback_days))
    recent = history[history["game_date"] >= lookback_start].copy()
    if recent.empty:
        recent = history.sort_values("game_date").tail(max_players * 8).copy()

    recent["lineup_slot"] = pd.to_numeric(recent.get("lineup_slot"), errors="coerce")
    recent["plate_appearances"] = pd.to_numeric(recent.get("plate_appearances"), errors="coerce").fillna(0.0)
    recent["days_ago"] = (pd.Timestamp(game_date) - recent["game_date"]).dt.days.clip(lower=0)
    recent["recency_weight"] = 1.0 / (recent["days_ago"] + 1.0)
    recent["selection_weight"] = recent["recency_weight"] * recent["plate_appearances"].clip(lower=1.0)

    rows = []
    for player_id, group in recent.groupby("player_id"):
        valid_slots = group["lineup_slot"].dropna().astype(float)
        inferred_slot = None
        if not valid_slots.empty:
            slot_weights = group.loc[valid_slots.index, "selection_weight"].astype(float)
            inferred_slot = float(np.average(valid_slots, weights=slot_weights))
        rows.append(
            {
                "player_id": int(player_id),
                "lineup_slot_estimate": inferred_slot,
                "recent_pa": float(group["plate_appearances"].sum()),
                "recent_games": int(group["game_date"].nunique()),
                "last_game_date": group["game_date"].max(),
            }
        )

    inferred = pd.DataFrame(rows)
    if inferred.empty:
        return pd.DataFrame(columns=columns)

    if players is not None and not players.empty:
        player_lookup = players[["player_id", "full_name"]].drop_duplicates(subset=["player_id"])
        inferred = inferred.merge(player_lookup, on="player_id", how="left")
        inferred["player_name"] = inferred["full_name"].fillna(inferred["player_id"].astype(str))
        inferred = inferred.drop(columns=["full_name"])
    else:
        inferred["player_name"] = inferred["player_id"].astype(str)

    inferred = inferred.sort_values(
        ["lineup_slot_estimate", "recent_pa", "recent_games", "last_game_date"],
        ascending=[True, False, False, False],
        na_position="last",
    ).head(max_players).copy()
    inferred["lineup_slot"] = _assign_lineup_slots(inferred["lineup_slot_estimate"].tolist(), len(inferred))
    inferred["team"] = team
    inferred["is_confirmed"] = False
    inferred = inferred.sort_values("lineup_slot").reset_index(drop=True)
    return inferred[columns]


def hitter_snapshot(
    player_id: int,
    game_date: date,
    player_batting: pd.DataFrame,
    hitter_priors: pd.DataFrame,
    full_weight_pa: int,
    prior_blend_mode: str = "standard",
    prior_weight_multiplier: float = 1.0,
) -> dict[str, float | None]:
    history = player_batting[
        (player_batting["player_id"] == player_id) & (player_batting["game_date"].dt.date < game_date)
    ].copy()
    history = history.sort_values("game_date")
    prior = hitter_priors.loc[player_id] if player_id in hitter_priors.index else {}
    history["had_hit"] = (history["hits"] > 0).astype(float)

    def _window(days: int) -> pd.DataFrame:
        window_start = pd.Timestamp(game_date - timedelta(days=days))
        return history[history["game_date"] >= window_start]

    window_7 = _window(7)
    window_14 = _window(14)
    window_30 = _window(30)
    pa_30 = float(window_30["plate_appearances"].fillna(0).sum())
    hit_rate_30 = _safe_mean(window_30["had_hit"])
    return {
        "hit_rate_7": _safe_mean(window_7["had_hit"]),
        "hit_rate_14": _safe_mean(window_14["had_hit"]),
        "hit_rate_30": hit_rate_30,
        "hit_rate_blended": blend_with_prior(
            hit_rate_30,
            prior.get("prior_hit_rate") if isinstance(prior, pd.Series) else None,
            pa_30,
            full_weight_pa,
            mode=prior_blend_mode,
            prior_weight_multiplier=prior_weight_multiplier,
        ),
        "xba_14": _safe_mean(window_14["xba"]),
        "xwoba_14": _safe_mean(window_14["xwoba"]),
        "hard_hit_pct_14": _safe_mean(window_14["hard_hit_pct"]),
        "k_pct_14": safe_rate(window_14["strikeouts"].fillna(0).sum(), window_14["plate_appearances"].fillna(0).sum()),
        "season_prior_hit_rate": prior.get("prior_hit_rate") if isinstance(prior, pd.Series) else None,
        "season_prior_xba": prior.get("prior_xba") if isinstance(prior, pd.Series) else None,
        "season_prior_xwoba": prior.get("prior_xwoba") if isinstance(prior, pd.Series) else None,
        "streak_len_capped": capped_hit_streak(history),
    }


def bullpen_snapshot(team: str, game_date: date, bullpens: pd.DataFrame) -> dict[str, float | int | None]:
    history = bullpens.copy()
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
    history = history[(history["team"] == team) & (history["game_date"].dt.date < game_date)].copy()
    history = history.sort_values("game_date")
    last3 = history.tail(3)
    previous_day = game_date - timedelta(days=1)
    b2b = int((history["game_date"].dt.date == previous_day).any()) if not history.empty else 0
    outs = int(last3["innings_pitched"].apply(baseball_ip_to_outs).sum()) if not last3.empty else 0
    return {
        "pitches_last3": int(last3["pitches_thrown"].fillna(0).sum()) if not last3.empty else 0,
        "innings_last3": outs_to_baseball_ip(outs),
        "b2b": b2b,
    }


def lineup_snapshot(
    game_id: int,
    team: str,
    cutoff_ts: datetime,
    lineups: pd.DataFrame,
    player_batting: pd.DataFrame,
    hitter_priors: pd.DataFrame,
    game_date: date,
    full_weight_pa: int,
    prior_blend_mode: str = "standard",
    prior_weight_multiplier: float = 1.0,
) -> dict[str, float | bool | None]:
    team_lineups = lineups[(lineups["game_id"] == game_id) & (lineups["team"] == team) & (lineups["snapshot_ts"] <= cutoff_ts)].copy()
    if team_lineups.empty:
        latest = infer_lineup_from_history(team, game_date, player_batting)
        if latest.empty:
            return {"top5_xwoba": None, "lineup_k_pct": None, "confirmed": False}
    else:
        latest_snapshot = team_lineups["snapshot_ts"].max()
        latest = team_lineups[team_lineups["snapshot_ts"] == latest_snapshot].sort_values("lineup_slot")
    player_rows = []
    for player_id in latest["player_id"].astype(int).tolist():
        player_rows.append(
            hitter_snapshot(
                player_id,
                game_date,
                player_batting,
                hitter_priors,
                full_weight_pa,
                prior_blend_mode=prior_blend_mode,
                prior_weight_multiplier=prior_weight_multiplier,
            )
        )
    top5 = player_rows[:5]
    return {
        "top5_xwoba": _safe_mean(pd.Series([row["season_prior_xwoba"] if row["xwoba_14"] is None else row["xwoba_14"] for row in top5])),
        "lineup_k_pct": _safe_mean(pd.Series([row["k_pct_14"] for row in player_rows])),
        "confirmed": bool(latest["is_confirmed"].fillna(False).any()),
    }


def latest_market_snapshot(game_id: int, cutoff_ts: datetime, markets: pd.DataFrame) -> dict[str, Any]:
    frame = markets[(markets["game_id"] == game_id) & (markets["snapshot_ts"] <= cutoff_ts)].copy()
    if frame.empty:
        return {"market_total": None, "market_over_price": None, "market_under_price": None, "line_snapshot_ts": None, "line_movement": None}
    frame = frame.sort_values("snapshot_ts")
    latest = frame.iloc[-1]
    opening = frame.iloc[0]
    return {
        "market_total": latest.get("line_value"),
        "market_over_price": latest.get("over_price"),
        "market_under_price": latest.get("under_price"),
        "line_snapshot_ts": latest.get("snapshot_ts"),
        "line_movement": None if pd.isna(latest.get("line_value")) or pd.isna(opening.get("line_value")) else float(latest.get("line_value") - opening.get("line_value")),
    }


def latest_weather_snapshot(game_id: int, cutoff_ts: datetime, weather: pd.DataFrame) -> dict[str, Any]:
    frame = weather[(weather["game_id"] == game_id) & (weather["snapshot_ts"] <= cutoff_ts)].copy()
    if frame.empty:
        return {"temperature_f": None, "wind_speed_mph": None, "wind_direction_deg": None, "humidity_pct": None}
    latest = frame.sort_values("snapshot_ts").iloc[-1]
    return {
        "temperature_f": latest.get("temperature_f"),
        "wind_speed_mph": latest.get("wind_speed_mph"),
        "wind_direction_deg": latest.get("wind_direction_deg"),
        "humidity_pct": latest.get("humidity_pct"),
    }


def park_snapshot(home_team: str, season: int, park_factors: pd.DataFrame, fallback_season: int) -> dict[str, Any]:
    current = park_factors[(park_factors["team_abbr"] == home_team) & (park_factors["season"] == season)]
    if current.empty:
        current = park_factors[(park_factors["team_abbr"] == home_team) & (park_factors["season"] == fallback_season)]
    if current.empty:
        return {"run_factor": None, "hr_factor": None}
    row = current.iloc[-1]
    return {"run_factor": row.get("run_factor"), "hr_factor": row.get("hr_factor")}


def projected_plate_appearances(lineup_slot: int | float | None) -> float | None:
    if lineup_slot is None or pd.isna(lineup_slot):
        return None
    slot = int(lineup_slot)
    lookup = {1: 4.8, 2: 4.7, 3: 4.6, 4: 4.5, 5: 4.3, 6: 4.1, 7: 4.0, 8: 3.9, 9: 3.8}
    return lookup.get(slot, 4.0)


def to_native(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def write_feature_snapshot(frame: pd.DataFrame, lane: str, start_date: date, end_date: date) -> Path:
    settings = get_settings()
    output_dir = settings.feature_dir / lane
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{start_date.isoformat()}_{end_date.isoformat()}.parquet"
    frame.to_parquet(output_path, index=False)
    return output_path


def _persist_feature_rows(frame: pd.DataFrame, table_name: str, meta_columns: list[str], target_column: str) -> int:
    rows = []
    for record in frame.to_dict(orient="records"):
        payload = {
            key: to_native(value)
            for key, value in record.items()
            if key not in set(meta_columns + [target_column])
        }
        row = {column: to_native(record.get(column)) for column in meta_columns}
        row["feature_payload"] = payload
        row[target_column] = to_native(record.get(target_column))
        rows.append(row)
    if table_name in {"game_features_totals", "game_features_first5_totals"}:
        return upsert_rows(table_name, rows, ["game_id", "feature_cutoff_ts", "feature_version"])
    entity_key = "player_id" if "player_id" in meta_columns else "pitcher_id"
    return upsert_rows(table_name, rows, ["game_id", entity_key, "feature_cutoff_ts", "feature_version"])


def persist_totals_features(frame: pd.DataFrame) -> int:
    return _persist_feature_rows(frame, "game_features_totals", TOTALS_META_COLUMNS, TOTALS_TARGET_COLUMN)


def persist_first5_totals_features(frame: pd.DataFrame) -> int:
    return _persist_feature_rows(
        frame,
        "game_features_first5_totals",
        FIRST5_TOTALS_META_COLUMNS,
        FIRST5_TOTALS_TARGET_COLUMN,
    )


def persist_hits_features(frame: pd.DataFrame) -> int:
    return _persist_feature_rows(frame, "player_features_hits", HITS_META_COLUMNS, HITS_TARGET_COLUMN)


def persist_strikeout_features(frame: pd.DataFrame) -> int:
    return _persist_feature_rows(
        frame,
        "game_features_pitcher_strikeouts",
        [
            "game_id",
            "game_date",
            "pitcher_id",
            "team",
            "opponent",
            "prediction_ts",
            "game_start_ts",
            "line_snapshot_ts",
            "feature_cutoff_ts",
            "feature_version",
        ],
        "actual_strikeouts",
    )


def default_cutoff(game_date: date, game_start_ts: pd.Timestamp | datetime | None) -> datetime:
    if game_start_ts is not None and not pd.isna(game_start_ts):
        if isinstance(game_start_ts, pd.Timestamp):
            game_start_ts = game_start_ts.to_pydatetime()
        return game_start_ts if game_start_ts.tzinfo else game_start_ts.replace(tzinfo=timezone.utc)
    return datetime.combine(game_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=16)