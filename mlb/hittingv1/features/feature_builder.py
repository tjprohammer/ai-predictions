import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

def _get_engine():
    url = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    return create_engine(url, pool_pre_ping=True)

def refresh_materialized_views(engine):
    with engine.begin() as conn:
        for mv in ["mv_hitter_form", "mv_bvp_agg", "mv_pa_distribution"]:
            conn.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv}"))

def _eb_rate(h, n, prior_mean=0.245, strength=40):
    a = prior_mean * strength
    b = (1 - prior_mean) * strength
    return (h + a) / (n + a + b) if (n + a + b) > 0 else prior_mean

def build_hitter_features(target_date: str, engine=None) -> pd.DataFrame:
    close_engine = False
    if engine is None:
        engine = _get_engine()
        close_engine = True

    hitters = pd.read_sql(text("""
        SELECT DISTINCT date, game_id, player_id, player_name, team
        FROM player_props_odds
        WHERE date=:d AND market='HITS'
    """), engine, params={"d": target_date})
    if hitters.empty:
        if close_engine: engine.dispose()
        return hitters

    last_ctx = pd.read_sql(text("""
        SELECT DISTINCT ON (player_id)
               player_id, team, opp_team, lineup_slot, pitcher_id, pitcher_hand
        FROM player_game_logs
        WHERE game_date < :d
        ORDER BY player_id, game_date DESC
    """), engine, params={"d": target_date})
    df = hitters.merge(last_ctx, on=["player_id","team"], how="left")

    form = pd.read_sql(text("""
        SELECT player_id, game_date, hits_l5, ab_l5, hits_l10, ab_l10, hits_l15, ab_l15, hits_season, ab_season
        FROM mv_hitter_form WHERE game_date < :d
    """), engine, params={"d": target_date}).sort_values(["player_id","game_date"]).groupby("player_id").tail(1)
    df = df.merge(form, on="player_id", how="left")

    bvp = pd.read_sql(text("""
        SELECT player_id, pitcher_id, hits_bvp, ab_bvp
        FROM mv_bvp_agg
    """), engine)
    df = df.merge(bvp, on=["player_id","pitcher_id"], how="left")

    pa_dist = pd.read_sql(text("""
        SELECT team, lineup_slot, pitcher_hand, pa_avg, ab_avg FROM mv_pa_distribution
    """), engine)
    df = df.merge(pa_dist, on=["team","lineup_slot","pitcher_hand"], how="left")

    df["ba_season"] = (df["hits_season"].fillna(0) / df["ab_season"].replace(0, np.nan)).fillna(0.240)
    ba_l10 = (df["hits_l10"].fillna(0) / df["ab_l10"].replace(0, np.nan))
    ba_l5  = (df["hits_l5" ].fillna(0) / df["ab_l5" ].replace(0, np.nan))
    ba_l15 = (df["hits_l15"].fillna(0) / df["ab_l15"].replace(0, np.nan))
    df["ba_form"] = ba_l10.fillna(ba_l5).fillna(ba_l15).fillna(df["ba_season"])

    df["ba_bvp_raw"] = (df["hits_bvp"].fillna(0) / df["ab_bvp"].replace(0, np.nan))
    df.loc[df["ab_bvp"].fillna(0) < 15, "ba_bvp_raw"] = np.nan

    df["ba_form_eb"] = [_eb_rate(h or 0, n or 0, 0.245, 40) for h, n in zip(df["hits_l10"].fillna(0), df["ab_l10"].fillna(0))]
    df["ba_form_eb"] = df["ba_form_eb"].where(df["ab_l10"].notna(), df["ba_season"])

    df["ba_bvp_eb"] = np.where(df["ba_bvp_raw"].notna(),
                                 [ _eb_rate(h or 0, n or 0, m, 20) for h, n, m in zip(df["hits_bvp"], df["ab_bvp"], df["ba_season"]) ],
                                 np.nan)

    df["ba_mix"] = df["ba_form_eb"].copy()
    idx = df["ba_bvp_eb"].notna()
    df.loc[idx, "ba_mix"] = 0.8*df.loc[idx, "ba_form_eb"] + 0.2*df.loc[idx, "ba_bvp_eb"]

    df["per_pa_hit_rate"] = np.clip(df["ba_mix"].fillna(0.245) * 0.86, 0.05, 0.6)

    df["pa_expected"] = df["pa_avg"].fillna(4.5)
    df["ab_expected"] = df["ab_avg"].fillna(4.0)

    df["hot_z"] = (df["ba_form"].fillna(0.245) - df["ba_season"].fillna(0.245)) / 0.04
    df["hotness"] = np.tanh(df["hot_z"])

    keep = ["date","game_id","player_id","player_name","team","opp_team","lineup_slot","pitcher_id","pitcher_hand",
            "ab_expected","pa_expected","per_pa_hit_rate","ba_season","ba_form","hotness"]
    out = df[keep].drop_duplicates(["date","game_id","player_id"]).copy()

    if close_engine: engine.dispose()
    return out
