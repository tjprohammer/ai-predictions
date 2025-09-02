import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from .utils import kelly_fraction
from ..features.feature_builder import build_hitter_features

def _get_engine():
    url = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    return create_engine(url, pool_pre_ping=True)

def prob_at_least_k_hits(p_hit_per_ab: float, ab_exp: float, k: int) -> float:
    n = int(round(max(0.0, ab_exp)))
    p = float(np.clip(p_hit_per_ab, 1e-6, 1-1e-6))
    if n == 0:
        return 0.0
    if k == 1:
        return 1.0 - (1.0 - p) ** n
    if k == 2:
        return 1.0 - (1.0 - p) ** n - n * p * (1.0 - p) ** (n - 1)
    from math import comb
    return 1.0 - sum(comb(n, i) * (p**i) * ((1-p)**(n-i)) for i in range(k))

def run(target_date: str, engine=None):
    close_engine = False
    if engine is None:
        engine = _get_engine()
        close_engine = True

    feats = build_hitter_features(target_date, engine=engine)
    if feats.empty:
        print(f"[hitprops] No hitter features for {target_date}")
        if close_engine: engine.dispose()
        return pd.DataFrame()

    odds = pd.read_sql(text("""
        SELECT date, game_id, player_id, player_name, team, market, line, over_odds, under_odds, COALESCE(book,'consensus') AS book
        FROM player_props_odds
        WHERE date=:d AND market='HITS' AND line IN ('0.5','1.5')
    """), engine, params={"d": target_date})

    df = odds.merge(feats, on=["date","game_id","player_id","player_name","team"], how="left")
    if df.empty:
        print(f"[hitprops] No matching odds+features on {target_date}")
        if close_engine: engine.dispose()
        return pd.DataFrame()

    p_hit_per_ab = np.clip(df["per_pa_hit_rate"].values / 0.86, 1e-6, 0.95)
    p_over_0_5 = []
    p_over_1_5 = []
    for i in range(len(df)):
        p1 = prob_at_least_k_hits(p_hit_per_ab[i], df.loc[i, "ab_expected"], 1)
        p2 = prob_at_least_k_hits(p_hit_per_ab[i], df.loc[i, "ab_expected"], 2)
        p_over_0_5.append(p1)
        p_over_1_5.append(p2)
    df["p_over_0_5"] = p_over_0_5
    df["p_over_1_5"] = p_over_1_5

    def ev_from_prob(p, american):
        if pd.isna(american): return np.nan
        payout = (american / 100.0) if american >= 100 else (100.0 / abs(american))
        return p * payout - (1.0 - p) * 1.0

    def kelly_from_prob(p, american):
        if pd.isna(american): return 0.0
        b = (american / 100.0) if american >= 100 else (100.0 / abs(american))
        q = 1 - p
        frac = (b*p - q) / b
        return max(0.0, frac)

    best = (df.sort_values(["date","game_id","player_id","line","book"]).groupby(["date","game_id","player_id","line"], as_index=False).first())
    o05 = best[best["line"]=='0.5'].copy()
    o15 = best[best["line"]=='1.5'].copy()

    o05["ev_over_0_5"] = [ev_from_prob(p, o) for p, o in zip(o05["p_over_0_5"], o05["over_odds"])]
    o05["kelly_over_0_5"] = [kelly_from_prob(p, o) for p, o in zip(o05["p_over_0_5"], o05["over_odds"])]
    o15["ev_over_1_5"] = [ev_from_prob(p, o) for p, o in zip(o15["p_over_1_5"], o15["over_odds"])]
    o15["kelly_over_1_5"] = [kelly_from_prob(p, o) for p, o in zip(o15["p_over_1_5"], o15["over_odds"])]
    
    cols_keep = ["date","game_id","player_id","player_name","team","opp_team","lineup_slot","pitcher_id","pitcher_hand","ab_expected","pa_expected","per_pa_hit_rate"]
    base = df[cols_keep].drop_duplicates(["date","game_id","player_id"]).copy()

    o05 = o05[["date","game_id","player_id","over_odds","under_odds","p_over_0_5","ev_over_0_5","kelly_over_0_5"]].rename(columns={"over_odds":"over_0_5_odds","under_odds":"under_0_5_odds"})
    o15 = o15[["date","game_id","player_id","over_odds","under_odds","p_over_1_5","ev_over_1_5","kelly_over_1_5"]].rename(columns={"over_odds":"over_1_5_odds","under_odds":"under_1_5_odds"})

    out = (base.merge(o05, on=["date","game_id","player_id"], how="left")
                .merge(o15, on=["date","game_id","player_id"], how="left"))

    up_sql = text("""
        INSERT INTO hitter_prop_predictions (
            date, game_id, player_id, player_name, team, opp_team, lineup_slot,
            pitcher_id, pitcher_hand, ab_expected, pa_expected, per_pa_hit_rate,
            p_over_0_5, p_over_1_5, over_0_5_odds, under_0_5_odds, over_1_5_odds, under_1_5_odds,
            ev_over_0_5, ev_over_1_5, kelly_over_0_5, kelly_over_1_5
        ) VALUES (
            :date, :game_id, :player_id, :player_name, :team, :opp_team, :lineup_slot,
            :pitcher_id, :pitcher_hand, :ab_expected, :pa_expected, :per_pa_hit_rate,
            :p_over_0_5, :p_over_1_5, :over_0_5_odds, :under_0_5_odds, :over_1_5_odds, :under_1_5_odds,
            :ev_over_0_5, :ev_over_1_5, :kelly_over_0_5, :kelly_over_1_5
        )
        ON CONFLICT (date, game_id, player_id) DO UPDATE SET
            player_name = EXCLUDED.player_name,
            team = EXCLUDED.team,
            opp_team = EXCLUDED.opp_team,
            lineup_slot = EXCLUDED.lineup_slot,
            pitcher_id = EXCLUDED.pitcher_id,
            pitcher_hand = EXCLUDED.pitcher_hand,
            ab_expected = EXCLUDED.ab_expected,
            pa_expected = EXCLUDED.pa_expected,
            per_pa_hit_rate = EXCLUDED.per_pa_hit_rate,
            p_over_0_5 = EXCLUDED.p_over_0_5,
            p_over_1_5 = EXCLUDED.p_over_1_5,
            over_0_5_odds = EXCLUDED.over_0_5_odds,
            under_0_5_odds = EXCLUDED.under_0_5_odds,
            over_1_5_odds = EXCLUDED.over_1_5_odds,
            under_1_5_odds = EXCLUDED.under_1_5_odds,
            ev_over_0_5 = EXCLUDED.ev_over_0_5,
            ev_over_1_5 = EXCLUDED.ev_over_1_5,
            kelly_over_0_5 = EXCLUDED.kelly_over_0_5,
            kelly_over_1_5 = EXCLUDED.kelly_over_1_5,
            created_at = now()
    """)
    with engine.begin() as conn:
        for r in out.to_dict(orient="records"):
            conn.execute(up_sql, r)

    if close_engine: engine.dispose()
    return out
