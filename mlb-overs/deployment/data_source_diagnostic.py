#!/usr/bin/env python3
"""
Data Source Diagnostic
======================
Compare legitimate_game_features vs enhanced_games to see data quality
and preview COALESCE results
"""

from sqlalchemy import create_engine, text
import pandas as pd

e = create_engine("postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
d = "2025-08-15"

def exists(table):
    q = text("""
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema='public' AND table_name=:t
        LIMIT 1
    """)
    with e.begin() as c:
        return c.execute(q, {"t": table}).fetchone() is not None

# 1) Quick peek at enhanced_games
if exists("enhanced_games"):
    eg = pd.read_sql(text("""
        SELECT game_id, home_team, away_team,
               home_sp_season_era AS eg_home_era,
               away_sp_season_era AS eg_away_era,
               market_total AS eg_market
        FROM enhanced_games
        WHERE "date" = :d
        ORDER BY game_id
    """), e, params={"d": d})
    print("Enhanced_games sample:")
    print(eg.head(5))
else:
    eg = pd.DataFrame()
    print("enhanced_games not found")

# 2) legit features for the same date
if exists("legitimate_game_features"):
    lgf = pd.read_sql(text("""
        SELECT game_id, home_team, away_team,
               home_sp_season_era AS lgf_home_era,
               away_sp_season_era AS lgf_away_era,
               market_total AS lgf_market
        FROM legitimate_game_features
        WHERE "date" = :d
        ORDER BY game_id
    """), e, params={"d": d})
    print("\nlegitimate_game_features sample:")
    print(lgf.head(5))
else:
    lgf = pd.DataFrame()
    print("legitimate_game_features not found")

# Bail early if either is missing
if eg.empty or lgf.empty:
    raise SystemExit("\nMissing data to compare; stopping.")

# 3) Merge & show mismatches
m = lgf.merge(eg, on=["game_id", "home_team", "away_team"], how="left")
m["away_era_same"]   = (m["lgf_away_era"] == m["eg_away_era"])
m["market_same"]     = (m["lgf_market"]   == m["eg_market"])
m["lgf_away_bad"]    = m["lgf_away_era"].isin([0.0, 4.5]) | m["lgf_away_era"].isna()
m["lgf_market_bad"]  = (m["lgf_market"].isna()) | (m["lgf_market"] == 0)

print("\nCounts ({}):".format(d))
print({
    "rows": len(m),
    "lgf_away_bad": int(m["lgf_away_bad"].sum()),
    "lgf_market_bad": int(m["lgf_market_bad"].sum()),
    "away_era_equal": int(m["away_era_same"].sum()),
    "market_equal": int(m["market_same"].sum()),
})

# 4) Show rows we can fix from enhanced_games immediately
can_fix_era = m[m["lgf_away_bad"] & m["eg_away_era"].notna()]
can_fix_mkt = m[m["lgf_market_bad"] & m["eg_market"].notna() & (m["eg_market"] > 0)]

print("\nRows where lgf.away_era is bad but eg has a value:")
print(can_fix_era[["game_id","home_team","away_team","lgf_away_era","eg_away_era"]].head(10))

print("\nRows where lgf.market_total is bad but eg has a value:")
print(can_fix_mkt[["game_id","home_team","away_team","lgf_market","eg_market"]].head(10))

# 5) What your COALESCE should yield (preview, no writes)
m["away_era_preview"] = m.apply(
    lambda r: (r["lgf_away_era"] if (pd.notna(r["lgf_away_era"]) and r["lgf_away_era"] not in (0.0, 4.5))
               else r["eg_away_era"]),
    axis=1,
)
m["market_preview"] = m.apply(
    lambda r: (r["lgf_market"] if (pd.notna(r["lgf_market"]) and r["lgf_market"] != 0)
               else r["eg_market"]),
    axis=1,
)

print("\nPreview (COALESCE logic) – first 8:")
print(m[["game_id","away_team","lgf_away_era","eg_away_era","away_era_preview",
         "lgf_market","eg_market","market_preview"]].head(8))

# 6) Quick variance sanity
def std(s):
    try:
        return float(pd.to_numeric(s, errors="coerce").std())
    except Exception:
        return None

print("\nSTD by table ({}):".format(d))
print({
    "eg_away_era_std": std(eg["eg_away_era"]),
    "lgf_away_era_std": std(lgf["lgf_away_era"]),
    "eg_market_std": std(eg["eg_market"]),
    "lgf_market_std": std(lgf["lgf_market"]),
})

# 7) Optional: snoop other "game/API" tables for pitcher columns
tables = pd.read_sql("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema='public'
    ORDER BY table_name
""", e)
game_tables = [t for t in tables.table_name if any(k in t.lower() for k in ("game","mlb","api"))]
print("\nGame/API tables:", game_tables)

if "mlb_games_today" in game_tables:
    today = pd.read_sql('SELECT * FROM mlb_games_today LIMIT 1', e)
    pcols = [c for c in today.columns if any(k in c.lower() for k in ("pitcher","starter","era","whip","k_per_9"))]
    print("\nmlb_games_today pitcher-ish columns:", pcols)
    if pcols:
        print(today[pcols])
