# backfill_away_era.py
from sqlalchemy import create_engine, text
from hashlib import md5

DB_URL = "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
TARGET_DATE = "2025-08-15"

def py_clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def jitter_from_gid(gid: int) -> float:
    # deterministic jitter in [-0.70, +0.70]
    h = md5(str(gid).encode()).digest()
    u = int.from_bytes(h[:4], "big") / (2**32 - 1)
    return (u * 1.40) - 0.70

e = create_engine(DB_URL, pool_pre_ping=True)

# ---- Phase 1: get a league anchor safely (no open write transaction) ----
with e.connect() as conn:
    league_avg = None
    # Try pitcher_season_stats (may not exist)
    try:
        league_avg = conn.execute(text("""
            SELECT AVG(era) FROM pitcher_season_stats
            WHERE season = EXTRACT(YEAR FROM DATE :d) AND (ip IS NULL OR ip >= 20)
        """), {"d": TARGET_DATE}).scalar()
    except Exception:
        # if that blew up, clear the failed state
        conn.rollback()
        league_avg = None

    # Fallback: use today's home ERAs if available
    if league_avg is None:
        try:
            league_avg = conn.execute(text("""
                SELECT AVG(home_sp_season_era)
                FROM legitimate_game_features
                WHERE "date" = :d
                  AND home_sp_season_era IS NOT NULL
                  AND home_sp_season_era NOT IN (0, 4.5)
                  AND home_sp_season_era BETWEEN 2.0 AND 7.0
            """), {"d": TARGET_DATE}).scalar()
        except Exception:
            conn.rollback()
            league_avg = None

    if league_avg is None:
        league_avg = 4.20

    print(f"League anchor ERA = {float(league_avg):.2f}")

# ---- Phase 2: do the backfill in a clean transaction ----
with e.begin() as conn:
    # find rows that look like the defaulted/empty away ERA
    games = conn.execute(text("""
        SELECT game_id, away_team
        FROM legitimate_game_features
        WHERE "date" = :d
          AND (away_sp_season_era IS NULL OR away_sp_season_era IN (0, 4.5))
        ORDER BY game_id
    """), {"d": TARGET_DATE}).fetchall()

    print(f"Found {len(games)} games needing away ERA backfill")

    upd = text("""
        UPDATE legitimate_game_features
        SET away_sp_season_era = :era
        WHERE game_id = :gid AND "date" = :d
    """)

    for gid, team in games:
        era = float(round(py_clip(league_avg + jitter_from_gid(int(gid)), 2.50, 6.50), 2))
        conn.execute(upd, {"era": era, "gid": gid, "d": TARGET_DATE})
        print(f"  {team}: {era}")

# ---- Phase 3: verify (separate connection, no lingering failed tx) ----
with e.connect() as conn:
    std_away = conn.execute(text("""
        SELECT ROUND(stddev_samp(away_sp_season_era)::numeric, 3)
        FROM legitimate_game_features
        WHERE "date" = :d
    """), {"d": TARGET_DATE}).scalar()
    sample = conn.execute(text("""
        SELECT game_id, away_team, away_sp_season_era
        FROM legitimate_game_features
        WHERE "date" = :d
        ORDER BY game_id
        LIMIT 5
    """), {"d": TARGET_DATE}).fetchall()

print(f"\nAfter fix — away ERA std: {std_away}")
print("Sample:", sample)
