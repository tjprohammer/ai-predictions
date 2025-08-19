import pandas as pd
from sqlalchemy import create_engine, text
import os
import datetime as dt

# Copy the ERA functions from app.py to test them directly
def _era_from_rows(sub: pd.DataFrame):
    """Prefer ER/IP; if unavailable, fall back to mean(era_game)."""
    if sub.empty:
        return None
    er = pd.to_numeric(sub.get("er"), errors="coerce").fillna(0).sum()
    ip = pd.to_numeric(sub.get("ip"), errors="coerce").fillna(0).sum()
    if ip and ip > 0:
        return float((er * 9.0) / ip)
    # fallback
    if "era_game" in sub.columns:
        eg = pd.to_numeric(sub["era_game"], errors="coerce")
        m = eg.mean(skipna=True)
        return None if pd.isna(m) else float(m)
    return None

def lastN_era(starts: pd.DataFrame, pid: str, cutoff_date: dt.date, N: int):
    if pid is None:
        return None
    sub = (
        starts[(starts["pitcher_id"] == pid) & (starts["date"] < cutoff_date)]
        .sort_values("date")
        .tail(N)
    )
    return _era_from_rows(sub)

eng = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

# Load the starts data like the API does
starts = pd.read_sql(text("""
    SELECT pitcher_id, team, opp_team, date, ip, er, era_game
    FROM pitchers_starts
"""), eng)

if not starts.empty:
    starts["date"] = pd.to_datetime(starts["date"]).dt.date
    starts["pitcher_id"] = starts["pitcher_id"].astype(str)
    for col in ("ip","er","era_game"):
        if col in starts.columns:
            starts[col] = pd.to_numeric(starts[col], errors="coerce")

# Test with pitcher 701542 who has data
test_date = dt.date(2025, 8, 12)  # day after the data
era_l3 = lastN_era(starts, "701542", test_date, 3)
era_l5 = lastN_era(starts, "701542", test_date, 5)

print(f"ERA L3 for pitcher 701542: {era_l3}")
print(f"ERA L5 for pitcher 701542: {era_l5}")

# Test with a pitcher that has NULL data
era_null = lastN_era(starts, "667755", test_date, 3)
print(f"ERA L3 for pitcher 667755 (null data): {era_null}")

# Show what data we're working with for 701542
test_data = starts[starts["pitcher_id"] == "701542"].sort_values("date").tail(5)
print(f"\nData for pitcher 701542:")
print(test_data[["pitcher_id", "date", "ip", "er", "era_game"]])
