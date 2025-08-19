import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sqlalchemy import create_engine, text
import psycopg2

# Database connection
conn = psycopg2.connect(
    host='localhost',
    database='mlb',
    user='mlbuser',
    password='mlbpass',
    port='5432'
)

# Create engine for pandas
engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb')

# Get calibration data (same as in probabilities script)
target_date = "2025-08-17"
cal_days = 30
start_dt = pd.to_datetime(target_date) - pd.Timedelta(days=cal_days)
end_dt = pd.to_datetime(target_date) - pd.Timedelta(days=1)

print(f"🔍 Debugging isotonic calibration for {target_date}")
print(f"📊 Calibration window: {start_dt.date()} to {end_dt.date()}")

with engine.begin() as conn_eng:
    # Get calibration data
    calib = pd.read_sql(text("""
        SELECT predicted_total, market_total, total_runs
        FROM enhanced_games
        WHERE "date" BETWEEN :s AND :e
          AND predicted_total IS NOT NULL
          AND market_total IS NOT NULL
          AND total_runs IS NOT NULL
    """), conn_eng, params={"s": start_dt, "e": end_dt})

    # Get today's data
    today = pd.read_sql(text("""
        SELECT game_id, predicted_total, market_total, over_odds, under_odds
        FROM enhanced_games
        WHERE "date" = :d
          AND predicted_total IS NOT NULL
          AND market_total IS NOT NULL
    """), conn_eng, params={"d": target_date})

print(f"📈 Calibration data: {len(calib)} games")
print(f"🎲 Today's data: {len(today)} games")

# Replicate the probability calculation
sigma = calib["total_runs"].std()
bias = (calib["total_runs"] - calib["predicted_total"]).mean()
s = 1.175  # temperature from the script

print(f"🔬 σ (robust): {sigma:.2f} runs")
print(f"🔄 Bias correction: {bias:.3f} runs")
print(f"🧪 Temperature s = {s}")

# Calculate calibration probabilities
z_cal = (calib["market_total"] - (calib["predicted_total"] + bias)) / sigma
z_cal = np.clip(z_cal, np.percentile(z_cal, 1), np.percentile(z_cal, 99))
p_over_raw_calib = 1.0 - norm.cdf(s * z_cal)
p_over_raw_calib = 0.5 + 0.55 * (p_over_raw_calib - 0.5)  # shrinkage

# Get actual outcomes
y = (calib["total_runs"] > calib["market_total"]).astype(float)

print(f"\n📊 Calibration relationship analysis:")
print(f"Raw probability range: {p_over_raw_calib.min():.3f} to {p_over_raw_calib.max():.3f}")
print(f"Actual outcome rate: {y.mean():.3f}")

# Fit isotonic regression
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_over_raw_calib, y)

# Show the relationship at key points
test_probs = np.linspace(p_over_raw_calib.min(), p_over_raw_calib.max(), 10)
calibrated_probs = iso.transform(test_probs)

print(f"\n📐 Isotonic calibration mapping:")
print("Raw Prob → Calibrated Prob")
for raw, cal in zip(test_probs, calibrated_probs):
    print(f"{raw:.3f} → {cal:.3f}")

# Now check our problematic games
print(f"\n🎯 Problematic games analysis:")
problematic_games = today[today["game_id"].isin([776700, 776707])]

for _, game in problematic_games.iterrows():
    game_id = game["game_id"]
    pred = game["predicted_total"]
    market = game["market_total"]
    
    # Calculate raw probability
    z_score = (market - (pred + bias)) / sigma
    z_score_clipped = np.clip(z_score, np.percentile(z_cal, 1), np.percentile(z_cal, 99))
    p_over_raw = 1.0 - norm.cdf(s * z_score_clipped)
    p_over_raw_shrunk = 0.5 + 0.55 * (p_over_raw - 0.5)
    
    # Apply isotonic calibration
    p_over_calibrated = iso.transform([p_over_raw_shrunk])[0]
    
    # Apply clamping
    delta = (p_over_calibrated - 0.5)
    delta_clamped = np.clip(delta, -0.15, 0.15)
    p_over_final = 0.5 + delta_clamped
    
    print(f"\nGame {game_id}: {pred:.1f} vs {market:.1f}")
    print(f"  Z-score: {z_score:.3f} → {z_score_clipped:.3f} (clipped)")
    print(f"  Raw prob: {p_over_raw:.3f} → {p_over_raw_shrunk:.3f} (shrunk)")
    print(f"  Isotonic: {p_over_raw_shrunk:.3f} → {p_over_calibrated:.3f}")
    print(f"  Clamped: {p_over_calibrated:.3f} → {p_over_final:.3f}")
    print(f"  Expected: Model predicts {'MORE' if pred > market else 'FEWER'} runs")
    print(f"  Logic OK: {'YES' if (pred > market and p_over_final > 0.5) or (pred < market and p_over_final < 0.5) else 'NO'}")

conn.close()
