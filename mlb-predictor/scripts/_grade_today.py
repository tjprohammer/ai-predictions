"""Grade today's totals predictions against actual final scores."""
import sqlite3

db = sqlite3.connect("db/mlb_predictor.sqlite3")
db.row_factory = sqlite3.Row

TODAY = "2026-04-06"

# ── Full-game totals grading ──
print("FULL-GAME TOTALS GRADING — " + TODAY)
print("=" * 85)
rows = db.execute(
    "SELECT p.predicted_total_runs, p.market_total, g.total_runs, g.away_team, g.home_team"
    " FROM predictions_totals p"
    " JOIN games g ON p.game_id = g.game_id"
    " WHERE p.game_date = ? AND g.total_runs IS NOT NULL"
    " ORDER BY g.away_team", (TODAY,)
).fetchall()

wins = losses = pushes = 0
total_mae = 0.0
total_actual = 0
total_pred = 0.0
for r in rows:
    pred = r["predicted_total_runs"]
    actual = r["total_runs"]
    mkt = r["market_total"]
    mae = abs(pred - actual)
    total_mae += mae
    total_actual += actual
    total_pred += pred
    matchup = r["away_team"] + "@" + r["home_team"]

    if mkt:
        model_says = "UNDER" if pred < mkt else "OVER"
        if actual < mkt:
            actual_result = "UNDER"
        elif actual > mkt:
            actual_result = "OVER"
        else:
            actual_result = "PUSH"

        if actual_result == "PUSH":
            grade = "PUSH"
            pushes += 1
        elif model_says == actual_result:
            grade = "WIN"
            wins += 1
        else:
            grade = "LOSS"
            losses += 1
        print(f"  {matchup:<14} pred={pred:.1f} mkt={mkt:.1f} actual={actual:>3}  model={model_says:<5} result={actual_result:<5} -> {grade}  (MAE={mae:.1f})")
    else:
        print(f"  {matchup:<14} pred={pred:.1f} mkt=---  actual={actual:>3}  (no line)  (MAE={mae:.1f})")

n = len(rows)
print(f"\nRecord: {wins}W-{losses}L-{pushes}P out of {wins+losses+pushes} with market lines")
print(f"Average MAE: {total_mae / n:.2f}")
print(f"Average actual total: {total_actual / n:.1f}")
print(f"Average predicted: {total_pred / n:.1f}")

# ── First-5 totals grading ──
print()
print("FIRST-5 TOTALS GRADING — " + TODAY)
print("=" * 85)
rows2 = db.execute(
    "SELECT p.predicted_total_runs, p.market_total, g.total_runs_first5, g.away_team, g.home_team"
    " FROM predictions_first5_totals p"
    " JOIN games g ON p.game_id = g.game_id"
    " WHERE p.game_date = ? AND g.total_runs_first5 IS NOT NULL"
    " ORDER BY g.away_team", (TODAY,)
).fetchall()

total_mae2 = 0.0
total_actual2 = 0
total_pred2 = 0.0
for r in rows2:
    pred = r["predicted_total_runs"]
    actual = r["total_runs_first5"]
    mae = abs(pred - actual)
    total_mae2 += mae
    total_actual2 += actual
    total_pred2 += pred
    matchup = r["away_team"] + "@" + r["home_team"]
    print(f"  {matchup:<14} pred={pred:.1f}  actual={actual:>3}  MAE={mae:.1f}")

n2 = len(rows2)
print(f"\nAverage MAE: {total_mae2 / n2:.2f}")
print(f"Average actual first5: {total_actual2 / n2:.1f}")
print(f"Average predicted: {total_pred2 / n2:.1f}")

db.close()
