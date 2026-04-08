"""Quick check of today's predictions and model state."""
import sqlite3, os, glob
from datetime import datetime

db = sqlite3.connect("db/mlb_predictor.sqlite3")
db.row_factory = sqlite3.Row

TODAY = "2026-04-06"

# ── Full-game totals ──
print("=" * 90)
print("FULL-GAME TOTALS PREDICTIONS — " + TODAY)
print("=" * 90)
rows = db.execute(
    "SELECT p.game_id, g.away_team, g.home_team, g.status, g.total_runs,"
    " p.predicted_total_runs, p.market_total, p.model_version"
    " FROM predictions_totals p"
    " JOIN games g ON p.game_id = g.game_id"
    " WHERE p.game_date = ? ORDER BY g.away_team", (TODAY,)
).fetchall()

hdr = f"{'Matchup':<16} {'Pred':>6} {'Market':>7} {'Dir':>6} {'Edge':>6} {'Actual':>7}"
print(hdr)
print("-" * len(hdr))
unders = overs = no_line = 0
for r in rows:
    matchup = r["away_team"] + "@" + r["home_team"]
    pred = f"{r['predicted_total_runs']:.1f}" if r["predicted_total_runs"] else "N/A"
    mkt = f"{r['market_total']:.1f}" if r["market_total"] else "---"
    actual = str(r["total_runs"]) if r["total_runs"] is not None else "---"
    if r["market_total"]:
        if r["predicted_total_runs"] < r["market_total"]:
            direction = "UNDER"
            unders += 1
        elif r["predicted_total_runs"] > r["market_total"]:
            direction = "OVER"
            overs += 1
        else:
            direction = "PUSH"
            no_line += 1
        edge = f"{r['predicted_total_runs'] - r['market_total']:+.2f}"
    else:
        direction = "no line"
        edge = "---"
        no_line += 1
    print(f"{matchup:<16} {pred:>6} {mkt:>7} {direction:>6} {edge:>6} {actual:>7}")

print(f"\nSummary: {unders} UNDER, {overs} OVER, {no_line} no market line")
if rows:
    avg_pred = sum(r["predicted_total_runs"] for r in rows if r["predicted_total_runs"]) / len(rows)
    print(f"Average predicted total: {avg_pred:.2f}")

# ── First-5 totals ──
print()
print("=" * 90)
print("FIRST-5 TOTALS PREDICTIONS — " + TODAY)
print("=" * 90)
rows2 = db.execute(
    "SELECT p.game_id, g.away_team, g.home_team,"
    " p.predicted_total_runs, p.market_total,"
    " p.confidence_level, p.suppress_reason, p.model_version,"
    " g.total_runs_first5"
    " FROM predictions_first5_totals p"
    " JOIN games g ON p.game_id = g.game_id"
    " WHERE p.game_date = ? ORDER BY g.away_team", (TODAY,)
).fetchall()

hdr2 = f"{'Matchup':<16} {'Pred':>6} {'Market':>7} {'Dir':>6} {'Edge':>6} {'Actual':>7} {'Conf':<10}"
print(hdr2)
print("-" * len(hdr2))
u2 = o2 = n2 = 0
for r in rows2:
    matchup = r["away_team"] + "@" + r["home_team"]
    pred = f"{r['predicted_total_runs']:.1f}" if r["predicted_total_runs"] else "N/A"
    mkt = f"{r['market_total']:.1f}" if r["market_total"] else "---"
    actual = str(r["total_runs_first5"]) if r["total_runs_first5"] is not None else "---"
    conf = r["confidence_level"] or "---"
    if r["market_total"]:
        if r["predicted_total_runs"] < r["market_total"]:
            direction = "UNDER"
            u2 += 1
        elif r["predicted_total_runs"] > r["market_total"]:
            direction = "OVER"
            o2 += 1
        else:
            direction = "PUSH"
            n2 += 1
        edge = f"{r['predicted_total_runs'] - r['market_total']:+.2f}"
    else:
        direction = "no line"
        edge = "---"
        n2 += 1
    print(f"{matchup:<16} {pred:>6} {mkt:>7} {direction:>6} {edge:>6} {actual:>7} {conf:<10}")

print(f"\nSummary: {u2} UNDER, {o2} OVER, {n2} no market line")
if rows2:
    avg2 = sum(r["predicted_total_runs"] for r in rows2 if r["predicted_total_runs"]) / len(rows2)
    print(f"Average predicted first5 total: {avg2:.2f}")

# ── Game status ──
print()
print("=" * 90)
print("GAME STATUS — " + TODAY)
print("=" * 90)
games_q = db.execute(
    "SELECT game_id, away_team, home_team, status, total_runs, total_runs_first5"
    " FROM games WHERE game_date = ? ORDER BY away_team", (TODAY,)
).fetchall()
for g in games_q:
    matchup = g["away_team"] + "@" + g["home_team"]
    runs = str(g["total_runs"]) if g["total_runs"] is not None else "---"
    f5 = str(g["total_runs_first5"]) if g["total_runs_first5"] is not None else "---"
    print(f"  {matchup:<16} status={g['status']:<12} total_runs={runs:<5} first5={f5}")

db.close()
import sys; sys.exit(0)

# Games
games = db.execute(
    "SELECT game_id, game_date, home_team, away_team, status, total_runs, total_runs_first5 "
    "FROM games WHERE game_date = ? ORDER BY game_id", (TODAY,)
).fetchall()
print(f"=== Games today ({TODAY}): {len(games)} ===")
for g in games[:8]:
    print(f"  {g['game_id']}  {g['away_team']}@{g['home_team']}  status={g['status']}  runs={g['total_runs']}  first5={g['total_runs_first5']}")
if len(games) > 8:
    print(f"  ... and {len(games)-8} more")

# Predictions per lane
# Discover actual prediction table names
pred_tables = [r[0] for r in db.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'predictions%'"
).fetchall()]
print(f"\nPrediction tables in DB: {pred_tables}")

lane_map = []
for t in pred_tables:
    label = t.replace("predictions_", "").replace("_", " ").title()
    lane_map.append((label, t))

for lane, table in lane_map:
    row = db.execute(f"SELECT count(*), count(distinct game_id) FROM {table} WHERE game_date = ?", (TODAY,)).fetchone()
    print(f"\n=== {lane} predictions: {row[0]} rows / {row[1]} games ===")
    if row[0] > 0:
        cols = [c[1] for c in db.execute(f"PRAGMA table_info({table})").fetchall()]
        # Show sample
        sample = db.execute(f"SELECT * FROM {table} WHERE game_date = ? LIMIT 2", (TODAY,)).fetchall()
        for s in sample:
            d = dict(s)
            # Show key fields
            out = f"  game={d.get('game_id')}"
            if "predicted_total" in d: out += f"  pred={d['predicted_total']}"
            if "predicted_strikeouts" in d: out += f"  pred_k={d['predicted_strikeouts']}"
            if "hit_probability" in d: out += f"  hit_prob={d['hit_probability']}"  
            if "market_total" in d: out += f"  mkt={d['market_total']}"
            if "model_version" in d: out += f"  v={d['model_version']}"
            if "confidence_level" in d: out += f"  conf={d.get('confidence_level')}"
            if "suppress_reason" in d: out += f"  suppress={d.get('suppress_reason')}"
            print(out)

# Outcomes (actual results available?)
finals = db.execute(
    "SELECT count(*) FROM games WHERE game_date = ? AND status = 'Final'", (TODAY,)
).fetchone()[0]
live = db.execute(
    "SELECT count(*) FROM games WHERE game_date = ? AND status NOT IN ('Final', 'Preview', 'Scheduled', 'Pre-Game')", (TODAY,)
).fetchone()[0]
print(f"\n=== Game status: {finals} Final, {live} Live/In-Progress, {len(games) - finals - live} Not Started ===")

# Model artifacts
print("\n=== Model artifacts ===")
for lane in ["totals", "first5_totals", "strikeouts", "hits"]:
    arts = sorted(glob.glob(f"data/models/{lane}/*.pkl"))
    if arts:
        latest = arts[-1]
        mt = datetime.fromtimestamp(os.path.getmtime(latest)).strftime("%Y-%m-%d %H:%M")
        print(f"  {lane}: {os.path.basename(latest)} (trained {mt})")
    else:
        print(f"  {lane}: NO MODEL FOUND")

# Check if features exist for today
# Check if features exist for today
feat_tables = [r[0] for r in db.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'game_features%'"
).fetchall()]
for table in feat_tables:
    row = db.execute(f"SELECT count(*) FROM {table} WHERE game_date = ?", (TODAY,)).fetchone()
    print(f"  {table}: {row[0]} rows")

db.close()
