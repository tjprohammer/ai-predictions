#!/usr/bin/env python3
"""
Enhanced Results Analysis with Run Traceability
"""

import sys, os
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from sqlalchemy import create_engine, text
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def analyze_latest_run():
    with engine.begin() as conn:
        # Get the latest run
        latest_run = conn.execute(text("""
            SELECT run_id, game_date, sigma, temp_s, bias, calibration_samples, created_at
            FROM calibration_meta 
            ORDER BY created_at DESC 
            LIMIT 1
        """)).fetchone()
        
        if not latest_run:
            print("❌ No calibration runs found")
            return
            
        run_id, game_date, sigma, temp_s, bias, cal_samples, created_at = latest_run
        
        print(f"🎯 LATEST RUN ANALYSIS")
        print(f"📅 Date: {game_date}")
        print(f"🆔 Run ID: {run_id}")
        print(f"🕒 Created: {created_at}")
        print(f"🔬 Calibration: σ={sigma:.3f}, temp_s={temp_s:.3f}, bias={bias:+.3f}")
        print(f"📊 Samples: {cal_samples}")
        
        # Get predictions for this run
        preds = pd.read_sql(text("""
            SELECT game_id, p_side, adj_edge, 
                   GREATEST(ev_over, ev_under) as best_ev,
                   GREATEST(kelly_over, kelly_under) as best_kelly,
                   fair_odds, over_odds, under_odds, recommendation
            FROM probability_predictions 
            WHERE run_id = :rid
            ORDER BY GREATEST(kelly_over, kelly_under) DESC
        """), conn, params={"rid": str(run_id)})
        
        if preds.empty:
            print("❌ No predictions found for this run")
            return
            
        print(f"\n📋 BETTING RECOMMENDATIONS ({len(preds)} games):")
        print("   Game ID | Side  | Prob | EV    | Kelly | Fair  | Book  | Edge")
        print("   --------|-------|------|-------|-------|-------|-------|------")
        
        total_stake = 0
        for _, row in preds.iterrows():
            side_odds = row['over_odds'] if row['recommendation'] == 'OVER' else row['under_odds']
            print(f"   {row['game_id']:>7} | {row['recommendation']:>5} | {row['p_side']:>4.1%} | {row['best_ev']:>+5.1%} | {row['best_kelly']:>5.1%} | {row['fair_odds']:>+5.0f} | {side_odds:>+5.0f} | {row['adj_edge']:>+4.1f}")
            total_stake += row['best_kelly']
        
        print(f"\n📊 RUN SUMMARY:")
        print(f"   Games recommended: {len(preds)}")
        print(f"   OVER bets: {(preds['recommendation'] == 'OVER').sum()}")
        print(f"   UNDER bets: {(preds['recommendation'] == 'UNDER').sum()}")
        print(f"   Average EV: {preds['best_ev'].mean():+.1%}")
        print(f"   Average Kelly: {preds['best_kelly'].mean():.1%}")
        print(f"   Total stake: {total_stake:.1%}")
        
        # Risk warnings
        if total_stake > 0.10:
            print(f"   ⚠️  WARNING: Total stake exceeds 10% daily limit")
        if len(preds) > 5:
            print(f"   ⚠️  WARNING: More than 5 bets recommended")

def show_run_history():
    with engine.begin() as conn:
        runs = pd.read_sql(text("""
            SELECT run_id, game_date, sigma, temp_s, bias, calibration_samples, 
                   created_at,
                   (SELECT COUNT(*) FROM probability_predictions p WHERE p.run_id = c.run_id) as bet_count
            FROM calibration_meta c
            ORDER BY created_at DESC 
            LIMIT 10
        """), conn)
        
        if runs.empty:
            print("❌ No run history found")
            return
            
        print(f"\n📈 RUN HISTORY (Last 10):")
        print("   Date       | Run ID   | Bets | σ     | temp_s | Bias   | Samples")
        print("   -----------|----------|------|-------|--------|--------|--------")
        
        for _, run in runs.iterrows():
            run_id_short = str(run['run_id'])[:8]
            print(f"   {run['game_date']} | {run_id_short} | {run['bet_count']:>4} | {run['sigma']:>5.3f} | {run['temp_s']:>6.3f} | {run['bias']:>+6.3f} | {run['calibration_samples']:>7}")

if __name__ == "__main__":
    analyze_latest_run()
    show_run_history()
