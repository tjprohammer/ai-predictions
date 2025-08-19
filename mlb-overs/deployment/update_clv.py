#!/usr/bin/env python3
"""
CLV (Closing Line Value) Updater

Compares our bet odds against closing market odds to measure bet quality.
CLV is the gold standard for measuring betting skill independent of outcomes.

Usage:
    python update_clv.py 2025-08-18
"""

from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple, Optional

# Database connection
DB = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
engine = create_engine(DB, pool_pre_ping=True)

def american_to_prob(odds: float) -> float:
    """Convert American odds to implied probability"""
    if pd.isna(odds) or odds == 0:
        return 0.5
    return (abs(odds)/(abs(odds)+100.0)) if odds < 0 else (100.0/(odds+100.0))

def novig_prob(p_a: float, p_b: float) -> Tuple[Optional[float], Optional[float]]:
    """Remove vig from two-way probabilities"""
    s = p_a + p_b
    if s > 0:
        return (p_a/s, p_b/s)
    return (None, None)

def update_clv_for_date(target_date: str) -> None:
    """Update CLV for all bets on target_date"""
    
    print(f"🔍 Updating CLV for {target_date}")
    
    with engine.begin() as conn:
        # Get bets we placed
        bets = pd.read_sql(text("""
            SELECT b.run_id, b.game_id, b.game_date, b.side, b.odds, b.fair_odds,
                   b.stake, b.book, b.total as bet_total,
                   eg.market_total, eg.home_team, eg.away_team
            FROM bet_outcomes b
            JOIN enhanced_games eg ON eg.game_id=b.game_id AND eg.date=b.game_date
            WHERE b.game_date=:d
              AND b.clv IS NULL  -- Only update bets without CLV
        """), conn, params={"d": target_date})

        if bets.empty: 
            print("ℹ️  No bets found or all CLV already calculated.")
            return

        print(f"📊 Found {len(bets)} bets to update CLV for")

        # Get closing snapshots
        close = pd.read_sql(text("""
            SELECT game_id, date AS game_date, book, total AS book_total,
                   over_odds, under_odds, collected_at
            FROM totals_odds_close
            WHERE date=:d
        """), conn, params={"d": target_date})

    if close.empty:
        print("⚠️  No closing odds found. Make sure totals_odds_close is populated.")
        return

    print(f"📈 Found closing odds for {len(close)} book/game combinations")

    clv_updates = []
    
    for _, bet in bets.iterrows():
        # Find closing odds for this game
        game_closes = close[close.game_id.eq(bet.game_id)].copy()
        if game_closes.empty: 
            print(f"⚠️  No closing odds for game {bet.game_id}")
            continue
        
        # Prefer same total, else find nearest within 0.5
        game_closes["total_diff"] = (game_closes.book_total - bet.bet_total).abs()
        game_closes = game_closes.sort_values(["total_diff", "collected_at"], ascending=[True, False])
        
        # Filter to reasonable total range
        valid_closes = game_closes[game_closes.total_diff <= 0.5]
        if valid_closes.empty:
            print(f"⚠️  No close total within 0.5 of bet total {bet.bet_total} for game {bet.game_id}")
            continue
        
        # Use best match (same total if available, latest timestamp)
        best_close = valid_closes.iloc[0]
        
        # Calculate cents CLV (exact line match only)
        clv_cents = None
        if abs(float(best_close.book_total) - float(bet.bet_total)) < 0.01:  # Same total
            close_odds = best_close.over_odds if bet.side == "OVER" else best_close.under_odds
            if pd.notna(close_odds) and pd.notna(bet.odds):
                clv_cents = int(close_odds - bet.odds)
        
        # Calculate no-vig probability CLV
        p_over_raw = american_to_prob(best_close.over_odds)
        p_under_raw = american_to_prob(best_close.under_odds)
        p_over_nv, p_under_nv = novig_prob(p_over_raw, p_under_raw)
        
        bet_prob_nv = p_over_nv if bet.side == "OVER" else p_under_nv
        fair_prob = american_to_prob(bet.fair_odds) if pd.notna(bet.fair_odds) else None
        
        prob_clv = None
        if bet_prob_nv and fair_prob:
            prob_clv = round(bet_prob_nv - fair_prob, 4)
        
        clv_updates.append({
            "run_id": bet.run_id,
            "game_id": bet.game_id,
            "clv": clv_cents,
            "clv_prob": prob_clv,
            "close_total": best_close.book_total,
            "close_over": best_close.over_odds,
            "close_under": best_close.under_odds
        })
        
        # Log the CLV calculation
        clv_str = f"+{clv_cents}" if clv_cents and clv_cents > 0 else str(clv_cents)
        prob_str = f"+{prob_clv:.3f}" if prob_clv and prob_clv > 0 else f"{prob_clv:.3f}" if prob_clv else "N/A"
        print(f"💰 {bet.game_id} {bet.side}: CLV = {clv_str} cents, Prob = {prob_str}")

    # Write CLV back to database
    if clv_updates:
        with engine.begin() as conn:
            for update in clv_updates:
                conn.execute(text("""
                    UPDATE bet_outcomes
                    SET clv = COALESCE(:clv, clv),
                        clv_prob = COALESCE(:clv_prob, clv_prob),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE run_id = :run_id AND game_id = :game_id
                """), {
                    "clv": update["clv"],
                    "clv_prob": update["clv_prob"], 
                    "run_id": update["run_id"],
                    "game_id": update["game_id"]
                })
        
        print(f"✅ Updated CLV for {len(clv_updates)} bets")
        
        # Summary stats
        cents_clv = [u["clv"] for u in clv_updates if u["clv"] is not None]
        prob_clv = [u["clv_prob"] for u in clv_updates if u["clv_prob"] is not None]
        
        if cents_clv:
            avg_cents = np.mean(cents_clv)
            print(f"📊 Average cents CLV: {avg_cents:+.1f} (range: {min(cents_clv):+d} to {max(cents_clv):+d})")
        
        if prob_clv:
            avg_prob = np.mean(prob_clv)
            print(f"📊 Average probability CLV: {avg_prob:+.3f} (range: {min(prob_clv):+.3f} to {max(prob_clv):+.3f})")
            
    else:
        print("ℹ️  No CLV updates needed")

def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python update_clv.py YYYY-MM-DD")
        sys.exit(1)
    
    target_date = sys.argv[1]
    
    try:
        update_clv_for_date(target_date)
    except Exception as e:
        print(f"❌ Error updating CLV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
