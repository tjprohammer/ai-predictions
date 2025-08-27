#!/usr/bin/env python3
"""
Quick check of today's predictions in the PostgreSQL database
"""
import psycopg2
import os
from pathlib import Path

def check_predictions():
    # PostgreSQL connection details
    try:
        print("🔍 Connecting to PostgreSQL database...")
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='mlb',
            user='mlbuser',
            password='mlbpass'
        )
        
        cursor = conn.cursor()
        
        # Check if enhanced_games table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'enhanced_games'
            )
        """)
        
        if not cursor.fetchone()[0]:
            print("❌ enhanced_games table not found")
            return
        
        print("✅ Connected to PostgreSQL database")
        
        # Check games for today
        cursor.execute("SELECT COUNT(*) FROM enhanced_games WHERE date = %s", ('2025-08-23',))
        count = cursor.fetchone()[0]
        print(f"📊 Games for 2025-08-23: {count}")
        
        if count > 0:
            # Check predictions
            cursor.execute("""
                SELECT home_team, away_team, predicted_total_learning, market_total, 
                       predicted_total, recommendation 
                FROM enhanced_games 
                WHERE date = %s 
                ORDER BY predicted_total_learning DESC NULLS LAST
                LIMIT 15
            """, ('2025-08-23',))
            games = cursor.fetchall()
            
            print("\n🎯 Today's ML Predictions:")
            print("=" * 80)
            for game in games:
                home, away, ml_pred, market, orig_pred, rec = game
                ml_str = f"{ml_pred:.2f}" if ml_pred is not None else "None"
                market_str = f"{market:.1f}" if market is not None else "None"
                orig_str = f"{orig_pred:.2f}" if orig_pred is not None else "None"
                rec_str = rec if rec is not None else "None"
                print(f"{away:20} @ {home:20} | ML={ml_str:>6} | Mkt={market_str:>5} | Orig={orig_str:>6} | {rec_str}")
        
        else:
            print("❌ No games found for 2025-08-23")
        
    except psycopg2.Error as e:
        print(f"❌ PostgreSQL Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_predictions()
