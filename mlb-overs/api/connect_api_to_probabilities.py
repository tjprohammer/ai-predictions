#!/usr/bin/env python3
"""
Quick fix to connect the API to use the new api_games_today view with proper confidence
"""
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

def update_api_query():
    """Update the API to use api_games_today view for today's games"""
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    
    try:
        with engine.begin() as conn:
            print("🔧 Testing api_games_today view...")
            result = conn.execute(text("""
                SELECT COUNT(*) as game_count,
                       AVG(confidence) as avg_confidence,
                       COUNT(CASE WHEN confidence IS NOT NULL THEN 1 END) as games_with_confidence
                FROM api_games_today
            """))
            stats = result.fetchone()
            print(f"✅ View has {stats.game_count} games, avg confidence: {stats.avg_confidence:.1f}%, games with confidence: {stats.games_with_confidence}")
            
            print("\n🔧 Sample of today's games with confidence...")
            result = conn.execute(text("""
                SELECT game_id, home_team, away_team, confidence, recommendation,
                       over_probability, under_probability, expected_value_over, expected_value_under
                FROM api_games_today
                ORDER BY confidence DESC
                LIMIT 3
            """))
            
            for row in result:
                print(f"Game {row.game_id}: {row.away_team} @ {row.home_team}")
                print(f"  Confidence: {row.confidence}% | Recommendation: {row.recommendation}")
                print(f"  Probabilities: Over {row.over_probability:.3f}, Under {row.under_probability:.3f}")
                print(f"  EV: Over {row.expected_value_over:.3f}, Under {row.expected_value_under:.3f}")
                print()
            
            return True
            
    except Exception as e:
        print(f"❌ Error testing view: {e}")
        return False
    finally:
        engine.dispose()

if __name__ == "__main__":
    print("🎯 Connecting API to probability-based confidence calculations...")
    if update_api_query():
        print("\n✅ API ready to use api_games_today view!")
        print("\n📝 Next steps:")
        print("1. Update app.py to use 'FROM api_games_today' instead of 'FROM enhanced_games' for today's games")
        print("2. This will automatically give you the proper confidence from probabilities")
        print("3. Plus you'll get the EV and Kelly betting recommendations!")
    else:
        print("❌ Failed to connect API")
