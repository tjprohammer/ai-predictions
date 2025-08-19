import pandas as pd
from models.infer import get_pitcher_era_stats_infer
from sqlalchemy import create_engine
import os

# Test the ERA function to see what it's actually returning
engine = create_engine(os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"))

# Test with a real pitcher ID from today's games
test_pitcher_id = 621244  # José Berríos from the output
test_date = pd.to_datetime("2025-08-12").date()

print("Testing get_pitcher_era_stats_infer function:")
print(f"Pitcher ID: {test_pitcher_id}")
print(f"Date: {test_date}")

try:
    result = get_pitcher_era_stats_infer(engine, test_pitcher_id, test_date)
    print(f"Result: {result}")
    print(f"Keys: {list(result.keys()) if result else 'None'}")
    
    # Test vs opponent
    result_vs_opp = get_pitcher_era_stats_infer(engine, test_pitcher_id, test_date, opponent_team="cubs")
    print(f"Vs Cubs: {result_vs_opp}")
    
except Exception as e:
    print(f"ERROR: {e}")
    print("This explains why predictions are unrealistic!")
