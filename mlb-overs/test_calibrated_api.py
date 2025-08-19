#!/usr/bin/env python3

from api.app import get_engine
from sqlalchemy import text

def main():
    try:
        engine = get_engine()
        print("Connected to database")
        
        with engine.begin() as conn:
            # Check calibrated predictions
            result = conn.execute(text("SELECT COUNT(*) FROM calibrated_predictions WHERE prediction_date = '2025-08-14'"))
            cal_count = result.fetchone()[0]
            print(f"Calibrated predictions count: {cal_count}")
            
            # Check enhanced games structure first
            result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'enhanced_games' ORDER BY ordinal_position"))
            columns = [row[0] for row in result.fetchall()]
            print(f"Enhanced games columns: {columns[:10]}...")  # First 10 columns
            
            # Try to find games for our date (look for date-like columns)
            date_columns = [col for col in columns if 'date' in col.lower() or 'time' in col.lower()]
            print(f"Date-like columns: {date_columns}")
            
            if date_columns:
                date_col = date_columns[0]  # Use first date column
                result = conn.execute(text(f"SELECT COUNT(*) FROM enhanced_games WHERE {date_col}::date = '2025-08-14'"))
                game_count = result.fetchone()[0]
                print(f"Enhanced games count for 2025-08-14: {game_count}")
            else:
                result = conn.execute(text("SELECT COUNT(*) FROM enhanced_games"))
                total_games = result.fetchone()[0]
                print(f"Total enhanced games: {total_games}")
            
            # Check if we have any calibrated predictions at all
            result = conn.execute(text("SELECT COUNT(*) FROM calibrated_predictions"))
            total_cal = result.fetchone()[0]
            print(f"Total calibrated predictions: {total_cal}")
            
            # Check dates in calibrated_predictions
            result = conn.execute(text("SELECT DISTINCT prediction_date FROM calibrated_predictions ORDER BY prediction_date"))
            dates = result.fetchall()
            print(f"Available prediction dates: {[str(d[0]) for d in dates]}")
            
            # Test actual API call with more detailed error info
            print("\nTesting API endpoint...")
            import requests
            try:
                response = requests.get("http://localhost:8000/api/comprehensive-games-with-calibrated/2025-08-14")
                print(f"API Response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"API Error: {response.text}")
                else:
                    data = response.json()
                    print(f"API returned {data.get('count', 0)} games")
                    if data.get('games'):
                        first_game = data['games'][0]
                        print(f"First game: {first_game['home_team']} vs {first_game['away_team']}")
                        print(f"Original prediction: {first_game['predicted_total']}")
                        print(f"Calibrated prediction: {first_game['calibrated_predictions']}")
            except Exception as api_error:
                print(f"API call failed: {api_error}")
                
            # Test the SQL query directly to isolate the issue
            print("\nTesting SQL query directly...")
            query = """
            SELECT 
                eg.game_id, eg.date, eg.home_team, eg.away_team,
                eg.predicted_total, eg.confidence, eg.recommendation, eg.edge,
                eg.market_total, eg.over_odds, eg.under_odds,
                cp.calibrated_predicted_total,
                cp.calibrated_confidence,
                cp.calibrated_recommendation,
                cp.calibrated_edge,
                cp.calibrated_betting_value,
                cp.model_version as calibrated_model_version
            FROM enhanced_games eg
            LEFT JOIN calibrated_predictions cp ON eg.game_id = cp.game_id 
                AND cp.prediction_date = %s
            WHERE eg.date = %s
            ORDER BY eg.game_id
            """
            
            result = conn.execute(text(query), ('2025-08-14', '2025-08-14'))
            rows = result.fetchall()
            print(f"SQL query returned {len(rows)} rows")
            
            if rows:
                first_row = rows[0]
                print(f"First row - Game: {first_row.home_team} vs {first_row.away_team}")
                print(f"Original pred: {first_row.predicted_total}, Calibrated pred: {first_row.calibrated_predicted_total}")
                print(f"Row data types: {[(k, type(v).__name__) for k, v in zip(first_row._fields, first_row)][:5]}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
