#!/usr/bin/env python3
"""
Comprehensive Data Validation Script
Compares database source data with API response to ensure accuracy
"""

import requests
import json
from sqlalchemy import create_engine, text
import os
import sys
from datetime import datetime, date

def get_engine():
    url = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    return create_engine(url)

def validate_game_data(target_date=None):
    """Comprehensive validation of all game data"""
    
    if target_date is None:
        target_date = date.today().strftime('%Y-%m-%d')
    
    print("🔍 COMPREHENSIVE DATA VALIDATION")
    print(f"📅 Target Date: {target_date}")
    print("=" * 50)
    
    # Get API data
    try:
        response = requests.get(f'http://localhost:8000/comprehensive-games/{target_date}')
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return
        
        api_data = response.json()
        
        # Handle both list and dict responses
        if isinstance(api_data, dict) and 'games' in api_data:
            api_games = api_data['games']
        elif isinstance(api_data, list):
            api_games = api_data
        elif isinstance(api_data, dict):
            # Convert dict to list if needed
            api_games = [api_data]
        else:
            print(f"❌ Unexpected API response format: {type(api_data)}")
            return
            
        print(f"✅ API Response: {len(api_games)} games")
        
    except Exception as e:
        print(f"❌ API Connection Error: {e}")
        return
    
    # Get database data
    engine = get_engine()
    
    print(f"\n📊 DETAILED VALIDATION FOR ALL {len(api_games)} GAMES:")
    print("=" * 50)
    
    validation_summary = {
        "total_games": len(api_games),
        "prediction_errors": 0,
        "betting_errors": 0,
        "weather_errors": 0,
        "pitcher_errors": 0,
        "team_stats_errors": 0,
        "venue_errors": 0
    }
    
    for i, api_game in enumerate(api_games):
        game_id = api_game.get("game_id") or api_game.get("id")
        
        print(f"\n🎯 GAME {i+1}/{len(api_games)}: {api_game.get('away_team')} @ {api_game.get('home_team')} (ID: {game_id})")
        print("-" * 60)
        
        # Get corresponding database data
        with engine.begin() as conn:
            db_data = conn.execute(text("""
                SELECT 
                    eg.*,
                    lpp.predicted_total as pred_total,
                    lpp.p_over, lpp.p_under,
                    lpp.recommendation, lpp.adj_edge,
                    lpp.priced_total,
                    lpp.ev_over, lpp.ev_under,
                    lpp.kelly_over, lpp.kelly_under,
                    lpp.n_books, lpp.spread_cents, lpp.pass_reason
                FROM enhanced_games eg
                LEFT JOIN latest_probability_predictions lpp ON eg.game_id = lpp.game_id AND eg.date = lpp.game_date
                WHERE eg.game_id = :game_id AND eg.date = :target_date
            """), {"game_id": game_id, "target_date": target_date}).fetchone()
            
            if not db_data:
                print("❌ No database data found")
                continue
        
        game_errors = 0
        
        # 1. PREDICTION DATA VALIDATION
        print("🎯 PREDICTION DATA:")
        prediction_errors = 0
        
        api_pred = api_game.get("predicted_total", "0")
        db_pred = str(db_data.pred_total) if db_data.pred_total else "0"
        if api_pred != db_pred:
            print(f"  ❌ Predicted Total: API={api_pred}, DB={db_pred}")
            prediction_errors += 1
        else:
            print(f"  Predicted Total: API={api_pred}, DB={db_pred} ✅")
        
        api_rec = api_game.get("recommendation", "")
        db_rec = db_data.recommendation or ""
        if api_rec != db_rec:
            print(f"  ❌ Recommendation: API={api_rec}, DB={db_rec}")
            prediction_errors += 1
        else:
            print(f"  Recommendation: API={api_rec}, DB={db_rec} ✅")
        
        # Check confidence calculation
        api_conf = float(api_game.get("confidence", 0))
        if db_data.recommendation == "OVER":
            expected_conf = float(db_data.p_over * 100) if db_data.p_over else 50.0
        elif db_data.recommendation == "UNDER":
            expected_conf = float(db_data.p_under * 100) if db_data.p_under else 50.0
        else:
            expected_conf = 50.0
        
        if abs(api_conf - expected_conf) > 0.1:
            print(f"  ❌ Confidence: API={api_conf}%, Expected={expected_conf:.1f}%")
            prediction_errors += 1
        else:
            print(f"  Confidence: API={api_conf}%, Expected={expected_conf:.1f}% ✅")
        
        api_edge = float(api_game.get("edge", 0))
        db_edge = float(db_data.adj_edge) if db_data.adj_edge else 0.0
        if abs(api_edge - db_edge) > 0.01:
            print(f"  ❌ Edge: API={api_edge}, DB={db_edge}")
            prediction_errors += 1
        else:
            print(f"  Edge: API={api_edge}, DB={db_edge} ✅")
        
        # 2. BETTING DATA VALIDATION
        print("\n💰 BETTING DATA:")
        betting_errors = 0
        betting_info = api_game.get("betting_info", {})
        
        api_market = float(betting_info.get("market_total", 0))
        db_market = float(db_data.priced_total) if db_data.priced_total else float(db_data.market_total or 0)
        if abs(api_market - db_market) > 0.01:
            print(f"  ❌ Market Total: API={api_market}, DB={db_market}")
            betting_errors += 1
        else:
            print(f"  Market Total: API={api_market}, DB={db_market} ✅")
        
        # Check new market depth fields
        api_n_books = int(betting_info.get("n_books", 1))
        db_n_books = int(db_data.n_books or 1)
        if api_n_books != db_n_books:
            print(f"  ❌ N Books: API={api_n_books}, DB={db_n_books}")
            betting_errors += 1
        else:
            print(f"  N Books: API={api_n_books}, DB={db_n_books} ✅")
        
        api_spread = int(betting_info.get("spread_cents", 0))
        db_spread = int(db_data.spread_cents or 0)
        if api_spread != db_spread:
            print(f"  ❌ Spread Cents: API={api_spread}, DB={db_spread}")
            betting_errors += 1
        else:
            print(f"  Spread Cents: API={api_spread}, DB={db_spread} ✅")
        
        api_pass_reason = str(betting_info.get("pass_reason", ""))
        db_pass_reason = str(db_data.pass_reason or "")
        if api_pass_reason != db_pass_reason:
            print(f"  ❌ Pass Reason: API=\"{api_pass_reason}\", DB=\"{db_pass_reason}\"")
            betting_errors += 1
        else:
            print(f"  Pass Reason: API=\"{api_pass_reason}\", DB=\"{db_pass_reason}\" ✅")
        
        # 3. WEATHER DATA VALIDATION
        print("\n🌤️ WEATHER DATA:")
        weather_errors = 0
        
        api_temp = api_game.get("temperature")
        db_temp = db_data.temperature
        if str(api_temp) != str(db_temp):
            print(f"  ❌ Temperature: API={api_temp}, DB={db_temp}")
            weather_errors += 1
        else:
            print(f"  Temperature: API={api_temp}, DB={db_temp} ✅")
        
        api_wind = api_game.get("wind_speed")
        db_wind = db_data.wind_speed
        if str(api_wind) != str(db_wind):
            print(f"  ❌ Wind Speed: API={api_wind}, DB={db_wind}")
            weather_errors += 1
        else:
            print(f"  Wind Speed: API={api_wind}, DB={db_wind} ✅")
        
        # GAME SUMMARY
        total_game_errors = prediction_errors + betting_errors + weather_errors
        if total_game_errors == 0:
            print(f"\n✅ GAME {i+1}: ALL DATA VALIDATED SUCCESSFULLY")
        else:
            print(f"\n❌ GAME {i+1} ERRORS: {total_game_errors} total errors")
        
        # Update summary
        validation_summary["prediction_errors"] += prediction_errors
        validation_summary["betting_errors"] += betting_errors
        validation_summary["weather_errors"] += weather_errors
    
    # FINAL SUMMARY
    print(f"\n\n📋 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total Games Validated: {validation_summary['total_games']}")
    print(f"Prediction Errors: {validation_summary['prediction_errors']}")
    print(f"Betting Data Errors: {validation_summary['betting_errors']}")
    print(f"Weather Data Errors: {validation_summary['weather_errors']}")
    
    total_errors = sum([validation_summary['prediction_errors'], validation_summary['betting_errors'], 
                       validation_summary['weather_errors']])
    
    if total_errors == 0:
        print(f"\n✅ ALL DATA VALIDATED SUCCESSFULLY")
    else:
        print(f"\n⚠️ FOUND {total_errors} TOTAL ERRORS across {validation_summary['total_games']} games")
        error_rate = (total_errors / (validation_summary['total_games'] * 3)) * 100  # 3 categories
        print(f"📊 Error Rate: {error_rate:.1f}%")

    print("\n" + "=" * 50)
    print("✅ VALIDATION COMPLETE")

if __name__ == "__main__":
    # Use August 18th if no date specified
    target_date = sys.argv[1] if len(sys.argv) > 1 else "2025-08-18"
    validate_game_data(target_date)
