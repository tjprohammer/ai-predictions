#!/usr/bin/env python3
"""
Quick HOLD Percentage Checker
Automatically checks current HOLD percentage and actionable picks
"""

import requests
import sys
from datetime import datetime

def check_holds(date=None):
    """Check current HOLD percentage and actionable picks"""
    
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Get current predictions
        response = requests.get(f'http://localhost:8000/api/comprehensive-games/{date}')
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return
            
        games = response.json()['games']
        
        total_games = len(games)
        if total_games == 0:
            print(f"📅 No games found for {date}")
            return
            
        hold_count = sum(1 for game in games if game['recommendation'] == 'HOLD')
        actionable_count = total_games - hold_count
        hold_percentage = (hold_count / total_games) * 100
        
        # Display results
        print(f'🎯 HOLD ANALYSIS FOR {date}')
        print(f'=' * 50)
        print(f'📊 Total games: {total_games}')
        print(f'⏸️  HOLD recommendations: {hold_count}')
        print(f'✅ Actionable picks: {actionable_count}')
        print(f'📈 HOLD percentage: {hold_percentage:.1f}%')
        
        # Status indicator
        if hold_percentage > 70:
            status = "🔴 HIGH (Too Conservative)"
        elif hold_percentage > 50:
            status = "🟡 MODERATE"
        else:
            status = "🟢 GOOD (Actionable)"
            
        print(f'📊 Status: {status}')
        print()
        
        if actionable_count > 0:
            print(f'🎯 ACTIONABLE PICKS ({actionable_count}):')
            for game in games:
                if game['recommendation'] != 'HOLD':
                    away = game['away_team']
                    home = game['home_team'] 
                    rec = game['recommendation']
                    conf = game['confidence']
                    edge = game.get('edge', 0)
                    print(f'   • {away} @ {home}: {rec} ({conf}% confidence, {edge:+.1f} edge)')
        else:
            print('❌ No actionable picks today')
            
        return {
            'total_games': total_games,
            'hold_count': hold_count,
            'actionable_count': actionable_count,
            'hold_percentage': hold_percentage
        }
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: API server not running (localhost:8000)")
        print("💡 Start the API server: cd mlb-overs && python -m uvicorn api.app:app --host 127.0.0.1 --port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Allow date parameter
    date = sys.argv[1] if len(sys.argv) > 1 else None
    check_holds(date)
