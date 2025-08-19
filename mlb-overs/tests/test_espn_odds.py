#!/usr/bin/env python3
"""
Test ESPN API for real betting odds
"""

import requests
import json

def test_espn_odds():
    try:
        print("🎯 Testing ESPN API for real betting odds...")
        url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ ESPN API responded with {len(data.get('events', []))} events")
        
        games_with_odds = []
        
        for event in data.get('events', []):
            competitions = event.get('competitions', [])
            if not competitions:
                continue
                
            comp = competitions[0]
            competitors = comp.get('competitors', [])
            
            # Get team names
            home_team = None
            away_team = None
            for competitor in competitors:
                team_name = competitor.get('team', {}).get('displayName', '')
                if competitor.get('homeAway') == 'home':
                    home_team = team_name
                else:
                    away_team = team_name
            
            # Check for odds
            odds_list = comp.get('odds', [])
            if odds_list:
                for odds in odds_list:
                    over_under = odds.get('overUnder')
                    if over_under:
                        games_with_odds.append({
                            'matchup': f"{away_team} @ {home_team}",
                            'over_under': over_under,
                            'odds_details': odds
                        })
                        print(f"   📊 {away_team} @ {home_team}: O/U {over_under}")
                        break
        
        print(f"\n✅ Found {len(games_with_odds)} games with real over/under totals")
        
        if games_with_odds:
            print("\n📋 Sample odds structure:")
            print(json.dumps(games_with_odds[0]['odds_details'], indent=2))
        
        return games_with_odds
        
    except Exception as e:
        print(f"❌ ESPN API test failed: {e}")
        return []

if __name__ == "__main__":
    test_espn_odds()
