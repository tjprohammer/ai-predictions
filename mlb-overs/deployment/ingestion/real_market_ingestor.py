#!/usr/bin/env python3
"""
WORKING Market Data Ingestor
===========================

Collects real betting market totals (over/under lines) for today's games.
This replaces the placeholder market ingestor.
"""

import requests
import pandas as pd
from sqlalchemy import create_engine, text
import os
import json
import argparse
from datetime import datetime

def get_engine():
    """Get database engine"""
    url = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    return create_engine(url)

def _table_columns(engine, table_name):
    """Get list of column names for a table."""
    try:
        result = engine.execute(text(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """))
        return [row[0] for row in result]
    except Exception as e:
        print(f"Warning: Could not get columns for {table_name}: {e}")
        return []

def get_market_totals_from_api(target_date=None, include_live=False):
    """
    Try to get market totals from various sources.
    Priority: The Odds API -> ESPN -> Intelligent estimates
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")
    
    print("🔍 Attempting to fetch market totals...")
    
    # First try The Odds API (real betting data)
    real_odds = get_odds_api_data(target_date, include_live)
    if real_odds:
        print(f"✅ Using real market data from The Odds API ({len(real_odds)} games)")
        return real_odds
    
    # Fallback to ESPN (free but limited)
    espn_odds = get_espn_market_data(target_date)
    if espn_odds:
        print(f"✅ Using ESPN market data ({len(espn_odds)} games)")
        return espn_odds
    
    # Final fallback to intelligent estimates
    print("⚠️  No real market data available, using intelligent estimates")
    return get_estimated_market_data(target_date)

def get_odds_api_data(target_date, include_live=False):
    """Get real market totals from The Odds API - PREGAME ONLY by default"""
    try:
        from datetime import datetime, timezone
        
        api_key = os.environ.get('THE_ODDS_API_KEY')
        if not api_key:
            print("   ℹ️  No Odds API key found (set THE_ODDS_API_KEY env var)")
            return None
        
        print(f"   🔑 Using The Odds API key: {api_key[:8]}...")
        
        # Parse target date for filtering
        target_dt = datetime.fromisoformat(f"{target_date}T00:00:00+00:00").date()
        now_utc = datetime.now(timezone.utc)
        
        url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': 'totals',
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        odds_data = response.json()
        market_data = []
        skipped_live = 0
        
        # Map API data to our games - include game_time for DH disambiguation
        engine = get_engine()
        with engine.begin() as conn:
            # Check if game_time column exists
            eg_columns = _table_columns(engine, 'enhanced_games')
            time_col = 'game_time' if 'game_time' in eg_columns else "'12:00'::text as game_time"
            
            todays_games = pd.read_sql(text(f"""
                SELECT game_id, home_team, away_team, venue_name, {time_col}
                FROM enhanced_games 
                WHERE date = :target_date
                AND game_id IS NOT NULL
                ORDER BY game_id
            """), conn, params={"target_date": target_date})
        
        # Track used API games to prevent reuse in doubleheaders
        used_api_games = set()
        
        # Process each API game - filter live games unless requested
        for api_idx, api_game in enumerate(odds_data):
            if api_idx in used_api_games:
                continue  # Skip already matched API games
                
            # Check game start time
            commence_iso = api_game.get("commence_time")
            try:
                ct = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
            except Exception:
                ct = None
            
            # Skip games that are live (unless include_live=True) or wrong date
            is_live = ct and ct <= now_utc
            if not ct or ct.date() != target_dt:
                continue  # Wrong date, always skip
            
            if is_live and not include_live:
                skipped_live += 1
                continue  # Live game, skip unless explicitly requested
            
            away_team = api_game.get('away_team', '')
            home_team = api_game.get('home_team', '')
            
            # Time-based doubleheader disambiguation
            api_hour = ct.hour if ct else 12  # Default to noon if no time
            
            # Find potential team matches first
            team_matches = []
            for _, db_game in todays_games.iterrows():
                if (db_game['home_team'] in home_team or home_team in db_game['home_team']) and \
                   (db_game['away_team'] in away_team or away_team in db_game['away_team']):
                    team_matches.append(db_game)
            
            if not team_matches:
                continue  # No team match found
            
            # If multiple matches (doubleheader), use time-based matching
            best_match = None
            if len(team_matches) == 1:
                best_match = team_matches[0]
            else:
                # Doubleheader: match by game time proximity
                best_time_diff = float('inf')
                for db_game in team_matches:
                    try:
                        # Parse database game time (format: "13:05" or "1:05 PM")
                        db_time_str = db_game.get('game_time', '12:00')
                        if ':' in db_time_str:
                            if 'PM' in db_time_str.upper() or 'AM' in db_time_str.upper():
                                db_hour = datetime.strptime(db_time_str.strip(), '%I:%M %p').hour
                            else:
                                db_hour = int(db_time_str.split(':')[0])
                        else:
                            db_hour = 12  # Fallback
                        
                        time_diff = abs(api_hour - db_hour)
                        if time_diff < best_time_diff:
                            best_time_diff = time_diff
                            best_match = db_game
                    except Exception:
                        continue  # Skip games with bad time data
            
            if best_match is not None:
                # Get the best total and odds from bookmakers
                best_total = None
                over_odds = None
                under_odds = None
                bookmaker_name = None
                
                for bookmaker in api_game.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'totals':
                            temp_total = None
                            temp_over_odds = None
                            temp_under_odds = None
                            
                            for outcome in market.get('outcomes', []):
                                point = outcome.get('point')
                                price = outcome.get('price')
                                name = outcome.get('name')
                                
                                if point and name:
                                    if name == 'Over':
                                        temp_total = float(point)
                                        temp_over_odds = int(price)
                                    elif name == 'Under':
                                        temp_under_odds = int(price)
                            
                            if temp_total and temp_over_odds and temp_under_odds:
                                best_total = temp_total
                                over_odds = temp_over_odds
                                under_odds = temp_under_odds
                                bookmaker_name = bookmaker.get('title', 'Unknown')
                                break
                    if best_total:
                        break
                
                if best_total:
                    market_data.append({
                        'game_id': best_match['game_id'],
                        'home_team': best_match['home_team'],
                        'away_team': best_match['away_team'],
                        'market_total': best_total,
                        'over_odds': over_odds,
                        'under_odds': under_odds,
                        'source': f'real_api_{bookmaker_name}'
                    })
                    print(f"   📊 {best_match['away_team']} @ {best_match['home_team']}: O/U {best_total} (O:{over_odds:+d}/U:{under_odds:+d}) [{bookmaker_name}]")
                    used_api_games.add(api_idx)  # Mark API game as used
        
        # Check API usage
        remaining = response.headers.get('x-requests-remaining')
        if remaining:
            print(f"   📈 API requests remaining: {remaining}")
        
        # Report filtering results
        if skipped_live > 0:
            print(f"   🚫 Skipped {skipped_live} live/started games (pregame only)")
        
        return market_data if market_data else None
        
    except Exception as e:
        print(f"   ❌ The Odds API failed: {e}")
        return None

def get_espn_market_data(target_date):
    """Get market totals from ESPN API (free)"""
    try:
        # ESPN doesn't have good historical data for older dates, so skip for non-current dates
        current_date = datetime.now().strftime("%Y-%m-%d")
        if target_date != current_date:
            print(f"   ℹ️  ESPN doesn't provide historical data for {target_date}")
            return None
            
        print("   🎯 Trying ESPN API...")
        
        url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        espn_data = response.json()
        market_data = []
        
        engine = get_engine()
        with engine.begin() as conn:
            todays_games = pd.read_sql(text("""
                SELECT game_id, home_team, away_team, venue_name
                FROM enhanced_games 
                WHERE date = :target_date
                AND game_id IS NOT NULL
                ORDER BY game_id
            """), conn, params={"target_date": target_date})
        
        for event in espn_data.get('events', []):
            competitions = event.get('competitions', [])
            if not competitions:
                continue
                
            comp = competitions[0]
            competitors = comp.get('competitors', [])
            
            # Get team names
            espn_home = None
            espn_away = None
            for competitor in competitors:
                team_name = competitor.get('team', {}).get('displayName', '')
                if competitor.get('homeAway') == 'home':
                    espn_home = team_name
                else:
                    espn_away = team_name
            
            if not espn_home or not espn_away:
                continue
            
            # Get over/under
            odds_list = comp.get('odds', [])
            espn_total = None
            for odds in odds_list:
                total = odds.get('overUnder')
                if total:
                    espn_total = float(total)
                    break
            
            if espn_total:
                # Match with our database games
                for _, db_game in todays_games.iterrows():
                    if (db_game['home_team'] in espn_home or espn_home in db_game['home_team']) and \
                       (db_game['away_team'] in espn_away or espn_away in db_game['away_team']):
                        
                        market_data.append({
                            'game_id': db_game['game_id'],
                            'home_team': db_game['home_team'],
                            'away_team': db_game['away_team'],
                            'market_total': espn_total,
                            'source': 'espn_api'
                        })
                        print(f"   📊 {db_game['away_team']} @ {db_game['home_team']}: O/U {espn_total} (ESPN)")
                        break
        
        return market_data if market_data else None
        
    except Exception as e:
        print(f"   ❌ ESPN API failed: {e}")
        return None

def get_estimated_market_data(target_date):
    """Get estimated market totals using intelligent algorithms"""
    engine = get_engine()
    market_data = []
    
    try:
        with engine.begin() as conn:
            # Get target date games that need market totals
            todays_games = pd.read_sql(text("""
                SELECT game_id, home_team, away_team, venue_name
                FROM enhanced_games 
                WHERE date = :target_date
                AND game_id IS NOT NULL
                ORDER BY game_id
            """), conn, params={"target_date": target_date})
            
            print(f"Found {len(todays_games)} games needing market totals")
            
            # Generate realistic market totals based on teams and venues
            for _, game in todays_games.iterrows():
                market_total = estimate_market_total(game['home_team'], game['away_team'], game['venue_name'])
                
                market_data.append({
                    'game_id': game['game_id'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'market_total': market_total,
                    'source': 'estimated'  # Mark as estimated
                })
                
                print(f"📊 {game['away_team']} @ {game['home_team']}: O/U {market_total}")
    
    except Exception as e:
        print(f"❌ Error getting market data: {e}")
        return []
    
    return market_data

def estimate_market_total(home_team, away_team, venue):
    """
    Estimate realistic market total based on teams, venue, and current MLB betting patterns.
    Updated with more accurate 2025 season data and realistic sportsbook totals.
    """
    
    # Start with base total
    base_total = 8.5
    
    # === TEAM SCORING TENDENCIES (2025 season) ===
    # High-scoring teams (typically see 9.5-11+ totals)
    elite_offense_teams = [
        'Houston Astros', 'Atlanta Braves', 'Los Angeles Dodgers', 
        'New York Yankees', 'Boston Red Sox', 'Texas Rangers',
        'Philadelphia Phillies', 'San Diego Padres'
    ]
    
    # Good offensive teams (typically see 9.0-10.5 totals)
    good_offense_teams = [
        'Toronto Blue Jays', 'Baltimore Orioles', 'Seattle Mariners',
        'Arizona Diamondbacks', 'Minnesota Twins', 'New York Mets'
    ]
    
    # Poor offensive teams (typically see 7.5-8.5 totals)
    poor_offense_teams = [
        'Detroit Tigers', 'Chicago White Sox', 'Miami Marlins',
        'Pittsburgh Pirates', 'Kansas City Royals', 'Oakland Athletics',
        'Colorado Rockies'  # Pitching struggles offset Coors advantage
    ]
    
    # === VENUE EFFECTS (Major Impact) ===
    # Extreme hitter parks
    if venue and 'Coors Field' in venue:
        base_total = 11.5  # Coors is always high due to altitude
    elif venue and any(park in venue for park in ['Fenway Park', 'Yankee Stadium', 'Great American Ball Park']):
        base_total += 1.0
    elif venue and any(park in venue for park in ['Minute Maid Park', 'Globe Life Field', 'Citizens Bank Park']):
        base_total += 0.5
    
    # Extreme pitcher parks
    elif venue and any(park in venue for park in ['Petco Park', 'Oracle Park', 'Tropicana Field']):
        base_total -= 1.0
    elif venue and any(park in venue for park in ['Kauffman Stadium', 'Comerica Park', 'Marlins Park']):
        base_total -= 0.5
    
    # === TEAM COMBINATION ADJUSTMENTS ===
    home_offense_tier = 0
    away_offense_tier = 0
    
    if home_team in elite_offense_teams:
        home_offense_tier = 2
    elif home_team in good_offense_teams:
        home_offense_tier = 1
    elif home_team in poor_offense_teams:
        home_offense_tier = -1
        
    if away_team in elite_offense_teams:
        away_offense_tier = 2
    elif away_team in good_offense_teams:
        away_offense_tier = 1
    elif away_team in poor_offense_teams:
        away_offense_tier = -1
    
    # Apply team adjustments
    total_offense_adjustment = (home_offense_tier + away_offense_tier) * 0.25
    base_total += total_offense_adjustment
    
    # === SPECIFIC GAME ADJUSTMENTS ===
    # Both teams are poor offensively
    if home_team in poor_offense_teams and away_team in poor_offense_teams:
        base_total -= 0.5
    
    # At least one elite offense
    if home_team in elite_offense_teams or away_team in elite_offense_teams:
        base_total += 0.25
    
    # Special case: Colorado Rockies (poor team but at Coors)
    if 'Colorado Rockies' in [home_team, away_team] and venue and 'Coors Field' in venue:
        base_total = 12.0  # Coors effect dominates team quality
    
    # Ensure realistic range (MLB totals rarely below 7.0 or above 13.0)
    base_total = max(7.0, min(13.0, base_total))
    
    # Round to nearest 0.5 (standard sportsbook practice)
    return round(base_total * 2) / 2

def _snap_to_half(x):
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    # snap to x.0 / x.5 like books quote
    return round(v * 2) / 2.0

def update_market_totals(market_data, target_date):
    """Update enhanced_games for the given target_date; also log into totals_odds."""
    if not market_data:
        return 0

    engine = get_engine()
    updated_count = 0

    # de-dupe by game_id
    seen = {}
    for m in market_data:
        seen[str(m["game_id"])] = m
    market_data = list(seen.values())
    print(f"📝 Processing {len(market_data)} unique games (removed duplicates)")

    # normalize totals/odds
    for m in market_data:
        m["game_id"] = str(m["game_id"])
        m["market_total"] = _snap_to_half(m.get("market_total"))
        if m.get("over_odds") is not None:
            m["over_odds"] = int(m["over_odds"])
        if m.get("under_odds") is not None:
            m["under_odds"] = int(m["under_odds"])

    try:
        # ensure columns exist (idempotent) - separate transaction
        with engine.begin() as conn:
            try:
                conn.execute(text("ALTER TABLE enhanced_games ADD COLUMN market_total DECIMAL(4,1)"))
                print("✅ Added market_total column to enhanced_games")
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE enhanced_games ADD COLUMN over_odds INTEGER"))
                conn.execute(text("ALTER TABLE enhanced_games ADD COLUMN under_odds INTEGER"))
                print("✅ Added odds columns to enhanced_games")
            except Exception:
                pass

        # create history table - separate transaction  
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS totals_odds (
                  game_id      varchar NOT NULL,
                  "date"       date    NOT NULL,
                  book         text    NOT NULL,
                  total        numeric NOT NULL,
                  over_odds    integer,
                  under_odds   integer,
                  collected_at timestamp NOT NULL DEFAULT now(),
                  PRIMARY KEY (game_id, "date", book, total, collected_at)
                )
            """))

        # now do the updates - separate transaction
        with engine.begin() as conn:
            upd = text("""
                UPDATE enhanced_games AS eg
                   SET market_total = :mt,
                       over_odds    = :oo,
                       under_odds   = :uo
                 WHERE eg.game_id = :gid
                   AND eg.date = :d
            """)

            ins_hist = text("""
                INSERT INTO totals_odds (game_id,date,book,total,over_odds,under_odds)
                VALUES (:gid, :d, :book, :mt, :oo, :uo)
            """)

            books = []
            for m in market_data:
                params = {
                    "gid": m["game_id"],
                    "d":   target_date,
                    "mt":  m["market_total"],
                    "oo":  m.get("over_odds"),
                    "uo":  m.get("under_odds"),
                    "book": (m.get("source") or "").replace("real_api_", "") or "consensus"
                }
                rc = conn.execute(upd, params).rowcount
                if rc > 0:
                    updated_count += 1
                    books.append(params["book"])
                    conn.execute(ins_hist, params)
                    odds_info = ""
                    if params["oo"] is not None and params["uo"] is not None:
                        odds_info = f" (O:{params['oo']:+d}/U:{params['uo']:+d})"
                    print(f"[MARKET] Updated {m['away_team']} @ {m['home_team']}: O/U {params['mt']}{odds_info}")
                else:
                    print(f"⚠️  No game found to update for game_id {m['game_id']} on {target_date}")

        if updated_count:
            if books:
                uniq = sorted({b for b in books if b})
                print(f"📚 Books recorded: {', '.join(uniq)}")
        return updated_count

    except Exception as e:
        print(f"❌ Error updating market totals: {e}")
        try:
            engine.dispose()
        except:
            pass
        return 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Collect market data for MLB games - PREGAME ONLY by default")
    parser.add_argument("--date", "--target-date", dest="target_date", 
                       help="YYYY-MM-DD (defaults to today)")
    parser.add_argument("--include-live", action="store_true",
                       help="Include live/in-progress games (default: pregame only)")
    args = parser.parse_args()
    
    target_date = args.target_date or datetime.now().strftime("%Y-%m-%d")
    
    print("[MARKET] Collecting Market Data (Betting Totals)")
    print("=" * 40)
    print(f"📅 Target date: {target_date}")
    if not args.include_live:
        print("🎯 Mode: PREGAME ONLY (use --include-live for live games)")
    else:
        print("⚠️  Mode: INCLUDING LIVE GAMES")
    
    # Clear any existing bad transactions
    try:
        engine = get_engine()
        engine.dispose()
        print("🔄 Cleared database connection state")
    except:
        pass
    
    market_data = get_market_totals_from_api(target_date, args.include_live)
    
    if market_data:
        updated = update_market_totals(market_data, target_date)
        print(f"✅ Updated market totals for {updated} games")
        
        # Save market data to file for reference
        with open('daily_market_totals.json', 'w') as f:
            json.dump(market_data, f, indent=2)
        print("💾 Saved market data to daily_market_totals.json")
    else:
        print("❌ No market data to update")

if __name__ == "__main__":
    main()
