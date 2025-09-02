#!/usr/bin/env python3
"""
Daily Pitcher Rolling Stats Ingestor
====================================

Updates the pitcher_daily_rolling table with recent pitching performance
for use in prediction models. Calculates 30-day rolling averages for ERA,
WHIP, K/9, BB/9, etc.
"""

import requests
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import argparse
import sys
import os

# Fix encoding issues on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

def get_connection():
    """Get database connection"""
    return psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser',
        password='mlbpass'
    )

def get_active_pitchers(target_date):
    """Get list of pitchers who are scheduled to start on target date"""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT home_sp_id, away_sp_id 
            FROM enhanced_games 
            WHERE date = %s 
            AND home_sp_id IS NOT NULL 
            AND away_sp_id IS NOT NULL
        ''', (target_date,))
        
        rows = cursor.fetchall()
        pitcher_ids = set()
        for row in rows:
            if row[0]:  # home_sp_id
                pitcher_ids.add(int(row[0]))
            if row[1]:  # away_sp_id
                pitcher_ids.add(int(row[1]))
        
        return list(pitcher_ids)
    finally:
        conn.close()

def calculate_rolling_stats(pitcher_id, target_date, days_back=30):
    """Calculate rolling stats for a pitcher using MLB game log data"""
    try:
        # Parse target_date flexibly (handle both MM-DD-YYYY and YYYY-MM-DD formats)
        try:
            # Try YYYY-MM-DD first
            end_date = datetime.strptime(target_date, '%Y-%m-%d')
        except ValueError:
            try:
                # Try MM-DD-YYYY format
                end_date = datetime.strptime(target_date, '%m-%d-%Y')
            except ValueError:
                # Try other common formats
                for fmt in ['%Y/%m/%d', '%m/%d/%Y', '%Y.%m.%d', '%m.%d.%Y']:
                    try:
                        end_date = datetime.strptime(target_date, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Unable to parse date format: {target_date}")
        
        start_date = end_date - timedelta(days=days_back)
        
        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
        params = {
            'stats': 'gameLog',
            'gameType': 'R',
            'season': '2025',
            'startDate': start_date.strftime('%m/%d/%Y'),
            'endDate': end_date.strftime('%m/%d/%Y')
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
            
        data = response.json()
        
        # Extract game log stats
        stats = data.get('stats', [])
        if not stats or not stats[0].get('splits'):
            return None
            
        games = stats[0]['splits']
        
        # Calculate totals for rolling period
        total_ip = 0.0
        total_er = 0
        total_h = 0
        total_bb = 0
        total_k = 0
        total_hr = 0
        game_count = 0
        
        for game in games:
            stat = game.get('stat', {})
            ip_str = stat.get('inningsPitched', '0.0')
            
            # Convert innings pitched (like "6.1" to 6.33)
            if '.' in str(ip_str):
                whole, partial = str(ip_str).split('.')
                ip_decimal = float(whole) + (float(partial) / 3.0)
            else:
                ip_decimal = float(ip_str or 0)
            
            if ip_decimal > 0:  # Only count games where pitcher actually pitched
                total_ip += ip_decimal
                total_er += int(stat.get('earnedRuns', 0))
                total_h += int(stat.get('hits', 0))
                total_bb += int(stat.get('baseOnBalls', 0))
                total_k += int(stat.get('strikeOuts', 0))
                total_hr += int(stat.get('homeRuns', 0))
                game_count += 1
        
        if total_ip == 0 or game_count == 0:
            return None
            
        # Calculate rates
        era = (total_er * 9.0) / total_ip
        whip = (total_h + total_bb) / total_ip
        k_per_9 = (total_k * 9.0) / total_ip
        bb_per_9 = (total_bb * 9.0) / total_ip
        hr_per_9 = (total_hr * 9.0) / total_ip
        
        return {
            'pitcher_id': pitcher_id,
            'stat_date': target_date,
            'gs': game_count,
            'ip': round(total_ip, 1),
            'er': total_er,
            'bb': total_bb,
            'k': total_k,
            'h': total_h,
            'hr': total_hr,
            'era': round(era, 2),
            'whip': round(whip, 3),
            'k_per_9': round(k_per_9, 1),
            'bb_per_9': round(bb_per_9, 1),
            'hr_per_9': round(hr_per_9, 1)
        }
        
    except Exception as e:
        print(f"   ⚠️ Error calculating stats for pitcher {pitcher_id}: {e}")
        return None

def update_rolling_stats(target_date=None):
    """Update pitcher_daily_rolling table with latest stats"""
    if not target_date:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"📊 Updating pitcher rolling stats for {target_date}")
    print("=" * 50)
    
    # Get active pitchers for the target date
    pitcher_ids = get_active_pitchers(target_date)
    print(f"🔍 Found {len(pitcher_ids)} active pitchers to update")
    
    if not pitcher_ids:
        print("⚠️ No active pitchers found for target date")
        return 0
    
    conn = get_connection()
    updated_count = 0
    
    try:
        cursor = conn.cursor()
        
        for pitcher_id in pitcher_ids:
            print(f"   📈 Processing pitcher {pitcher_id}...")
            
            # Calculate rolling stats
            stats = calculate_rolling_stats(pitcher_id, target_date)
            
            if stats:
                # Upsert into pitcher_daily_rolling
                upsert_sql = '''
                    INSERT INTO pitcher_daily_rolling (
                        pitcher_id, stat_date, gs, ip, er, bb, k, h, hr,
                        era, whip, k_per_9, bb_per_9, hr_per_9
                    ) VALUES (
                        %(pitcher_id)s, %(stat_date)s, %(gs)s, %(ip)s, %(er)s, 
                        %(bb)s, %(k)s, %(h)s, %(hr)s, %(era)s, %(whip)s, 
                        %(k_per_9)s, %(bb_per_9)s, %(hr_per_9)s
                    )
                    ON CONFLICT (pitcher_id, stat_date) 
                    DO UPDATE SET
                        gs = EXCLUDED.gs,
                        ip = EXCLUDED.ip,
                        er = EXCLUDED.er,
                        bb = EXCLUDED.bb,
                        k = EXCLUDED.k,
                        h = EXCLUDED.h,
                        hr = EXCLUDED.hr,
                        era = EXCLUDED.era,
                        whip = EXCLUDED.whip,
                        k_per_9 = EXCLUDED.k_per_9,
                        bb_per_9 = EXCLUDED.bb_per_9,
                        hr_per_9 = EXCLUDED.hr_per_9
                '''
                
                cursor.execute(upsert_sql, stats)
                updated_count += 1
                
                print(f"      ✅ Updated: ERA {stats['era']}, WHIP {stats['whip']}, {stats['gs']} games")
            else:
                print(f"      ⚠️ No recent stats found for pitcher {pitcher_id}")
        
        conn.commit()
        
    except Exception as e:
        print(f"❌ Error updating rolling stats: {e}")
        conn.rollback()
    finally:
        conn.close()
    
    print(f"\n✅ Updated rolling stats for {updated_count}/{len(pitcher_ids)} pitchers")
    return updated_count

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Update pitcher rolling stats')
    parser.add_argument('--target-date', type=str, help='Target date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    result = update_rolling_stats(args.target_date)
    sys.exit(0 if result > 0 else 1)

if __name__ == "__main__":
    main()
