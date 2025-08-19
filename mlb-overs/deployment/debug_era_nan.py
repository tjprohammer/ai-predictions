#!/usr/bin/env python3
"""Debug script to investigate ERA NaN issue."""

from sqlalchemy import create_engine, text
import pandas as pd
import os

def main():
    # Database connection
    db_url = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
    engine = create_engine(db_url)
    
    # Check the Athletics home pitcher for 2025-08-16
    print("🔍 Investigating Athletics home pitcher for 2025-08-16...")
    
    query = text("""
        SELECT home_sp_id, home_sp_name 
        FROM enhanced_games 
        WHERE date = '2025-08-16' AND home_team = 'Athletics'
    """)
    
    result = pd.read_sql(query, engine)
    
    if len(result) == 0:
        print("❌ No Athletics game found for 2025-08-16")
        return
    
    pitcher_id = int(result['home_sp_id'].iloc[0])
    pitcher_name = result['home_sp_name'].iloc[0]
    print(f"✅ Found: {pitcher_name} (ID: {pitcher_id})")
    
    # Check their rolling stats
    print(f"\n🔍 Checking rolling stats for {pitcher_name}...")
    
    query2 = text("""
        SELECT stat_date, ip, er, era, whip 
        FROM pitcher_daily_rolling 
        WHERE pitcher_id = :pid 
        ORDER BY stat_date DESC 
        LIMIT 5
    """)
    
    rolling = pd.read_sql(query2, engine, params={'pid': pitcher_id})
    
    if len(rolling) == 0:
        print("❌ No rolling stats found")
        return
    
    print(f"✅ Found {len(rolling)} rolling stat records:")
    print("\nRolling stats:")
    for _, row in rolling.iterrows():
        date_val = row['stat_date']
        ip_val = row['ip']
        er_val = row['er']
        era_val = row['era']
        whip_val = row['whip']
        print(f"  {date_val}: IP={ip_val}, ER={er_val}, ERA={era_val}, WHIP={whip_val}")
    
    # Check the latest stats for issues
    latest = rolling.iloc[0]
    era_val = latest['era']
    ip_val = latest['ip']
    er_val = latest['er']
    whip_val = latest['whip']
    
    print(f"\n📊 Latest stats analysis:")
    print(f"   ERA: {era_val}")
    print(f"   IP:  {ip_val}")
    print(f"   ER:  {er_val}")
    print(f"   WHIP: {whip_val}")
    
    # Check for potential issues
    issues_found = []
    
    if era_val == 0.0:
        issues_found.append(f"ERA=0.00 (could cause downstream NaN in divisions)")
    
    if ip_val < 1.0:
        issues_found.append(f"Low innings pitched: {ip_val}")
    
    if pd.isna(era_val):
        issues_found.append("ERA is NaN")
    
    if pd.isna(whip_val):
        issues_found.append("WHIP is NaN")
    
    if issues_found:
        print(f"\n❌ Potential issues found:")
        for issue in issues_found:
            print(f"   • {issue}")
    else:
        print(f"\n✅ No obvious data quality issues in rolling stats")
    
    # Test downstream calculations
    print(f"\n🧮 Testing downstream calculations...")
    
    # Simulate what happens in feature engineering
    test_era = era_val
    if test_era == 0.0:
        print(f"   ERA=0.00 detected - testing mathematical operations:")
        print(f"   ERA + 1 = {test_era + 1}")
        print(f"   1 / ERA = ", end="")
        try:
            result = 1 / test_era
            print(f"{result}")
        except ZeroDivisionError:
            print("ZeroDivisionError!")
        
        print(f"   ERA * 2 = {test_era * 2}")
        print(f"   ERA - 3.60 = {test_era - 3.60}")
        
        # Check if this could cause NaN in combined calculations
        print(f"   Testing combined calculations:")
        combined_test = (test_era + 3.60) / 2  # Example combined calculation
        print(f"   (ERA + 3.60) / 2 = {combined_test}")
        
        if pd.isna(combined_test):
            print(f"   ❌ Combined calculation produces NaN!")
        else:
            print(f"   ✅ Combined calculation OK")

if __name__ == "__main__":
    main()
