#!/usr/bin/env python3

import psycopg2

# Check database schema
conn = psycopg2.connect(host='localhost', database='mlb', user='mlbuser', password='mlbpass')
cursor = conn.cursor()

print("📊 CHECKING DATABASE SCHEMA RELATIONSHIPS")
print("=" * 60)

# Check all pitcher-related tables
cursor.execute("""
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND (table_name LIKE '%pitcher%' OR table_name LIKE '%enhanced%')
ORDER BY table_name;
""")

tables = cursor.fetchall()
print("🔍 PITCHER AND ENHANCED TABLES:")
for table in tables:
    print(f"   {table[0]}")

print("\n📋 ENHANCED_GAMES TABLE COLUMNS:")
cursor.execute("""
SELECT column_name, data_type, is_nullable
FROM information_schema.columns 
WHERE table_name = 'enhanced_games' 
  AND (column_name LIKE '%era%' OR column_name LIKE '%sp_%' OR column_name LIKE '%pitcher%')
ORDER BY column_name;
""")

columns = cursor.fetchall()
for col in columns:
    print(f"   {col[0]:<30} {col[1]:<15} {'NULL' if col[2] == 'YES' else 'NOT NULL'}")

print("\n📋 PITCHER_DAILY_ROLLING TABLE STRUCTURE:")
cursor.execute("""
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'pitcher_daily_rolling'
ORDER BY ordinal_position;
""")

rolling_columns = cursor.fetchall()
if rolling_columns:
    for col in rolling_columns:
        print(f"   {col[0]:<30} {col[1]}")
else:
    print("   ❌ Table doesn't exist or no columns found")

print("\n🔗 CHECKING RELATIONSHIPS:")
# Check if there are foreign keys
cursor.execute("""
SELECT 
    tc.table_name, 
    kcu.column_name, 
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name 
FROM information_schema.table_constraints AS tc 
JOIN information_schema.key_column_usage AS kcu
  ON tc.constraint_name = kcu.constraint_name
  AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
  ON ccu.constraint_name = tc.constraint_name
  AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY' 
  AND (tc.table_name LIKE '%enhanced%' OR tc.table_name LIKE '%pitcher%');
""")

fks = cursor.fetchall()
if fks:
    for fk in fks:
        print(f"   {fk[0]}.{fk[1]} -> {fk[2]}.{fk[3]}")
else:
    print("   ❌ No foreign keys found between pitcher and enhanced tables")

print("\n📊 SAMPLE ERA DATA FROM ENHANCED_GAMES:")
cursor.execute("""
SELECT 
    date,
    home_team,
    away_team,
    home_sp_id,
    away_sp_id,
    home_sp_season_era,
    away_sp_season_era
FROM enhanced_games 
WHERE date >= '2025-08-20'
ORDER BY date DESC
LIMIT 5;
""")

sample_data = cursor.fetchall()
for row in sample_data:
    print(f"   {row[0]} | {row[1]:<20} vs {row[2]:<20} | Home SP: {row[3]} (ERA: {row[5]}) | Away SP: {row[4]} (ERA: {row[6]})")

conn.close()
