"""
Check the actual schema of enhanced_games table
"""

import psycopg2

def check_schema():
    """Check the actual schema"""
    
    conn = psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser',
        password='mlbpass'
    )
    cursor = conn.cursor()
    
    # Get column names
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'enhanced_games'
        ORDER BY ordinal_position
    """)
    
    columns = cursor.fetchall()
    
    print("📋 ENHANCED_GAMES TABLE SCHEMA:")
    print("=" * 50)
    for col_name, data_type in columns:
        print(f"{col_name:<25} | {data_type}")
    
    # Get a sample row to see what data we have
    cursor.execute("""
        SELECT * FROM enhanced_games 
        WHERE date = '2025-08-20'
        LIMIT 1
    """)
    
    sample = cursor.fetchone()
    
    print(f"\n📊 SAMPLE DATA:")
    print("=" * 50)
    for i, (col_name, _) in enumerate(columns):
        value = sample[i] if sample and i < len(sample) else 'N/A'
        print(f"{col_name:<25} | {value}")
    
    conn.close()

if __name__ == "__main__":
    check_schema()
