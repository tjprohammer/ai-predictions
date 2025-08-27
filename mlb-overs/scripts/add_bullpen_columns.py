import psycopg2

def add_bullpen_columns():
    """Add bullpen stats columns to enhanced_games table"""
    conn = psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser',
        password='mlbpass'
    )
    cursor = conn.cursor()
    
    # Check existing bullpen columns
    cursor.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'enhanced_games' 
    AND column_name LIKE '%bp%'
    ORDER BY column_name
    """)
    existing_cols = [row[0] for row in cursor.fetchall()]
    
    print("Existing bullpen columns:")
    for col in existing_cols:
        print(f"  {col}")
    
    # Define needed bullpen columns
    needed_cols = [
        'home_bp_ip',    # Home bullpen innings pitched
        'home_bp_er',    # Home bullpen earned runs
        'home_bp_k',     # Home bullpen strikeouts
        'home_bp_bb',    # Home bullpen walks
        'home_bp_h',     # Home bullpen hits allowed
        'away_bp_ip',    # Away bullpen innings pitched
        'away_bp_er',    # Away bullpen earned runs
        'away_bp_k',     # Away bullpen strikeouts
        'away_bp_bb',    # Away bullpen walks
        'away_bp_h'      # Away bullpen hits allowed
    ]
    
    print("\nAdding missing bullpen columns:")
    for col in needed_cols:
        if col not in existing_cols:
            try:
                if col.endswith('_ip'):
                    # Innings pitched as decimal
                    cursor.execute(f"ALTER TABLE enhanced_games ADD COLUMN {col} DECIMAL(4,1) DEFAULT 0")
                else:
                    # All other stats as integers
                    cursor.execute(f"ALTER TABLE enhanced_games ADD COLUMN {col} INTEGER DEFAULT 0")
                print(f"  ✅ Added {col}")
            except Exception as e:
                print(f"  ❌ Failed to add {col}: {e}")
        else:
            print(f"  ⚪ {col} already exists")
    
    conn.commit()
    conn.close()
    print("\n✅ Bullpen columns added successfully!")

if __name__ == "__main__":
    add_bullpen_columns()
