from sqlalchemy import create_engine, text

engine = create_engine('postgresql://mlbuser:mlbpass@localhost/mlb')

with engine.connect() as conn:
    result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'enhanced_games' ORDER BY column_name"))
    columns = [row[0] for row in result.fetchall()]
    print('Available columns in enhanced_games:')
    for col in columns:
        print(f'  {col}')
