#!/usr/bin/env python3
"""Quick data sanity check for residuals"""

import pandas as pd
from sqlalchemy import create_engine, text

# Quick data sanity check on residuals
engine = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')

sql = """
SELECT
  COUNT(*) AS n,
  SUM(CASE WHEN ABS(eg.total_runs - eg.market_total) < 1e-9 THEN 1 ELSE 0 END) AS n_equal,
  AVG(eg.total_runs - eg.market_total) AS mean_resid
FROM enhanced_games eg
JOIN game_conditions gc ON gc.game_id::text = eg.game_id::text
WHERE eg.date BETWEEN '2024-04-01' AND '2025-08-25'
  AND eg.total_runs IS NOT NULL
  AND eg.market_total IS NOT NULL;
"""

result = pd.read_sql(text(sql), engine)
print('Data sanity check for residuals:')
print(result.to_string(index=False))

if len(result) > 0:
    n_equal = result['n_equal'].iloc[0]
    total = result['n'].iloc[0]
    
    if n_equal > 0:
        print(f'\nWARNING: {n_equal}/{total} ({100*n_equal/total:.1f}%) games have identical total_runs = market_total')
    else:
        print(f'\nGOOD: No games have identical total_runs = market_total')

# Calculate standard deviation manually
sql2 = """
SELECT (eg.total_runs - eg.market_total) AS resid
FROM enhanced_games eg
JOIN game_conditions gc ON gc.game_id::text = eg.game_id::text
WHERE eg.date BETWEEN '2024-04-01' AND '2025-08-25'
  AND eg.total_runs IS NOT NULL
  AND eg.market_total IS NOT NULL;
"""

resids = pd.read_sql(text(sql2), engine)['resid']
sd_resid = resids.std()
print(f'Residual standard deviation: {sd_resid:.3f}')

if sd_resid < 0.1:
    print('WARNING: Residual standard deviation is very low')
else:
    print('GOOD: Residual standard deviation looks reasonable')

print(f'Sample residuals: {resids.head(10).tolist()}')
