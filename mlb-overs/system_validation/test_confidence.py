#!/usr/bin/env python3

from sqlalchemy import create_engine, text
import os

url = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(url)

with engine.begin() as conn:
    result = conn.execute(text("""
        SELECT 
            lpp.game_id,
            lpp.recommendation,
            lpp.p_over,
            lpp.p_under,
            ROUND(
                CAST(
                    CASE 
                        WHEN lpp.recommendation = 'OVER' THEN lpp.p_over * 100
                        WHEN lpp.recommendation = 'UNDER' THEN lpp.p_under * 100
                        ELSE GREATEST(lpp.p_over, lpp.p_under) * 100
                    END AS NUMERIC
                ), 1
            ) as confidence
        FROM latest_probability_predictions lpp
        WHERE lpp.game_date = CURRENT_DATE
        AND lpp.game_id IN ('776700', '776707')
        ORDER BY lpp.game_id
    """)).fetchall()
    
    print("✅ CONFIDENCE CALCULATION FIXED:")
    for row in result:
        print(f"Game {row.game_id}: {row.recommendation} bet - Confidence: {row.confidence}%")
        print(f"  P(Over): {row.p_over*100:.1f}%, P(Under): {row.p_under*100:.1f}%")
        print()
