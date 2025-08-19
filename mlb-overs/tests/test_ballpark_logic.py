#!/usr/bin/env python3
"""Test ballpark factor logic directly."""

import pandas as pd
import numpy as np

def main():
    # Test the ballpark factor logic directly
    test_data = [
        {'venue_name': 'Coors Field', 'ballpark': None},
        {'venue_name': 'Oracle Park', 'ballpark': None},
        {'venue_name': 'Kauffman Stadium', 'ballpark': None}
    ]
    
    df = pd.DataFrame(test_data)
    
    ballpark_factors = {
        'Coors Field': {'run': 1.15, 'hr': 1.25},
        'Oracle Park': {'run': 0.96, 'hr': 0.88},
        'Kauffman Stadium': {'run': 0.97, 'hr': 0.93},
    }
    
    def get_ballpark_factor(row, factor_type):
        park_name = row.get('venue_name') or row.get('ballpark') or row.get('venue')
        if pd.isna(park_name) or park_name is None:
            return 1.0
        if park_name in ballpark_factors:
            return ballpark_factors[park_name][factor_type]
        return 1.0
    
    df['run_factor'] = df.apply(lambda row: get_ballpark_factor(row, 'run'), axis=1)
    df['hr_factor'] = df.apply(lambda row: get_ballpark_factor(row, 'hr'), axis=1)
    
    print('Test ballpark factor application:')
    print(df.to_string())
    print(f'Run factor std: {df["run_factor"].std():.3f}')
    print(f'HR factor std: {df["hr_factor"].std():.3f}')
    
    print('\nThe logic works - variance should be > 0')
    print('Problem must be that ballpark factors are being set BEFORE the enhanced pipeline')

if __name__ == "__main__":
    main()
