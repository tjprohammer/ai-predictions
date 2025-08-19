#!/usr/bin/env python3
"""Debug the venue name extraction logic."""

import pandas as pd

def main():
    # Reproduce the exact logic from _ensure_ballpark_factors
    test_data = [
        {'venue': None, 'venue_name': 'Coors Field'},
        {'venue': None, 'venue_name': 'Oracle Park'},
    ]
    
    df = pd.DataFrame(test_data)
    
    # Reproduce lines 203-205
    names = df.get('venue')
    print('df.get("venue"):', names)
    print('Type:', type(names))
    
    if names is None:
        names = df.get('venue_name')
        print('df.get("venue_name"):', names)
        print('Type:', type(names))
    
    if names is not None:
        for i in range(len(df)):
            nm = str(names.iloc[i]) if names is not None else None
            print(f'Row {i}: nm = "{nm}"')
    
    # Test BALLPARK_FACTORS lookup
    BALLPARK_FACTORS = {
        "Coors Field": {"run_factor": 1.12, "hr_factor": 1.25},
        "Oracle Park": {"run_factor": 0.96, "hr_factor": 0.80},
        "_DEFAULT": {"run_factor": 1.00, "hr_factor": 1.00},
    }
    
    print('\nTesting BALLPARK_FACTORS lookup:')
    for i in range(len(df)):
        nm = str(names.iloc[i]) if names is not None else None
        if nm and nm in BALLPARK_FACTORS:
            meta = BALLPARK_FACTORS[nm]
            print(f'  "{nm}" -> {meta}')
        else:
            print(f'  "{nm}" -> NOT FOUND, using _DEFAULT')

if __name__ == "__main__":
    main()
