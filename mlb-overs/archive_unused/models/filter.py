"""
Bettable-leg selector
"""

def select_bets(predictions):
    raise NotImplementedError

# minimal gate compatible with gameday.ps1

def strong_plays(df, min_edge=0.6, min_conf=0.80, require_consensus=False):
    import pandas as pd
    if isinstance(df, str):
        df = pd.read_parquet(df)
    df = df.copy()
    df["conf"] = df[["p_over_cal","p_under_cal"]].max(axis=1)
    df["edge_abs"] = df["edge"].abs()
    out = df[(df["edge_abs"] >= min_edge) & (df["conf"] >= min_conf)].copy()
    out.sort_values(["conf","edge_abs"], ascending=[False, False], inplace=True)
    return out
