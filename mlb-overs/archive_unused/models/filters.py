# models/filter.py
import pandas as pd

def strong_plays(df: pd.DataFrame, min_edge: float = 0.6, min_conf: float = 0.80) -> pd.DataFrame:
    df = df.copy()
    # Build confidence as max calibrated probability toward either side
    if "conf" not in df.columns:
        if {"p_over_cal","p_under_cal"}.issubset(df.columns):
            df["conf"] = df[["p_over_cal","p_under_cal"]].max(axis=1)
        else:
            # Fallback: no calibration available -> neutral 0.5 (will be filtered out)
            df["conf"] = 0.5

    # Pick direction based on calibrated probs
    if {"p_over_cal","p_under_cal"}.issubset(df.columns):
        df["pick"] = df.apply(lambda r: "OVER" if r["p_over_cal"] >= r["p_under_cal"] else "UNDER", axis=1)
    else:
        df["pick"] = pd.NA

    # Edge filter (use absolute)
    df["edge_abs"] = df["edge"].abs() if "edge" in df.columns else (df["y_pred"] - df["k_close"]).abs()

    keep = (df["edge_abs"] >= float(min_edge)) & (df["conf"] >= float(min_conf))
    out = df.loc[keep].sort_values("edge_abs", ascending=False)
    return out
