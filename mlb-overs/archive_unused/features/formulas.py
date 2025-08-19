"""
Feature helper formulas: ewma(), weather_run_factor(), interactions()
"""

from typing import Sequence
import numpy as np


def ewma(values: Sequence[float], span: int) -> float:
    """Exponentially weighted moving average for last N values.
    Returns np.nan if empty.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    alpha = 2 / (span + 1)
    avg = arr[0]
    for v in arr[1:]:
        avg = alpha * v + (1 - alpha) * avg
    return float(avg)


def weather_run_factor(temp_f: float | None, wind_mph: float | None, wind_dir: str | None) -> float:
    """Toy placeholder: positive means boost to runs, negative suppresses.
    Replace with park-specific/stadium orientation-aware model.
    """
    if temp_f is None or wind_mph is None:
        return 0.0
    base = (temp_f - 60) * 0.01
    wind = (wind_mph or 0) * 0.02
    return float(base + wind)


def interactions(row: dict) -> dict:
    """Return dict of interaction features from a row-like mapping."""
    out = {}
    # Example placeholders; replace during port
    kbb = row.get("sp_k_pct", 0) - row.get("sp_bb_pct", 0)
    out["sp_kbb"] = kbb
    return out
