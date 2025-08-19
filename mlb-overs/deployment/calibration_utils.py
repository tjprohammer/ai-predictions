# calibration_utils.py
import json, math, numpy as np, pandas as pd
from dataclasses import dataclass
from sqlalchemy import text

@dataclass
class Calibrator:
    method: str                   # "shrink", "isotonic"
    params: dict                  # {"s": ...} or {"x": [...], "y": [...]}

    def apply(self, p):
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        if self.method == "shrink":
            s = float(self.params["s"])
            return 0.5 + s * (p - 0.5)
        elif self.method == "isotonic":
            # piecewise-constant monotone: x sorted asc, y nondecreasing
            x = np.asarray(self.params["x"], dtype=float)
            y = np.asarray(self.params["y"], dtype=float)
            idx = np.searchsorted(x, p, side="right") - 1
            idx = np.clip(idx, 0, len(x) - 1)
            return y[idx]
        else:
            return p

def _fit_shrink(df):
    """Fit slope-only shrinkage: p' = 0.5 + s*(p-0.5), minimize Brier."""
    p = np.clip(df["p_win"].values.astype(float), 1e-6, 1-1e-6)
    y = df["won"].values.astype(float)
    x = (p - 0.5)
    den = float(np.dot(x, x)) or 1e-9
    num = float(np.dot(x, (y - 0.5)))
    s = num / den
    # Guardrails: never expand > 1.2 or shrink < 0.2 with tiny N
    n = len(df)
    s = float(np.clip(s, 0.2, 1.2))
    # Heavier shrink when N is very small (pull toward 0.5)
    if n < 30:
        lam = 30 - n
        s = (n * s) / (n + lam)
    return Calibrator(method="shrink", params={"s": s})

def _fit_isotonic(df):
    """Simple pooled isotonic using 10 bins → step function."""
    # bins and stats
    bins = np.linspace(0., 1., 11)
    df = df.copy()
    df["bin"] = pd.cut(df["p_win"].clip(1e-6, 1-1e-6), bins=bins, include_lowest=True, right=False)
    g = df.groupby("bin", observed=True)
    tbl = g.agg(n=("won","size"), pbar=("p_win","mean"), ybar=("won","mean")).reset_index()
    tbl = tbl[(tbl["n"] > 0) & tbl["pbar"].notna() & tbl["ybar"].notna()].sort_values("pbar")
    if tbl.empty:
        return _fit_shrink(df)

    # pool adjacent violators (isotonic regression in 1D)
    p = tbl["pbar"].to_numpy()
    y = tbl["ybar"].to_numpy()
    w = tbl["n"].to_numpy().astype(float)

    # monotone nondecreasing y with weights
    blocks = [(p[i], y[i], w[i]) for i in range(len(p))]
    # merge until nondecreasing
    merged = []
    for (px, yx, wx) in blocks:
        merged.append([px, yx, wx])
        while len(merged) >= 2 and merged[-2][1] > merged[-1][1]:
            p1, y1, w1 = merged[-2]
            p2, y2, w2 = merged[-1]
            y12 = (y1*w1 + y2*w2) / (w1 + w2)
            p12 = (p1*w1 + p2*w2) / (w1 + w2)
            merged[-2] = [p12, y12, w1+w2]
            merged.pop()

    x_steps = [m[0] for m in merged]
    y_steps = [m[1] for m in merged]
    # Ensure cover entire [0,1]
    if x_steps[0] > 0.0:
        x_steps = [0.0] + x_steps
        y_steps = [y_steps[0]] + y_steps
    if x_steps[-1] < 1.0:
        x_steps = x_steps + [1.0]
        y_steps = y_steps + [y_steps[-1]]
    return Calibrator(method="isotonic", params={"x": x_steps, "y": y_steps})

def fit_best_calibrator(df):
    """Choose method by sample size."""
    n = len(df)
    if n >= 80:
        return _fit_isotonic(df)
    return _fit_shrink(df)

def evaluate(df, p_col="p_win"):
    p = np.clip(df[p_col].values.astype(float), 1e-6, 1-1e-6)
    y = df["won"].values.astype(float)
    brier = float(np.mean((p - y)**2))
    # ECE with 10 bins
    bins = np.linspace(0.0, 1.0, 11)
    df = df.copy()
    df["bin"] = pd.cut(p, bins=bins, include_lowest=True, right=False)
    g = df.groupby("bin", observed=True)
    calib = g.agg(n=("won","size"), avg_pred=(p_col,"mean"), emp_rate=("won","mean")).reset_index()
    calib["abs_gap"] = (calib["emp_rate"] - calib["avg_pred"]).abs()
    N = max(int(df.shape[0]), 1)
    ece = float((calib["n"] / N * calib["abs_gap"]).sum())
    return brier, ece

def train_and_store_calibrator(engine, end_date, days=60, model_version=None, table="probability_predictions"):
    # Pull latest per-game predictions in window and grade vs finals (pushes removed)
    q = """
    WITH latest AS (
      SELECT pp.*,
             ROW_NUMBER() OVER (PARTITION BY game_id, game_date ORDER BY created_at DESC) rn
      FROM probability_predictions pp
      WHERE game_date BETWEEN :s AND :e
        AND (:mv IS NULL OR model_version = :mv)
    )
    SELECT l.*, 
           COALESCE(eg.total_runs, eg.home_score + eg.away_score) AS total_runs,
           COALESCE(NULLIF(l.priced_total,0), l.market_total) AS line_total
    FROM latest l
    LEFT JOIN enhanced_games eg ON eg.game_id = l.game_id AND eg."date" = l.game_date
    WHERE rn = 1
    """
    import datetime as dt
    e = pd.to_datetime(end_date).date()
    s = e - pd.Timedelta(days=days-1)
    df = pd.read_sql(text(q), engine, params={"s": s, "e": e, "mv": model_version})

    if df.empty:
        return None, {"note":"no rows"}

    # choose side with higher EV and define p_win / won; drop pushes
    choose_over = df["ev_over"] >= df["ev_under"]
    df["side"]  = np.where(choose_over, "OVER", "UNDER")
    df["p_win"] = np.where(choose_over, df["p_over"], df["p_under"])
    df = df.loc[df["total_runs"].notna() & df["line_total"].notna() & (df["total_runs"] != df["line_total"])].copy()
    df["won"] = np.where(df["side"].eq("OVER"), (df["total_runs"] > df["line_total"]).astype(float),
                                                (df["total_runs"] < df["line_total"]).astype(float))
    if df.empty:
        return None, {"note":"no graded rows"}

    # fit + evaluate
    raw_brier, raw_ece = evaluate(df, "p_win")
    cal = fit_best_calibrator(df)
    df["p_cal"] = cal.apply(df["p_win"].values)
    cal_brier, cal_ece = evaluate(df.rename(columns={"p_cal":"p_win"}), "p_win")

    meta = {
        "n": int(len(df)),
        "method": cal.method,
        "params": cal.params,
        "raw_brier": raw_brier, "raw_ece": raw_ece,
        "cal_brier": cal_brier, "cal_ece": cal_ece,
        "window_days": int(days),
        "model_version": model_version or "all",
        "end_date": str(e),
    }
    return cal, meta

def apply_calibrator_to_table(engine, calibrator: Calibrator, model_version=None, target_version=None):
    """Write p_over_cal/p_under_cal + tag. 
    
    Args:
        target_version: If provided, apply calibration to this version's predictions.
                       If None, uses model_version (original behavior).
    """
    apply_version = target_version or model_version
    
    # Add columns if missing (Postgres-safe)
    with engine.begin() as c:
        for col in ("p_over_cal","p_under_cal","calibration_tag"):
            c.execute(text(f"ALTER TABLE probability_predictions ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION"))
        c.execute(text(f"ALTER TABLE probability_predictions ADD COLUMN IF NOT EXISTS calibration_method TEXT"))
        c.execute(text(f"ALTER TABLE probability_predictions ADD COLUMN IF NOT EXISTS calibration_params JSONB"))

        q_sel = """
        WITH ranked AS (
          SELECT id, p_over, p_under, ROW_NUMBER() OVER (PARTITION BY game_id, game_date ORDER BY created_at DESC) rn
          FROM probability_predictions
          WHERE (:mv IS NULL OR model_version = :mv)
        )
        SELECT id, p_over, p_under FROM ranked WHERE rn=1
        """
        rows = pd.read_sql(text(q_sel), engine, params={"mv": apply_version})

        if rows.empty:
            return 0

        p_over_cal = calibrator.apply(rows["p_over"].values)
        p_under_cal = 1.0 - p_over_cal  # keep a simplex

        # batch update
        q_upd = text("""
            UPDATE probability_predictions
            SET p_over_cal = :poc, p_under_cal = :puc,
                calibration_tag = :tag,
                calibration_method = :m,
                calibration_params = :params
            WHERE id = :id
        """)
        train_tag = model_version or 'all' 
        apply_tag = apply_version or 'all'
        tag = f"{calibrator.method}_cal_train_{train_tag}_apply_{apply_tag}"
        params_json = json.dumps(calibrator.params)
        for i in range(len(rows)):
            c.execute(q_upd, {
                "poc": float(p_over_cal[i]), "puc": float(p_under_cal[i]),
                "tag": tag, "m": calibrator.method, "params": params_json,
                "id": int(rows.iloc[i]["id"])
            })
        return len(rows)

def calibrate_mv_train_apply(engine, train_version, apply_version, end_date, days=60):
    """Train calibrator on historical train_version data, apply to current apply_version predictions.
    
    Returns:
        tuple: (calibrator, metadata, applied_count)
    """
    # Step 1: Train calibrator on historical data from train_version
    calibrator, meta = train_and_store_calibrator(
        engine, end_date, days=days, model_version=train_version
    )
    
    if calibrator is None:
        return None, meta, 0
    
    # Step 2: Apply calibrator to current apply_version predictions  
    applied_count = apply_calibrator_to_table(
        engine, calibrator, model_version=train_version, target_version=apply_version
    )
    
    # Update metadata to reflect cross-version application
    meta["train_version"] = train_version
    meta["apply_version"] = apply_version
    meta["applied_count"] = applied_count
    
    return calibrator, meta, applied_count
