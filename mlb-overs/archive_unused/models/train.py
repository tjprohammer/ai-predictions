# models/train.py
from __future__ import annotations
import argparse, json, os, datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from joblib import dump

DEF_FEATURES = [
    # core signals — keep in sync with models/infer.py build
    "k_close",
    "hxwoba14","axwoba14","hiso14","aiso14","hbb14","abb14",
    "hxwoba30","axwoba30","hiso30","aiso30","hbb30","abb30",
    "hbp_fip_yday","abp_fip_yday","hbp_b2b","abp_b2b",
    # optional starter aggregates if you've populated them
    "hsp_xwoba10","asp_xwoba10","hsp_csw10","asp_csw10","hsp_velo10","asp_velo10",
    # pitcher ERA features (L3/L5/L10 and season ERAs)
    "home_sp_era_season","away_sp_era_season",
    "home_sp_era_l3","away_sp_era_l3",
    "home_sp_era_l5","away_sp_era_l5", 
    "home_sp_era_l10","away_sp_era_l10",
    # vs opponent ERA features
    "home_sp_era_vs_opp","away_sp_era_vs_opp",
    # Enhanced features for better accuracy
    "home_xwoba","away_xwoba","home_iso","away_iso",
    "home_bp_era","away_bp_era","home_bp_fip","away_bp_fip",
    "temp_f","wind_mph","pf_runs_3y","park_altitude_ft",
    "wx_run_boost","home_bp_b2b_flag","away_bp_b2b_flag",
    # Pitcher vs offense matchup indicators
    "home_sp_xwoba_allowed3","away_sp_xwoba_allowed3",
    "home_sp_csw3","away_sp_csw3","home_sp_velo3","away_sp_velo3",
]

def load_training_frame(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Expect columns:
    #  - date, game_id
    #  - target_y (actual total runs)  <-- ensure your feature builder wrote this
    #  - the features in DEF_FEATURES (some may be missing; we'll fill)
    return df

def prepare_Xy(df: pd.DataFrame):
    # Only keep rows with a target
    df = df.copy()
    if "target_y" not in df.columns:
        raise ValueError("features/train.parquet must include column 'target_y' (actual total runs).")

    # build feature matrix with safe defaults
    X = pd.DataFrame(index=df.index)
    for c in DEF_FEATURES:
        if c not in df.columns:
            X[c] = np.nan
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce")

    # sensible priors/fill - updated with better defaults based on 2025 season data
    priors = {
        "hxwoba14":0.320,"axwoba14":0.320,"hiso14":0.160,"aiso14":0.160,"hbb14":0.085,"abb14":0.085,
        "hxwoba30":0.320,"axwoba30":0.320,"hiso30":0.160,"aiso30":0.160,"hbb30":0.085,"abb30":0.085,
        "hbp_fip_yday":4.10,"abp_fip_yday":4.10,"hbp_b2b":0,"abp_b2b":0,
        "hsp_xwoba10":0.300,"asp_xwoba10":0.300,"hsp_csw10":0.27,"asp_csw10":0.27,
        "hsp_velo10":94.0,"asp_velo10":94.0,
        # ERA feature priors - adjusted based on 2025 MLB averages
        "home_sp_era_season":4.20,"away_sp_era_season":4.20,
        "home_sp_era_l3":4.20,"away_sp_era_l3":4.20,
        "home_sp_era_l5":4.20,"away_sp_era_l5":4.20,
        "home_sp_era_l10":4.20,"away_sp_era_l10":4.20,
        # vs opponent ERA priors
        "home_sp_era_vs_opp":4.20,"away_sp_era_vs_opp":4.20,
        # Enhanced feature priors
        "home_xwoba":0.320,"away_xwoba":0.320,"home_iso":0.160,"away_iso":0.160,
        "home_bp_era":4.10,"away_bp_era":4.10,"home_bp_fip":4.10,"away_bp_fip":4.10,
        "temp_f":72.0,"wind_mph":8.0,"pf_runs_3y":1.0,"park_altitude_ft":500.0,
        "wx_run_boost":0.0,"home_bp_b2b_flag":0,"away_bp_b2b_flag":0,
        "home_sp_xwoba_allowed3":0.300,"away_sp_xwoba_allowed3":0.300,
        "home_sp_csw3":0.27,"away_sp_csw3":0.27,"home_sp_velo3":94.0,"away_sp_velo3":94.0,
    }
    for k,v in priors.items():
        if k in X.columns:
            X[k] = X[k].fillna(v)

    y = pd.to_numeric(df["target_y"], errors="coerce")
    mask = y.notna()
    return X.loc[mask], y.loc[mask], df.loc[mask]

def time_series_cv_rmse(X, y, n_splits=5):
    tss = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    for tr, va in tss.split(X):
        # Enhanced model parameters matching final model
        model = GradientBoostingRegressor(
            n_estimators=1000,       # Increased for better learning
            learning_rate=0.015,     # Slightly reduced for more careful learning
            max_depth=5,             # Increased depth to capture ERA interactions  
            subsample=0.85,          # Better sampling
            max_features='sqrt',     # Feature selection
            min_samples_split=15,    # Reduced for more flexibility
            min_samples_leaf=5,      # Reduced for more flexibility
            random_state=42,
            alpha=0.10              # Reduced regularization
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        mse = mean_squared_error(y.iloc[va], pred)
        rmses.append(np.sqrt(mse))
    return float(np.mean(rmses)), float(np.std(rmses))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(Path("features") / "train.parquet"))
    ap.add_argument("--out-model", default=str(Path("models") / "model_totals.joblib"))
    ap.add_argument("--out-meta",  default=str(Path("models") / "model_totals_meta.json"))
    args = ap.parse_args()

    df = load_training_frame(args.features)
    # chronological split guard (avoid leakage)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date")

    X, y, df2 = prepare_Xy(df)

    # Cross-validated score (time-aware)
    cv_rmse_mean, cv_rmse_std = time_series_cv_rmse(X, y, n_splits=5)
    print(f"[train] time-series CV RMSE: {cv_rmse_mean:.3f} ± {cv_rmse_std:.3f}")

    # Fit final model on all data with enhanced parameters based on accuracy analysis
    model = GradientBoostingRegressor(
        n_estimators=1000,       # Increased for better learning
        learning_rate=0.015,     # Slightly reduced for more careful learning  
        max_depth=5,             # Increased depth to capture ERA interactions
        subsample=0.85,          # Better sampling
        max_features='sqrt',     # Feature selection
        min_samples_split=15,    # Reduced for more flexibility
        min_samples_leaf=5,      # Reduced for more flexibility
        random_state=42,
        alpha=0.10              # Reduced regularization
    )
    model.fit(X, y)
    pred_all = model.predict(X)
    resid = y - pred_all
    resid_std = float(np.std(resid))

    # save
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    dump(model, args.out_model)
    meta = {
        "features": DEF_FEATURES,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "residual_std": resid_std,
        "trained_on_rows": int(len(X)),
        "trained_at": dt.datetime.utcnow().isoformat() + "Z"
    }
    with open(args.out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] saved model -> {args.out_model}")
    print(f"[train] saved meta  -> {args.out_meta} (residual_std={resid_std:.3f})")

if __name__ == "__main__":
    main()
