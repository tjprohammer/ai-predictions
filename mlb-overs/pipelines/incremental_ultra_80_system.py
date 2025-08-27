#!/usr/bin/env python3
"""
ULTRA 80% PREDICTION INTERVAL COVERAGE SYSTEM
- Team-level modeling: home/away runs separately (better calibration)
- Heteroskedastic variance: learned log-variance from features
- Rich features: pitcher splits, bullpen fatigue, lineup, weather, umpire (optional if missing)
- Conformal prediction intervals: ~80% coverage (adaptive, trailing window)
- Market calibration: isotonic on (pred_total - market_total) for ROI tracking
- No leakage: pre-game features only, chronological ordering by start_ts
"""
import os
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path


# ----------------------------- CalibratorStack -----------------------------

class CalibratorStack:
    """
    Bucketed isotonic (by market-total & sigma) + feature-augmented logistic stacker.
    Predicts P(Over) from diff and a few context features, with smart shrinkage.
    """
    def __init__(self):
        self.iso_by_bucket = {}      # (mtot_bucket, sig_bucket) -> (iso_model, n)
        self.X_hist, self.y_hist = [], []
        self.logit = LogisticRegression(max_iter=200, C=1.0)
        self.logit_fitted = False

    @staticmethod
    def _mtot_bucket(m):   return 'low' if m <= 7.5 else ('high' if m >= 10.0 else 'mid')
    @staticmethod
    def _sig_bucket(s):    return 'low' if s < 2.2 else ('high' if s >= 3.0 else 'mid')

    def _features(self, diff, feats, sigma, market_total):
        return [
            float(diff),
            float(sigma),
            float(market_total),
            float(feats.get('park_pf_runs', 1.0)),
            float(feats.get('wind_speed', 0.0)),
            float(feats.get('temperature', 72.0)),
            float(feats.get('park_roof_closed', 0.0)),
        ]

    def update(self, diff, outcome, feats, sigma, market_total):
        # logistic buffer & periodic refit
        x = self._features(diff, feats, sigma, market_total)
        self.X_hist.append(x); self.y_hist.append(int(outcome))
        n = len(self.y_hist)
        if n >= 300 and n % 75 == 0:
            try:
                self.logit.fit(np.array(self.X_hist, dtype=float), np.array(self.y_hist, dtype=int))
                self.logit_fitted = True
            except Exception:
                self.logit_fitted = False

        # isotonic per bucket
        b = (self._mtot_bucket(market_total), self._sig_bucket(sigma))
        buf_name = 'buf_' + '_'.join(b)
        buf = getattr(self, buf_name, {'d': [], 'y': []})
        buf['d'].append(float(diff)); buf['y'].append(int(outcome))
        setattr(self, buf_name, buf)

        if len(buf['y']) >= 150 and len(buf['y']) % 50 == 0:
            try:
                iso = IsotonicRegression(out_of_bounds='clip').fit(buf['d'], buf['y'])
                self.iso_by_bucket[b] = (iso, len(buf['y']))
            except Exception:
                pass

    def predict(self, diff, feats, sigma, market_total):
        # iso by bucket (if available)
        b = (self._mtot_bucket(market_total), self._sig_bucket(sigma))
        if b in self.iso_by_bucket:
            try:
                p_iso = float(self.iso_by_bucket[b][0].predict([diff])[0])
            except Exception:
                p_iso = 0.5
        else:
            p_iso = 0.5

        # feature-augmented logistic (if fitted)
        if self.logit_fitted:
            try:
                p_log = float(self.logit.predict_proba([self._features(diff, feats, sigma, market_total)])[0, 1])
            except Exception:
                p_log = 0.5
        else:
            p_log = 0.5

        # blend + shrink toward 0.5 when data is sparse and |diff| small
        p_raw = 0.5 * (p_iso + p_log)
        n_hist = len(self.y_hist)
        shrink_size = float(np.clip(1.0 - 1.0/np.sqrt(max(n_hist, 1)), 0.50, 0.90))  # 0.5..0.9
        shrink_diff = float(np.clip(abs(diff)/1.2, 0.0, 1.0))                        # 0..1 by ~1.2 runs
        shrink = 0.5*shrink_size + 0.5*shrink_diff
        return 0.5 + (p_raw - 0.5) * shrink


# ----------------------------- Helpers -----------------------------

def _to_float(x, default=0.0):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return float(default)
    except Exception:
        return float(default)


def _clean_row_dict(row_like: Any) -> Dict[str, Any]:
    """Ensure we can call .get on a row from Pandas (Series -> dict-like)."""
    if isinstance(row_like, dict):
        return row_like
    if hasattr(row_like, 'to_dict'):
        return row_like.to_dict()
    return dict(row_like)


# ----------------------------- Core System -----------------------------

class IncrementalUltra80System:
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
        self.engine = create_engine(self.db_url, pool_pre_ping=True)

        # Team-level mean models (home / away)
        self.home_model = SGDRegressor(
            loss='huber', epsilon=1.0,
            learning_rate='adaptive', eta0=0.01, alpha=1e-3,
            random_state=42, warm_start=True
        )
        self.away_model = SGDRegressor(
            loss='huber', epsilon=1.0,
            learning_rate='adaptive', eta0=0.01, alpha=1e-3,
            random_state=43, warm_start=True
        )

        # Heteroskedastic variance heads (predict log var)
        self.home_var_model = SGDRegressor(
            loss='squared_error', learning_rate='adaptive',
            eta0=0.01, alpha=1e-4, random_state=44, warm_start=True
        )
        self.away_var_model = SGDRegressor(
            loss='squared_error', learning_rate='adaptive',
            eta0=0.01, alpha=1e-4, random_state=45, warm_start=True
        )

        # Optional batch models (periodic retrain for stability / nonlinearity)
        self.home_batch_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        self.away_batch_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=43, n_jobs=-1)

        # Market calibration (diff -> prob over)
        self.market_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.market_diffs: List[float] = []
        self.market_outcomes: List[float] = []
        
        # Advanced probability calibration
        self.prob_cal = CalibratorStack()

        # Conformal scores (absolute total residuals)
        self.conformal_scores: List[float] = []
        self.coverage_target = 0.80
        self.conformal_window = 2000  # trailing window for adaptivity
        
        # Context-aware conformal buckets (tighter intervals where noise is lower)
        self.conformal_bucket_scores = {'low': [], 'mid': [], 'high': []}
        self.conformal_zscores = {'low': [], 'mid': [], 'high': []}  # |residual| / sigma_total
        self.var_boost = {'pf': 0.12, 'wind': 0.10, 'heat': 0.06}  # Context-aware variance adjustments

        # Scaling
        self.scaler = StandardScaler()
        self.update_scaler = False  # set True to adapt scaling slowly
        self.warmup_X: List[List[float]] = []
        self.warmup_y_home: List[float] = []
        self.warmup_y_away: List[float] = []
        self.warmup_n = 200
        self.is_fitted = False
        self.feature_columns: List[str] = []

        # Bias terms (per team model)
        self.bias_alpha = 0.01
        self.home_bias_mu = 0.0
        self.away_bias_mu = 0.0

        # Residual buffers (for diagnostics)
        self.home_residuals: List[float] = []
        self.away_residuals: List[float] = []

        # Rolling retrain policy
        self.batch_window = 2000
        self.retrain_every = 200
        self.min_retrain_gap = 150
        self.last_retrain_at = -10**9

        # Season/aging decay
        self.season_decay = 0.97
        self._last_decay_month: Optional[str] = None

        # State & caches
        self.prediction_history: List[Dict[str, Any]] = []
        self.team_stats: Dict[str, Dict[str, Any]] = {}
        self.matchup_history: Dict[str, List[Dict[str, float]]] = {}
        self.pitcher_stats: Dict[int, Dict[str, float]] = {}

        # Book preferences and display settings from environment
        self.preferred_books = None
        try:
            books_env = os.getenv('BOOKS')
            if books_env:
                self.preferred_books = [b.strip() for b in books_env.split(',') if b.strip()]
        except Exception:
            self.preferred_books = None

        try:
            self.display_bankroll = float(os.getenv('BANKROLL', '100'))
        except Exception:
            self.display_bankroll = 100.0
        self.bullpen_usage: Dict[str, Dict[str, float]] = {}
        self.pitcher_handedness: Dict[int, str] = {}
        self.umpire_factors: Dict[str, float] = {}
        self.venue_factors: Dict[str, float] = {}

        # Try to pre-load optional factors (safe if tables missing)
        self._maybe_load_static_factors()

    # ----------------------- DB & Ordering -----------------------

    def get_completed_games_chronological(self, start_date=None, end_date=None) -> pd.DataFrame:
        """
        Completed games ordered by start_ts, with 'as-of' snapshots for:
        - Pitchers: pitcher_daily_rolling (ERA/WHIP/K9/BB9/HR9)
        - Teams: teams_offense_daily (wRC+, wOBA, runs_pg, rolling runs_pg_l5/l10/l20, splits)
        - Bullpens: bullpens_daily (bp_era, bp_fip, bp_kbb_pct)
        - Parks: parks (pf_runs_3y, pf_hr_3y, altitude_ft, roof_type)
        All snapshots are taken at stat_date <= date - 1 day to avoid leakage.
        Fallbacks: enhanced_games season/park columns if a snapshot is missing.
        """
        q = text("""
            SELECT
              eg.game_id, eg.date, eg.home_team, eg.away_team,
              eg.total_runs, eg.home_score, eg.away_score,
              eg.temperature, eg.wind_speed, eg.venue_name,
              COALESCE(eg.opening_total, rmg.opening_total, eg.market_total, 9.0) AS opening_total,
              eg.home_sp_id, eg.away_sp_id,
              COALESCE(hb.throws, eg.home_sp_hand, 'R') AS home_sp_hand,
              COALESCE(ab.throws, eg.away_sp_hand, 'R') AS away_sp_hand,
              eg.start_ts,

              -- Fallback season fields (from enhanced_games if rolling rows missing)
              eg.home_sp_season_era     AS home_sp_era_season,
              eg.away_sp_season_era     AS away_sp_era_season,
              eg.home_sp_whip           AS home_sp_whip_season,
              eg.away_sp_whip           AS away_sp_whip_season,

              -- ===== Pitcher as-of (home) =====
              hpd.era       AS home_sp_era_roll,
              hpd.whip      AS home_sp_whip_roll,
              hpd.k_per_9   AS home_sp_k9_roll,
              hpd.bb_per_9  AS home_sp_bb9_roll,
              hpd.hr_per_9  AS home_sp_hr9_roll,

              -- ===== Pitcher as-of (away) =====
              apd.era       AS away_sp_era_roll,
              apd.whip      AS away_sp_whip_roll,
              apd.k_per_9   AS away_sp_k9_roll,
              apd.bb_per_9  AS away_sp_bb9_roll,
              apd.hr_per_9  AS away_sp_hr9_roll,

              -- ===== Team offense as-of (home) =====
              hto.wrcplus        AS home_wrcplus_roll,
              hto.woba           AS home_woba_roll,
              hto.xwoba          AS home_xwoba_roll,
              hto.runs_pg        AS home_runs_pg_roll,
              hto.runs_pg_l5     AS home_runs_pg_l5,
              hto.runs_pg_l10    AS home_runs_pg_l10,
              hto.runs_pg_l20    AS home_runs_pg_l20,
              hto.vs_rhp_xwoba   AS home_vs_rhp_xwoba,
              hto.vs_lhp_xwoba   AS home_vs_lhp_xwoba,
              hto.ops            AS home_ops_roll,

              -- ===== Team offense as-of (away) =====
              ato.wrcplus        AS away_wrcplus_roll,
              ato.woba           AS away_woba_roll,
              ato.xwoba          AS away_xwoba_roll,
              ato.runs_pg        AS away_runs_pg_roll,
              ato.runs_pg_l5     AS away_runs_pg_l5,
              ato.runs_pg_l10    AS away_runs_pg_l10,
              ato.runs_pg_l20    AS away_runs_pg_l20,
              ato.vs_rhp_xwoba   AS away_vs_rhp_xwoba,
              ato.vs_lhp_xwoba   AS away_vs_lhp_xwoba,
              ato.ops            AS away_ops_roll,

              -- ===== Bullpen as-of (home) =====
              hbp.bp_era         AS home_bp_era_roll,
              hbp.bp_fip         AS home_bp_fip_roll,
              hbp.bp_kbb_pct     AS home_bp_kbb_pct_roll,

              -- ===== Bullpen as-of (away) =====
              abp.bp_era         AS away_bp_era_roll,
              abp.bp_fip         AS away_bp_fip_roll,
              abp.bp_kbb_pct     AS away_bp_kbb_pct_roll,

              -- ===== Park data =====
              COALESCE(p.pf_runs_3y, 1.0)::numeric    AS park_pf_runs_3y,
              COALESCE(p.pf_hr_3y, 1.0)::numeric      AS park_pf_hr_3y,
              COALESCE(p.altitude_ft, 0.0)::numeric   AS park_altitude_ft,
              COALESCE(p.roof_type, 'unknown')         AS park_roof_type,

              -- Enhanced games park fallback
              eg.ballpark_run_factor   AS eg_ballpark_run_factor,

              -- ===== Recency features (21-day pitcher, 7-day bullpen, 14-day offense) =====
              hpd21.home_sp_n_days_21,
              hpd21.home_sp_era_l21,
              hpd21.home_sp_whip_l21,
              
              apd21.away_sp_n_days_21,
              apd21.away_sp_era_l21,
              apd21.away_sp_whip_l21,
              
              hbp7.home_bp_n_days_7,
              hbp7.home_bp_era_l7,
              
              abp7.away_bp_n_days_7,
              abp7.away_bp_era_l7,
              
              h_off14.home_off_n_games_14,
              h_off14.home_runs_pg_l14_eg,
              h_off14.home_rapg_l14_eg,
              
              a_off14.away_off_n_games_14,
              a_off14.away_runs_pg_l14_eg,
              a_off14.away_rapg_l14_eg

            FROM enhanced_games eg
            LEFT JOIN real_market_games rmg USING (game_id)

            -- HOME starter snapshot (≤ date - 1)
            LEFT JOIN LATERAL (
              SELECT pdr.*
              FROM pitcher_daily_rolling pdr
              WHERE pdr.pitcher_id = eg.home_sp_id
                AND pdr.stat_date <= eg.date - INTERVAL '1 day'
              ORDER BY pdr.stat_date DESC
              LIMIT 1
            ) hpd ON TRUE

            -- AWAY starter snapshot
            LEFT JOIN LATERAL (
              SELECT pdr.*
              FROM pitcher_daily_rolling pdr
              WHERE pdr.pitcher_id = eg.away_sp_id
                AND pdr.stat_date <= eg.date - INTERVAL '1 day'
              ORDER BY pdr.stat_date DESC
              LIMIT 1
            ) apd ON TRUE

            -- HOME team offense snapshot
            LEFT JOIN LATERAL (
              SELECT tod.*
              FROM teams_offense_daily tod
              WHERE tod.team = eg.home_team
                AND tod.date <= eg.date - INTERVAL '1 day'
              ORDER BY tod.date DESC
              LIMIT 1
            ) hto ON TRUE

            -- AWAY team offense snapshot
            LEFT JOIN LATERAL (
              SELECT tod.*
              FROM teams_offense_daily tod
              WHERE tod.team = eg.away_team
                AND tod.date <= eg.date - INTERVAL '1 day'
              ORDER BY tod.date DESC
              LIMIT 1
            ) ato ON TRUE

            -- HOME bullpen snapshot
            LEFT JOIN LATERAL (
              SELECT bd.*
              FROM bullpens_daily bd
              WHERE bd.team = eg.home_team
                AND bd.date <= eg.date - INTERVAL '1 day'
              ORDER BY bd.date DESC
              LIMIT 1
            ) hbp ON TRUE

            -- AWAY bullpen snapshot
            LEFT JOIN LATERAL (
              SELECT bd.*
              FROM bullpens_daily bd
              WHERE bd.team = eg.away_team
                AND bd.date <= eg.date - INTERVAL '1 day'
              ORDER BY bd.date DESC
              LIMIT 1
            ) abp ON TRUE

            -- HOME starter recent (last 21 days)
            LEFT JOIN LATERAL (
              SELECT
                COUNT(*) AS home_sp_n_days_21,
                AVG(pdr.era) AS home_sp_era_l21,
                AVG(pdr.whip) AS home_sp_whip_l21
              FROM pitcher_daily_rolling pdr
              WHERE pdr.pitcher_id = eg.home_sp_id
                AND pdr.stat_date BETWEEN eg.date - INTERVAL '21 days' AND eg.date - INTERVAL '1 day'
            ) hpd21 ON TRUE

            -- AWAY starter recent (last 21 days)
            LEFT JOIN LATERAL (
              SELECT
                COUNT(*) AS away_sp_n_days_21,
                AVG(pdr.era) AS away_sp_era_l21,
                AVG(pdr.whip) AS away_sp_whip_l21
              FROM pitcher_daily_rolling pdr
              WHERE pdr.pitcher_id = eg.away_sp_id
                AND pdr.stat_date BETWEEN eg.date - INTERVAL '21 days' AND eg.date - INTERVAL '1 day'
            ) apd21 ON TRUE

            -- HOME bullpen recent (last 7 days)
            LEFT JOIN LATERAL (
              SELECT
                COUNT(*) AS home_bp_n_days_7,
                AVG(bd.bp_era) AS home_bp_era_l7
              FROM bullpens_daily bd
              WHERE bd.team = eg.home_team
                AND bd.date BETWEEN eg.date - INTERVAL '7 days' AND eg.date - INTERVAL '1 day'
            ) hbp7 ON TRUE

            -- AWAY bullpen recent (last 7 days)
            LEFT JOIN LATERAL (
              SELECT
                COUNT(*) AS away_bp_n_days_7,
                AVG(bd.bp_era) AS away_bp_era_l7
              FROM bullpens_daily bd
              WHERE bd.team = eg.away_team
                AND bd.date BETWEEN eg.date - INTERVAL '7 days' AND eg.date - INTERVAL '1 day'
            ) abp7 ON TRUE

            -- HOME team offense recent (last 14 days) from game results only
            LEFT JOIN LATERAL (
              SELECT
                COUNT(*) AS home_off_n_games_14,
                AVG(CASE
                      WHEN eg2.home_team = eg.home_team THEN eg2.home_score
                      ELSE eg2.away_score
                    END) AS home_runs_pg_l14_eg,
                AVG(CASE
                      WHEN eg2.home_team = eg.home_team THEN eg2.away_score
                      ELSE eg2.home_score
                    END) AS home_rapg_l14_eg
              FROM enhanced_games eg2
              WHERE eg2.date BETWEEN eg.date - INTERVAL '14 days' AND eg.date - INTERVAL '1 day'
                AND (eg2.home_team = eg.home_team OR eg2.away_team = eg.home_team)
            ) h_off14 ON TRUE

            -- AWAY team offense recent (last 14 days) from game results only
            LEFT JOIN LATERAL (
              SELECT
                COUNT(*) AS away_off_n_games_14,
                AVG(CASE
                      WHEN eg2.home_team = eg.away_team THEN eg2.home_score
                      ELSE eg2.away_score
                    END) AS away_runs_pg_l14_eg,
                AVG(CASE
                      WHEN eg2.home_team = eg.away_team THEN eg2.away_score
                      ELSE eg2.home_score
                    END) AS away_rapg_l14_eg
              FROM enhanced_games eg2
              WHERE eg2.date BETWEEN eg.date - INTERVAL '14 days' AND eg.date - INTERVAL '1 day'
                AND (eg2.home_team = eg.away_team OR eg2.away_team = eg.away_team)
            ) a_off14 ON TRUE

            -- Park meta (improved join via normalized venue names)
            LEFT JOIN parks_dim p
              ON p.park_id = norm_venue(eg.venue_name)
            
            -- Player handedness
            LEFT JOIN player_bio hb ON hb.player_id = eg.home_sp_id
            LEFT JOIN player_bio ab ON ab.player_id = eg.away_sp_id

            WHERE eg.total_runs IS NOT NULL
              AND eg.total_runs > 0
              AND (:start_date IS NULL OR eg.date >= :start_date)
              AND (:end_date   IS NULL OR eg.date <  :end_date)
            ORDER BY eg.start_ts, eg.game_id;
        """)
        params = {"start_date": start_date, "end_date": end_date}
        df = pd.read_sql(q, self.engine, params=params)
        if df.empty:
            print("⚠️  No completed games found for given window.")
        else:
            print(f"✅ Loaded {len(df):,} games ({df['date'].min()} → {df['date'].max()}) with as-of P/T/BP + park data")
        return df

    # ----------------------- Factor loaders (optional) -----------------------

    def _maybe_load_static_factors(self):
        # Venue factors
        try:
            vdf = pd.read_sql(text("SELECT park_name, run_factor FROM ballpark_enhanced_factors"), self.engine)
            for _, r in vdf.iterrows():
                self.venue_factors[str(r['park_name'])] = _to_float(r['run_factor'], 1.0)
            print(f"🏟️ Loaded {len(self.venue_factors)} venue factors")
        except Exception:
            pass
        # Umpire factors
        try:
            udf = pd.read_sql(text("SELECT name, runs_factor FROM umpire_comprehensive_stats"), self.engine)
            for _, r in udf.iterrows():
                self.umpire_factors[str(r['name'])] = _to_float(r['runs_factor'], 0.0)
            print(f"👨‍⚖️ Loaded {len(self.umpire_factors)} umpire factors")
        except Exception:
            pass

    # ----------------------- Decay logic -----------------------

    def _maybe_monthly_decay(self, month_key: str):
        if self._last_decay_month is None:
            self._last_decay_month = month_key
            return
        if month_key != self._last_decay_month:
            self._last_decay_month = month_key
            # Decay team aggregates
            for s in self.team_stats.values():
                s['total_runs_for']     *= self.season_decay
                s['total_runs_against'] *= self.season_decay
                s['total_games_runs']   *= self.season_decay
            # Decay matchup history totals
            for games in self.matchup_history.values():
                for g in games:
                    g['total_runs'] *= self.season_decay
            print('🧪 Applied monthly season/matchup decay.')

    # ----------------------- Team stat updates -----------------------

    def update_team_stats(self, game_row: Any, prediction: Optional[float] = None):
        row = _clean_row_dict(game_row)
        home = row['home_team']; away = row['away_team']
        total = _to_float(row['total_runs']); h = _to_float(row['home_score']); a = _to_float(row['away_score'])

        for t in (home, away):
            if t not in self.team_stats:
                self.team_stats[t] = {
                    'games_played': 0,
                    'total_runs_for': 0.0,
                    'total_runs_against': 0.0,
                    'total_games_runs': 0.0,
                    'recent_runs_ewma': 4.5,
                    'recent_against_ewma': 4.5,
                    'home_games': 0,
                    'away_games': 0,
                    'wins': 0,
                    'losses': 0,
                }
        th, ta = self.team_stats[home], self.team_stats[away]

        th['games_played'] += 1; ta['games_played'] += 1
        th['total_runs_for'] += h; th['total_runs_against'] += a; th['total_games_runs'] += total; th['home_games'] += 1
        ta['total_runs_for'] += a; ta['total_runs_against'] += h; ta['total_games_runs'] += total; ta['away_games'] += 1

        # EWMA recency
        th['recent_runs_ewma'] = 0.8 * th['recent_runs_ewma'] + 0.2 * h
        th['recent_against_ewma'] = 0.8 * th['recent_against_ewma'] + 0.2 * a
        ta['recent_runs_ewma'] = 0.8 * ta['recent_runs_ewma'] + 0.2 * a
        ta['recent_against_ewma'] = 0.8 * ta['recent_against_ewma'] + 0.2 * h

        # W/L
        if h > a:
            th['wins'] += 1; ta['losses'] += 1
        else:
            ta['wins'] += 1; th['losses'] += 1

        # Log prediction quality if provided
        if prediction is not None and np.isfinite(prediction):
            err = abs(prediction - total)
            self.prediction_history.append({
                'date': row['date'], 'game_id': row['game_id'],
                'actual': total, 'predicted': float(prediction), 'error': float(err),
                'within_1': err <= 1.0, 'within_15': err <= 1.5
            })

        # Matchup history append
        key = f"{home}_vs_{away}"
        self.matchup_history.setdefault(key, []).append({'date': str(row['date']), 'total_runs': total, 'home_score': h, 'away_score': a})

    # ----------------------- Features -----------------------

    def engineer_features_from_history(self, game_row: Any) -> Dict[str, float]:
        row = _clean_row_dict(game_row)
        home = row['home_team']; away = row['away_team']
        f: Dict[str, float] = {}

        # --- Market (opening only) ---
        mt = _to_float(row.get('opening_total', 9.0), 9.0)
        f['market_total'] = mt
        f['market_high']  = 1.0 if mt >= 10.0 else 0.0
        f['market_low']   = 1.0 if mt <= 7.5 else 0.0

        # --- Weather (still simple; you can add humidity/pressure if desired) ---
        temp = _to_float(row.get('temperature', 72), 72)
        wind = _to_float(row.get('wind_speed', 5), 5)
        f['temperature']   = temp
        f['wind_speed']    = wind
        f['temp_extreme']  = 1.0 if (temp >= 85 or temp <= 50) else 0.0
        f['wind_factor']   = min(wind / 15.0, 2.0)

        # --- Park factors (parks table or EG fallback) ---
        pf_runs  = row.get('park_pf_runs_3y')
        if pf_runs is None or not np.isfinite(float(pf_runs or 0)):
            pf_runs = _to_float(row.get('eg_ballpark_run_factor', 1.0), 1.0)
        else:
            pf_runs = _to_float(pf_runs, 1.0)
        f['park_pf_runs']  = pf_runs
        f['park_pf_hr']    = _to_float(row.get('park_pf_hr_3y', 1.0), 1.0)
        altitude           = _to_float(row.get('park_altitude_ft', 0.0), 0.0)
        f['park_altitude_kft'] = altitude / 1000.0
        roof               = str(row.get('park_roof_type', '') or '').lower()
        f['park_roof_closed'] = 1.0 if roof in ('dome', 'closed', 'fixed') else 0.0

        # --- Pitchers (as-of roll -> season -> default) ---
        def _pitcher_block(prefix: str):
            era  = row.get(f'{prefix}_sp_era_roll')
            whip = row.get(f'{prefix}_sp_whip_roll')
            if era  is None: era  = row.get(f'{prefix}_sp_era_season', 4.50)
            if whip is None: whip = row.get(f'{prefix}_sp_whip_season', 1.30)
            f[f'{prefix}_sp_era']  = _to_float(era,  4.50)
            f[f'{prefix}_sp_whip'] = _to_float(whip, 1.30)
            f[f'{prefix}_sp_k9']   = _to_float(row.get(f'{prefix}_sp_k9_roll',  8.0), 8.0)
            f[f'{prefix}_sp_bb9']  = _to_float(row.get(f'{prefix}_sp_bb9_roll', 3.0), 3.0)
            f[f'{prefix}_sp_hr9']  = _to_float(row.get(f'{prefix}_sp_hr9_roll', 1.2), 1.2)
            f[f'{prefix}_sp_lefty']= 1.0 if str(row.get(f'{prefix}_sp_hand', 'R')) == 'L' else 0.0
            f[f'{prefix}_sp_quality'] = max(0.0, 5.0 - f[f'{prefix}_sp_era'])

        _pitcher_block('home')
        _pitcher_block('away')

        # --- Team offense (as-of) ---
        def _team_off(prefix: str):
            f[f'{prefix}_wrcplus']     = _to_float(row.get(f'{prefix}_wrcplus_roll', 100.0), 100.0)
            f[f'{prefix}_woba']        = _to_float(row.get(f'{prefix}_woba_roll', 0.310), 0.310)
            f[f'{prefix}_xwoba']       = _to_float(row.get(f'{prefix}_xwoba_roll', 0.310), 0.310)
            f[f'{prefix}_runs_pg']     = _to_float(row.get(f'{prefix}_runs_pg_roll', 4.5), 4.5)
            f[f'{prefix}_runs_pg_l5']  = _to_float(row.get(f'{prefix}_runs_pg_l5',  4.5), 4.5)
            f[f'{prefix}_runs_pg_l10'] = _to_float(row.get(f'{prefix}_runs_pg_l10', 4.5), 4.5)
            f[f'{prefix}_runs_pg_l20'] = _to_float(row.get(f'{prefix}_runs_pg_l20', 4.5), 4.5)
            f[f'{prefix}_ops']         = _to_float(row.get(f'{prefix}_ops_roll', 0.700), 0.700)

        _team_off('home')
        _team_off('away')

        # Handedness splits vs opposing SP
        home_vs_split = _to_float(
            row.get('home_vs_lhp_xwoba') if f['away_sp_lefty'] == 1.0 else row.get('home_vs_rhp_xwoba'),
            f['home_xwoba']
        )
        away_vs_split = _to_float(
            row.get('away_vs_lhp_xwoba') if f['home_sp_lefty'] == 1.0 else row.get('away_vs_rhp_xwoba'),
            f['away_xwoba']
        )
        f['home_vs_split_xwoba'] = home_vs_split
        f['away_vs_split_xwoba'] = away_vs_split

        # --- Bullpens (as-of) ---
        def _bp(prefix: str):
            f[f'{prefix}_bp_era']     = _to_float(row.get(f'{prefix}_bp_era_roll', 4.20), 4.20)
            f[f'{prefix}_bp_fip']     = _to_float(row.get(f'{prefix}_bp_fip_roll', 4.20), 4.20)
            f[f'{prefix}_bp_kbb_pct'] = _to_float(row.get(f'{prefix}_bp_kbb_pct_roll', 0.14), 0.14)

        _bp('home')
        _bp('away')

        # --- Recency-blended features (safe version using only common columns) ---
        
        # Pitcher recency (L21) - only ERA and WHIP
        home_era_season  = _to_float(row.get('home_sp_era_roll',  row.get('home_sp_era_season', 4.50)), 4.50)
        home_whip_season = _to_float(row.get('home_sp_whip_roll', row.get('home_sp_whip_season', 1.30)), 1.30)
        away_era_season  = _to_float(row.get('away_sp_era_roll',  row.get('away_sp_era_season', 4.50)), 4.50)
        away_whip_season = _to_float(row.get('away_sp_whip_roll', row.get('away_sp_whip_season', 1.30)), 1.30)

        # Recency values (may be None)
        home_era_l21  = row.get('home_sp_era_l21')
        home_whip_l21 = row.get('home_sp_whip_l21')
        away_era_l21  = row.get('away_sp_era_l21')
        away_whip_l21 = row.get('away_sp_whip_l21')

        f['home_sp_era_blend']  = self._blend(home_era_season,  home_era_l21,  row.get('home_sp_n_days_21', 0), K=10, default=home_era_season)
        f['home_sp_whip_blend'] = self._blend(home_whip_season, home_whip_l21, row.get('home_sp_n_days_21', 0), K=10, default=home_whip_season)
        f['away_sp_era_blend']  = self._blend(away_era_season,  away_era_l21,  row.get('away_sp_n_days_21', 0), K=10, default=away_era_season)
        f['away_sp_whip_blend'] = self._blend(away_whip_season, away_whip_l21, row.get('away_sp_n_days_21', 0), K=10, default=away_whip_season)

        # Update pitcher quality using blended ERA
        f['home_sp_quality'] = max(0.0, 5.0 - f['home_sp_era_blend'])
        f['away_sp_quality'] = max(0.0, 5.0 - f['away_sp_era_blend'])

        # Bullpen recency (L7) - safe ERA only
        home_bp_era_season = _to_float(row.get('home_bp_era_roll', 4.20), 4.20)
        away_bp_era_season = _to_float(row.get('away_bp_era_roll', 4.20), 4.20)
        f['home_bp_era_blend'] = self._blend(home_bp_era_season, row.get('home_bp_era_l7'), row.get('home_bp_n_days_7', 0), K=5, default=home_bp_era_season)
        f['away_bp_era_blend'] = self._blend(away_bp_era_season, row.get('away_bp_era_l7'), row.get('away_bp_n_days_7', 0), K=5, default=away_bp_era_season)

        # Team offense recent (L14) from enhanced_games, blend with season/as-of runs_pg
        home_rpg_season = _to_float(row.get('home_runs_pg_roll', 4.5), 4.5)
        away_rpg_season = _to_float(row.get('away_runs_pg_roll', 4.5), 4.5)
        f['home_runs_pg_l14'] = _to_float(row.get('home_runs_pg_l14_eg', np.nan), np.nan)
        f['away_runs_pg_l14'] = _to_float(row.get('away_runs_pg_l14_eg', np.nan), np.nan)
        f['home_rpg_blend']   = self._blend(home_rpg_season, f['home_runs_pg_l14'], row.get('home_off_n_games_14', 0), K=8, default=home_rpg_season)
        f['away_rpg_blend']   = self._blend(away_rpg_season, f['away_runs_pg_l14'], row.get('away_off_n_games_14', 0), K=8, default=away_rpg_season)

        # Also blend recent allowed runs (defense/form)
        home_rapg_season = _to_float(row.get('home_rapg_season', 4.5), 4.5)  # from team_stats if available
        away_rapg_season = _to_float(row.get('away_rapg_season', 4.5), 4.5)
        f['home_rapg_l14'] = _to_float(row.get('home_rapg_l14_eg', np.nan), np.nan)
        f['away_rapg_l14'] = _to_float(row.get('away_rapg_l14_eg', np.nan), np.nan)
        f['home_rapg_blend'] = self._blend(home_rapg_season, f['home_rapg_l14'], row.get('home_off_n_games_14', 0), K=8, default=home_rapg_season)
        f['away_rapg_blend'] = self._blend(away_rapg_season, f['away_rapg_l14'], row.get('away_off_n_games_14', 0), K=8, default=away_rapg_season)

        # Hot/cold momentum ratios (safe version)
        f['home_sp_era_ratio'] = (home_era_l21 / home_era_season) if (home_era_l21 is not None and np.isfinite(float(home_era_l21)) and home_era_season > 0) else 1.0
        f['away_sp_era_ratio'] = (away_era_l21 / away_era_season) if (away_era_l21 is not None and np.isfinite(float(away_era_l21)) and away_era_season > 0) else 1.0
        f['home_rpg_ratio'] = (f['home_runs_pg_l14'] / home_rpg_season) if (np.isfinite(f['home_runs_pg_l14']) and home_rpg_season > 0) else 1.0
        f['away_rpg_ratio'] = (f['away_runs_pg_l14'] / away_rpg_season) if (np.isfinite(f['away_runs_pg_l14']) and away_rpg_season > 0) else 1.0

        # --- Your existing incremental team priors (kept; they self-decay monthly) ---
        for team, prefix in ((home, 'home'), (away, 'away')):
            s = self.team_stats.get(team, None)
            if s is None or s.get('games_played', 0) == 0:
                f[f'{prefix}_recent_runs']     = 4.5
                f[f'{prefix}_recent_against']  = 4.5
                f[f'{prefix}_rpg_season']      = 4.5
                f[f'{prefix}_rapg_season']     = 4.5
                f[f'{prefix}_run_diff_recent'] = 0.0
            else:
                gp = max(1, int(s['games_played']))
                f[f'{prefix}_recent_runs']     = _to_float(s['recent_runs_ewma'], 4.5)
                f[f'{prefix}_recent_against']  = _to_float(s['recent_against_ewma'], 4.5)
                f[f'{prefix}_rpg_season']      = s['total_runs_for'] / gp
                f[f'{prefix}_rapg_season']     = s['total_runs_against'] / gp
                f[f'{prefix}_run_diff_recent'] = f[f'{prefix}_recent_runs'] - f[f'{prefix}_recent_against']

        # Matchup history (recent L5)
        key = f"{home}_vs_{away}"; rkey = f"{away}_vs_{home}"
        hist = (self.matchup_history.get(key, []) + self.matchup_history.get(rkey, []))[-5:]
        if hist:
            f['matchup_avg'] = float(np.mean([_to_float(g['total_runs'], 9.0) for g in hist]))
            f['matchup_games'] = float(len(hist))
        else:
            f['matchup_avg'] = 9.0; f['matchup_games'] = 0.0

        # --- Derived combos (using blended values) ---
        f['pitching_quality_diff'] = f['home_sp_quality'] - f['away_sp_quality']
        f['offense_strength_roll'] = f['home_rpg_blend'] + f['away_rpg_blend']  # Use blended runs
        f['offense_form_l10']      = f['home_runs_pg_l10'] + f['away_runs_pg_l10']
        f['team_wrcplus_sum']      = f['home_wrcplus'] + f['away_wrcplus']
        f['vs_split_xwoba_sum']    = f['home_vs_split_xwoba'] + f['away_vs_split_xwoba']
        f['bullpen_era_sum']       = f['home_bp_era_blend'] + f['away_bp_era_blend']  # Use blended BP ERA
        f['park_offense_factor']   = f['park_pf_runs'] * (1.0 + 0.05 * f['park_altitude_kft'])
        f['temp_wind_boost']       = (temp - 72.0) * f['wind_factor']
        f['roof_dampener']         = 0.90 if f['park_roof_closed'] == 1.0 else 1.00

        # --- Hot/cold momentum signals (safe version) ---
        f['offense_momentum_edge'] = f['home_rpg_ratio'] - f['away_rpg_ratio']
        f['pitching_momentum_edge'] = (2.0 - f['away_sp_era_ratio']) - (2.0 - f['home_sp_era_ratio'])  # Lower ERA ratios = better
        
        # Volatility indicators (when ratios are extreme, prediction may be less reliable)
        f['sp_volatility'] = abs(f['home_sp_era_ratio'] - 1.0) + abs(f['away_sp_era_ratio'] - 1.0)
        f['offense_volatility'] = abs(f['home_rpg_ratio'] - 1.0) + abs(f['away_rpg_ratio'] - 1.0)

        return f

    # ----------------------- Team-level prediction -----------------------

    def _predict_team_with_sigma(self, Xs: np.ndarray, is_home: bool) -> Tuple[float, float]:
        if is_home:
            mean = float(self.home_model.predict(Xs)[0])
            logv = float(self.home_var_model.predict(Xs)[0])
            bias = self.home_bias_mu
        else:
            mean = float(self.away_model.predict(Xs)[0])
            logv = float(self.away_var_model.predict(Xs)[0])
            bias = self.away_bias_mu
        mean = max(0.0, mean + bias)  # clamp to 0 to avoid negatives
        raw_sigma = float(np.sqrt(np.exp(logv)))
        # cap per-team sigma to [0.4, 4.0] to prevent absurd totals
        sigma = float(np.clip(raw_sigma, 0.4, 4.0))
        return mean, sigma

    # ----------------------- Context-Aware Conformal Methods -----------------------
    
    def _bucket_from_market(self, market_total: float) -> str:
        """Categorize games by market total for context-aware intervals"""
        if market_total <= 7.5: 
            return 'low'
        if market_total >= 10.0: 
            return 'high'
        return 'mid'

    def _conformal_margin(self, market_total: float, sigma_total: float) -> float:
        """Return PI half-width using bucketed quantile of |residual|/sigma_total."""
        bucket = self._bucket_from_market(float(market_total or 9.0))
        zbuf = self.conformal_zscores[bucket]
        sigma = max(0.4, float(sigma_total))

        # small configurable cushion to ensure target coverage
        safety = float(os.getenv('PI_SAFETY', '1.10'))  # 10% wider by default

        # conservative prior
        prior_q = 1.55

        if len(zbuf) < 300:
            q = prior_q
        else:
            recent = zbuf[-self.conformal_window:] if len(zbuf) > self.conformal_window else zbuf
            # use the 0.80 quantile for 80% two-sided coverage on |res|
            q_emp = float(np.quantile(recent, self.coverage_target))  # 0.80
            q = 0.6 * q_emp + 0.4 * prior_q
            q = float(np.clip(q, 1.1, 3.0))

        return safety * q * sigma

    def _recent_iso(self, n: int = 800):
        """Past-only isotonic from most recent N market diffs/outcomes."""
        from sklearn.isotonic import IsotonicRegression
        iso = None
        if len(self.market_diffs) >= 200:
            diffs = self.market_diffs[-min(n, len(self.market_diffs)):]
            outs  = self.market_outcomes[-min(n, len(self.market_outcomes)):]
            try:
                iso = IsotonicRegression(out_of_bounds='clip').fit(diffs, outs)
            except Exception:
                iso = None
        return iso

    def _recent_density(self, diff: float, band: float = 0.6, n: int = 1000) -> float:
        """Fraction of recent diffs that fall within ±band of this diff."""
        if not self.market_diffs:
            return 0.0
        recent = self.market_diffs[-min(n, len(self.market_diffs)):]
        if not recent:
            return 0.0
        c = sum(1 for d in recent if abs(d - diff) <= band)
        frac = c / max(1, len(recent))
        return float(np.clip(frac / 0.08, 0.0, 1.0))  # ~8% within band → good density

    def _trust_score(self, sigma_total: float, margin: float, diff: float,
                     sgd_total: float, rf_total: Optional[float]) -> float:
        """0..1 trust score combining volatility, width, density, agreement."""
        # volatility: 1 at ~1.2; 0 at ~3.4
        var = float(np.clip(1.0 - (sigma_total - 1.2) / 2.2, 0.0, 1.0))
        # width: 1 when PI width<=12; 0 when >=30
        width = float(np.clip(1.0 - ((2*margin) - 12.0) / 18.0, 0.0, 1.0))
        # density around this diff
        dens = self._recent_density(diff, band=0.6, n=1000)
        # model agreement (SGD vs RF)
        agree = 1.0
        if rf_total is not None:
            agree = float(np.clip(1.0 - abs((rf_total - sgd_total)) / 1.4, 0.0, 1.0))
        return 0.35*var + 0.25*width + 0.25*dens + 0.15*agree

    def _best_totals_odds_for_game(self, game_id: str, game_date, preferred_books: Optional[List[str]] = None) -> dict:
        """
        Return best OVER/UNDER odds for this game_id/date; restrict to preferred_books if provided.
        Fallback to -110 if none found.
        """
        try:
            q = text("""
                SELECT book, total, over_odds, under_odds
                FROM totals_odds
                WHERE game_id = :gid AND date = :d
            """)
            df = pd.read_sql(q, self.engine, params={'gid': str(game_id), 'd': pd.to_datetime(game_date).date()})
            if df.empty:
                return {'over_odds': -110, 'under_odds': -110, 'book_over': 'N/A', 'book_under': 'N/A'}

            if preferred_books:
                df = df[df['book'].isin(preferred_books)]
                if df.empty:
                    return {'over_odds': -110, 'under_odds': -110, 'book_over': 'N/A', 'book_under': 'N/A'}

            # pick the *best* (highest) over and under odds within the filter
            over_row = df.loc[df['over_odds'].astype(int).idxmax()]
            under_row = df.loc[df['under_odds'].astype(int).idxmax()]
            return {
                'over_odds': int(over_row['over_odds']),
                'under_odds': int(under_row['under_odds']),
                'book_over': str(over_row['book']),
                'book_under': str(under_row['book']),
            }
        except Exception:
            return {'over_odds': -110, 'under_odds': -110, 'book_over': 'N/A', 'book_under': 'N/A'}

    def _blend(self, season_val, recent_val, n_recent, K=7, default=None):
        """Empirical Bayes blend: weight recent by n/(n+K)."""
        if recent_val is None or not np.isfinite(float(recent_val or 0)) or n_recent is None:
            return season_val if season_val is not None else (default if default is not None else 0.0)
        w = float(n_recent) / (float(n_recent) + float(K))
        base = season_val if (season_val is not None and np.isfinite(float(season_val))) else (default if default is not None else 0.0)
        return (1.0 - w) * base + w * float(recent_val)

    def _update_conformal(self, y_true: float, y_pred: float, market_total: float, sigma_total: float):
        """Update raw & sigma-normalized residual buffers per market bucket."""
        res = abs(y_true - y_pred)
        bucket = self._bucket_from_market(float(market_total or 9.0))

        # keep raw residuals for diagnostics (already had this)
        self.conformal_bucket_scores[bucket].append(res)
        if len(self.conformal_bucket_scores[bucket]) > 5000:
            self.conformal_bucket_scores[bucket] = self.conformal_bucket_scores[bucket][-5000:]

        # z-score buffer (residual scaled by model sigma)
        z = res / max(0.4, float(sigma_total))
        self.conformal_zscores[bucket].append(float(z))
        if len(self.conformal_zscores[bucket]) > 5000:
            self.conformal_zscores[bucket] = self.conformal_zscores[bucket][-5000:]

    def _conformal_margin_legacy(self) -> float:
        if len(self.conformal_scores) < 50:
            return 1.282 * 2.0  # ~80% two-sided with rough 2-run sigma
        scores = np.array(self.conformal_scores[-self.conformal_window:])
        alpha = 1 - self.coverage_target
        q = np.quantile(scores, 1 - alpha)
        return float(q)

    def _update_market_calibration(self, pred_total: float, market_total: float, actual_total: float):
        if market_total and market_total > 0:
            diff = float(pred_total - market_total)
            outcome = 1.0 if actual_total > market_total else 0.0
            self.market_diffs.append(diff); self.market_outcomes.append(outcome)
            if len(self.market_diffs) >= 120 and len(self.market_diffs) % 40 == 0:
                try:
                    self.market_calibrator.fit(self.market_diffs, self.market_outcomes)
                except Exception:
                    pass

    def _update_prob_calibration(self, pred_total: float, market_total: float,
                                 actual_total: float, feats: dict, sigma_total: float):
        if market_total and market_total > 0:
            diff = float(pred_total - market_total)
            outcome = 1.0 if actual_total > market_total else 0.0
            self.prob_cal.update(diff, outcome, feats, sigma_total, float(market_total))

    # ----------------------- Export / Persist / Slate -----------------------

    def predictions_to_dataframe(self, predictions: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(predictions)
        base_cols = [
            'game_id','date','home_team','away_team','pred_total','pred_home','pred_away',
            'actual_total','lower_80','upper_80','market_total','edge','sigma_indep'
        ]
        for c in base_cols:
            if c not in df.columns:
                df[c] = np.nan
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(['date','game_id'], ascending=[True, True], ignore_index=True)
            except Exception:
                pass
        return df

    def persist_predictions_to_db(self, df: pd.DataFrame, table: str = 'team_level_predictions') -> None:
        try:
            df.to_sql(table, self.engine, if_exists='append', index=False, method='multi', chunksize=1000)
            print(f"🗄️  Persisted {len(df)} rows to {table}")
        except Exception as e:
            print(f"⚠️  Could not persist predictions to DB: {e}")

    def export_predictions_to_files(self, df: pd.DataFrame, outdir: str = 'outputs', tag: str = 'backtest') -> Dict[str, str]:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        ts = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
        csv_path = str(Path(outdir) / f'{tag}_predictions_{ts}.csv')
        pq_path  = str(Path(outdir) / f'{tag}_predictions_{ts}.parquet')
        try:
            df.to_csv(csv_path, index=False)
            df.to_parquet(pq_path, index=False)
            print(f"📁 Wrote {csv_path}\n📦 Wrote {pq_path}")
        except Exception as e:
            print(f"⚠️  Export failed: {e}")
        return {'csv': csv_path, 'parquet': pq_path}

    # ---------- Odds & ROI utilities ----------

    def _american_to_decimal(self, american: int) -> float:
        a = int(american)
        return 1.0 + (a/100.0 if a > 0 else 100.0/abs(a))

    def _breakeven_prob(self, american: int) -> float:
        # p* = 1 / decimal_odds
        return 1.0 / self._american_to_decimal(american)
    
    def recommend_slate_bets(self, target_date: str, thresholds: Optional[dict] = None) -> pd.DataFrame:
        """
        If thresholds=None, auto-calibrate from recent books.
        thresholds dict keys: min_trust, min_ev, min_diff, max_sigma, odds_floor, max_picks
        """
        if thresholds is None:
            thresholds = self.calibrate_thresholds_from_books()

        df = self.predict_future_slate(target_date)
        if df.empty:
            return df

        def odds_ok(o): 
            try: return int(o) >= int(thresholds['odds_floor'])
            except: return False

        picks = df[
            (df['trust'] >= thresholds['min_trust']) &
            (df['ev']    >= thresholds['min_ev']) &
            (df['sigma_indep'] <= thresholds['max_sigma']) &
            (df['diff'].abs()  >= thresholds['min_diff']) &
            df['best_odds'].apply(odds_ok)
        ].copy()

        picks = picks.sort_values(['trust','ev'], ascending=[False, False]).head(int(thresholds['max_picks'])).reset_index(drop=True)
        if not picks.empty:
            out = self.export_predictions_to_files(picks, outdir='outputs', tag=f'slate_recs_{target_date}')
            print(f"� Recommended {len(picks)} plays saved: {out['csv']}")
        else:
            print("🤷 No high-confidence plays for the current thresholds.")
        return picks

    def make_daily_onepager(self, target_date: str, thresholds: Optional[dict] = None, max_plays: int = 3) -> str:
        """
        Write outputs/onepager_<date>.md with up to 3 high-confidence plays.
        Uses recommend_slate_bets() and prints stake suggestions (trust-weighted Kelly).
        Returns filepath.
        """
        if thresholds is None:
            thresholds = self.calibrate_thresholds_from_books(max_picks=max_plays)

        picks = self.recommend_slate_bets(target_date, thresholds)
        outdir = Path('outputs'); outdir.mkdir(exist_ok=True, parents=True)
        path = outdir / f"onepager_{target_date}.md"

        if picks is None or picks.empty:
            path.write_text(f"# {target_date} — Totals One-Pager\n\nNo high-confidence totals today.\n")
            print(f"🧾 One-pager: {path}")
            return str(path)

        lines = [f"# {target_date} — Top {min(max_plays,len(picks))} Totals Plays\n"]
        base_bankroll = float(getattr(self, 'display_bankroll', 10000.0))

        def dec(a): return 1.0 + (a/100.0 if a>0 else 100.0/abs(a))

        for i, r in picks.head(max_plays).iterrows():
            side = r['best_side']
            odds = int(r['best_odds'])
            p_win = float(r['p_over'] if side=='OVER' else (1.0 - r['p_over']))
            b = dec(odds) - 1.0
            # trust-weighted 1/5 Kelly, capped at 1% BR
            f_star = max(0.0, (p_win*b - (1.0 - p_win)) / max(b,1e-9))
            stake_pct = min(0.01, 0.20 * f_star * float(r['trust']))
            stake_dollars = round(base_bankroll * stake_pct, 2)

            lines += [
                f"## {r['away_team']} @ {r['home_team']}",
                f"- **Market**: {r['market_total']:.1f} | **Model**: {r['pred_total']:.2f} | **80% PI**: [{r['lower_80']:.1f}, {r['upper_80']:.1f}]",
                f"- **Pick**: **{side} {r['market_total']:.1f}** @ **{odds}** ({r['book']})",
                f"- **Edge**: EV={r['ev']:.3f} | P(win)≈{p_win:.3f} | Trust={r['trust']:.2f} | σ={r['sigma_indep']:.2f}",
                f"- **Stake guide**: ~{stake_pct*100:.2f}% BR  (~${stake_dollars})",
                ""
            ]

        path.write_text("\n".join(lines))
        print(f"🧾 One-pager: {path}")
        return str(path)

    def _kelly_fraction(self, p: float, american: int) -> float:
        # Kelly f* = (p*(b) - (1-p)) / b , b = decimal_odds-1
        b = self._american_to_decimal(american) - 1.0
        return max(0.0, (p*b - (1.0 - p)) / max(b, 1e-9))

    def _ev_roi(self, p: float, american: int) -> float:
        # Expected ROI per $1 stake at given odds
        dec = self._american_to_decimal(american)
        return p*(dec - 1.0) - (1.0 - p)

    def calibrate_thresholds_from_books(self, lookback_days: int = 30,
                                        target_iso_margin: float = 0.07,
                                        max_picks: int = 6) -> dict:
        """
        Inspect recent totals odds to set sensible gates:
          - min_ev: tied to market hold
          - odds_floor: avoid ugly juice
          - min_diff: where iso predicts ~7% edge (p>=0.57 or <=0.43)
          - min_trust/max_sigma: stricter if hold is fat
        """
        # 1) pull recent odds
        try:
            q = text("""
                SELECT date, book, over_odds, under_odds
                FROM totals_odds
                WHERE date >= CURRENT_DATE - INTERVAL :lb || ' day'
            """)
            df = pd.read_sql(q, self.engine, params={'lb': int(lookback_days)})
            if self.preferred_books:
                df = df[df['book'].isin(self.preferred_books)]
        except Exception:
            df = pd.DataFrame()

        # 2) estimate hold (juice) distribution
        def imp_prob(am):
            am = int(am)
            return 100/(am+100) if am>0 else abs(am)/(abs(am)+100)
        if not df.empty:
            holds = imp_prob(df['over_odds'].astype(int)) + imp_prob(df['under_odds'].astype(int)) - 1.0
            med_hold = float(np.nanmedian(holds))
            p25_over = int(np.nanpercentile(df['over_odds'].astype(int), 25))
            p25_under = int(np.nanpercentile(df['under_odds'].astype(int), 25))
            odds_floor = max(p25_over, p25_under, -120)  # don't accept worse than -120
        else:
            med_hold = 0.045  # ~4.5% typical
            odds_floor = -118

        # 3) min_ev needs to clear hold with a cushion
        min_ev = float(np.clip(1.15*med_hold, 0.04, 0.10))

        # 4) min_diff from isotonic (past-only)
        iso = self._recent_iso(n=1000)
        if iso is not None:
            grid = np.linspace(-3.0, 3.0, 121)
            probs = iso.predict(grid)
            # smallest |diff| where |p-0.5| >= target_iso_margin
            mask = np.where(np.abs(probs - 0.5) >= target_iso_margin)[0]
            min_diff = float(abs(grid[mask[0]])) if mask.size else 1.0
            min_diff = float(max(0.8, min(1.4, min_diff)))
        else:
            min_diff = 1.0

        # 5) trust/sigma tuned by hold severity
        if med_hold >= 0.055:
            min_trust, max_sigma = 0.65, 3.00
        else:
            min_trust, max_sigma = 0.60, 3.20

        return {
            'min_trust': min_trust,
            'min_ev': min_ev,
            'min_diff': min_diff,
            'max_sigma': max_sigma,
            'odds_floor': int(odds_floor),
            'max_picks': int(max_picks),
        }

    # ---------- Bet simulator on exported predictions ----------

    def simulate_bets(self, df_preds: pd.DataFrame,
                      ev_min: float = 0.03,        # require +3% EV
                      diff_min: float = 0.6,       # require model vs market >= 0.6 runs
                      kelly_fraction: float = 0.25,# use 1/4 Kelly
                      bankroll: float = 100.0,   # paper bankroll
                      max_bet: float = 200.0,      # stake cap
                      outdir: str = 'outputs',
                      tag: str = 'bets') -> pd.DataFrame:
        """
        Use the isotonic calibrator to map (pred_total - market_total) -> P(Over),
        then compute EV for Over and Under at best available odds from totals_odds.
        Place a bet only if EV >= ev_min and |diff| >= diff_min.
        Stake = kelly_fraction * Kelly(*).
        """
        if df_preds is None or df_preds.empty:
            print("⚠️  No predictions to simulate bets.")
            return pd.DataFrame()

        rows = []
        equity = bankroll

        # ensure calibrator is trained (we already update it online during training)
        have_cal = len(self.market_diffs) >= 120

        for _, r in df_preds.iterrows():
            game_id = r['game_id']
            date = r['date']
            pred_total = float(r['pred_total'])
            market_total = float(r.get('market_total', 9.0))
            actual_total = float(r.get('actual_total', np.nan))  # may be NaN for pure future slates
            diff = pred_total - market_total

            # Probabilities
            p_over = float(self.prob_cal.predict(diff, feats={}, sigma=1.0, market_total=market_total))
            p_under = 1.0 - p_over

            # Odds (best across books today)
            odds = self._best_totals_odds_for_game(str(game_id), date, self.preferred_books)
            over_odds  = odds['over_odds']
            under_odds = odds['under_odds']

            # EVs
            ev_over  = self._ev_roi(p_over,  over_odds)
            ev_under = self._ev_roi(p_under, under_odds)

            # Pick side
            side = None; use_odds = None; p = None; ev = None; book = None
            if ev_over >= ev_under:
                side, use_odds, p, ev, book = ('OVER', over_odds, p_over, ev_over, odds['book_over'])
            else:
                side, use_odds, p, ev, book = ('UNDER', under_odds, p_under, ev_under, odds['book_under'])

            place = (abs(diff) >= diff_min) and (ev is not None and ev >= ev_min)
            stake = 0.0; profit = np.nan; won = np.nan

            if place:
                # Fractional Kelly stake
                f_star = self._kelly_fraction(p, use_odds)
                stake = float(min(max_bet, equity * kelly_fraction * f_star))
                if stake > 0 and np.isfinite(actual_total):
                    # settle if we know result
                    if (side == 'OVER' and actual_total > market_total) or (side == 'UNDER' and actual_total < market_total):
                        profit = stake * (self._american_to_decimal(use_odds) - 1.0)  # net profit
                        won = True
                    else:
                        profit = -stake
                        won = False
                    equity += profit

            rows.append({
                'date': pd.to_datetime(date).date(),
                'game_id': game_id,
                'home_team': r['home_team'],
                'away_team': r['away_team'],
                'pred_total': pred_total,
                'market_total': market_total,
                'diff': diff,
                'side': side,
                'odds': use_odds,
                'book': book,
                'p_win': p,
                'ev': ev,
                'stake': round(stake, 2),
                'result_total': actual_total,
                'won': won,
                'profit': profit if np.isfinite(actual_total) else np.nan,
                'bankroll_after': round(equity, 2) if np.isfinite(actual_total) else np.nan,
            })

        bets = pd.DataFrame(rows)
        # Summary
        settled = bets[bets['profit'].notna()]
        n_placed = (bets['stake'] > 0).sum()
        roi = settled['profit'].sum() / max(1.0, settled['stake'].sum()) if not settled.empty else 0.0
        hit = settled['won'].mean() if not settled.empty else np.nan
        print(f"🧾 Bet sim: placed {n_placed} bets | hit={hit:.1%} | ROI={roi:+.2%} | P&L=${settled['profit'].sum():.2f}")

        # Export
        files = self.export_predictions_to_files(bets, outdir=outdir, tag=tag)
        print(f"🗃️  Bet slip saved: {files['csv']}")

        return bets


    def daily_rollup_from_predictions(self, df_preds: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-day scorecard: coverage, MAE, interval width, edge, ROI."""
        if df_preds is None or df_preds.empty:
            return pd.DataFrame()
        df = df_preds.copy()
        # Ensure types
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['abs_err'] = (df['actual_total'] - df['pred_total']).abs()
        df['in_80'] = (df['actual_total'] >= df['lower_80']) & (df['actual_total'] <= df['upper_80'])
        df['width_80'] = (df['upper_80'] - df['lower_80']).astype(float)
        # Same simple staking as backtest (only when |edge|>0.02)
        df['bet'] = df['edge'].abs() > 0.02
        df['won'] = np.where(df['bet'], np.where(df['edge'] > 0, df['actual_total'] > df['market_total'], df['actual_total'] < df['market_total']), np.nan)
        df['roi'] = np.where(df['bet'], np.where(df['won'], 0.9091, -1.0), np.nan)
        daily = df.groupby('date', as_index=False).agg(
            n=('game_id','count'),
            coverage_80=('in_80','mean'),
            mae=('abs_err','mean'),
            avg_width_80=('width_80','mean'),
            avg_edge=('edge','mean'),
            roi=('roi','mean')
        )
        return daily

    def persist_daily_rollup_to_db(self, df_daily: pd.DataFrame, table: str = 'team_level_daily_rollup') -> None:
        try:
            df_daily.to_sql(table, self.engine, if_exists='append', index=False, method='multi', chunksize=1000)
            print(f"🗄️  Persisted {len(df_daily)} daily rows to {table}")
        except Exception as e:
            print(f"⚠️  Could not persist daily rollup to DB: {e}")

    def get_upcoming_games_for_date(self, date_str: str) -> pd.DataFrame:
        # Normalize param to a date object
        d = pd.to_datetime(date_str).date()

        # Try daily_games + totals_odds first
        q1 = text("""
            SELECT dg.game_id, dg.date, dg.home_team, dg.away_team,
                dg.temperature, dg.wind_speed, dg.venue_name,
                dg.home_sp_id, dg.away_sp_id,
                COALESCE(tood.opening_total, rmg.opening_total, 9.0) AS opening_total,
                dg.start_ts
            FROM daily_games dg
            LEFT JOIN totals_odds tood ON dg.game_id = tood.game_id
            LEFT JOIN real_market_games rmg ON dg.game_id = rmg.game_id
            WHERE dg.date = :d
        """)
        try:
            df = pd.read_sql(q1, self.engine, params={'d': d})
            if not df.empty:
                return df
        except Exception:
            pass  # fall through

        # Fallback: api_games_today + real_market_games
        q2 = text("""
            SELECT
                agt.game_id,
                agt.game_date AS date,
                agt.home_team, agt.away_team,
                72 AS temperature, 5 AS wind_speed, 'Unknown' AS venue_name,
                0 AS home_sp_id, 0 AS away_sp_id,
                COALESCE(agt.market_total, rmg.opening_total, 9.0) AS opening_total,
                agt.game_date AS start_ts
            FROM api_games_today agt
            LEFT JOIN real_market_games rmg USING (game_id)
            WHERE agt.game_date = :d
        """)
        try:
            return pd.read_sql(q2, self.engine, params={'d': d})
        except Exception as e:
            print(f"⚠️  Could not load upcoming games for {date_str}: {e}")
            return pd.DataFrame()



    def predict_future_slate(self, target_date: str, outdir: str = 'outputs') -> pd.DataFrame:
        games = self.get_upcoming_games_for_date(target_date)
        if games.empty:
            print(f"⚠️  No upcoming games found for {target_date}")
            return games

        # recent past-only isotonic for live EV
        iso = self._recent_iso(n=800)
        rows = []

        for _, g in games.iterrows():
            feats = self.engineer_features_from_history(g)
            feats = self._clean_features(feats)  # Clean NaN/inf values
            x = np.array([[feats.get(c, 0.0) for c in self.feature_columns]], dtype=float)
            xs = self.scaler.transform(x)

            # SGD heads (with heteroskedastic sigmas)
            h_sgd, hs = self._predict_team_with_sigma(xs, True)
            a_sgd, as_ = self._predict_team_with_sigma(xs, False)
            sgd_total = h_sgd + a_sgd
            sigma_total = float(np.sqrt(hs**2 + as_**2))

            # RF blend for agreement & a little nonlinearity
            rf_total = None
            try:
                if hasattr(self, 'home_batch_model') and hasattr(self, 'away_batch_model'):
                    h_rf = float(self.home_batch_model.predict(xs)[0])
                    a_rf = float(self.away_batch_model.predict(xs)[0])
                    rf_total = h_rf + a_rf
                    pred_total = 0.6*sgd_total + 0.4*rf_total
                else:
                    pred_total = sgd_total
            except Exception:
                pred_total = sgd_total

            # variance boosters (same logic as training)
            is_high_pf  = 1.0 if (feats.get('park_pf_runs', 1.0) >= 1.05 or feats.get('park_altitude_kft', 0.0) >= 1.0) else 0.0
            strong_wind = 1.0 if feats.get('wind_speed', 0.0) >= 12 else 0.0
            hot         = 1.0 if feats.get('temperature', 72.0) >= 88 else 0.0
            sigma_adj   = sigma_total * (1.0 + self.var_boost['pf']*is_high_pf + self.var_boost['wind']*strong_wind + self.var_boost['heat']*hot)

            market_total = float(feats.get('market_total', 9.0))
            diff = pred_total - market_total

            # context-aware conformal for this game (✔ bug: compute per-game)
            margin = self._conformal_margin(market_total, sigma_adj)
            low  = max(0.0, pred_total - margin)
            high = pred_total + margin

            # past-only probability map for EV
            p_over = float(self.prob_cal.predict(diff, feats, sigma_adj, market_total))

            trust = self._trust_score(sigma_adj, margin, diff, sgd_total, rf_total)
            odds = self._best_totals_odds_for_game(str(g['game_id']), g['date'], self.preferred_books)
            dec_over  = self._american_to_decimal(odds['over_odds'])
            dec_under = self._american_to_decimal(odds['under_odds'])
            ev_over   = p_over*(dec_over-1.0) - (1.0 - p_over)
            ev_under  = (1.0-p_over)*(dec_under-1.0) - p_over

            side = 'OVER' if ev_over >= ev_under else 'UNDER'
            ev   = max(ev_over, ev_under)
            use_odds = odds['over_odds'] if side=='OVER' else odds['under_odds']
            book = odds['book_over'] if side=='OVER' else odds['book_under']

            rows.append({
                'date': g['date'], 'game_id': g['game_id'],
                'home_team': g['home_team'], 'away_team': g['away_team'],
                'market_total': market_total,
                'pred_total': round(pred_total, 2),
                'pred_home': round(h_sgd, 2), 'pred_away': round(a_sgd, 2),
                'sigma_indep': round(sigma_adj, 3),
                'lower_80': round(low, 2), 'upper_80': round(high, 2),
                'diff': round(diff, 2),
                'p_over': round(p_over, 3),
                'best_side': side, 'best_odds': use_odds, 'book': book,
                'ev': round(ev, 3),
                'trust': round(trust, 3),
            })

        df = pd.DataFrame(rows)
        # keep & export everything; you can filter downstream on trust/ev/diff
        self.export_predictions_to_files(df, outdir=outdir, tag=f'slate_{target_date}')
        return df

    # ----------------------- Training (team-level, conformal) -----------------------

    def team_level_incremental_learn(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        print('🧠 STARTING TEAM-LEVEL INCREMENTAL LEARNING (80% coverage target)')
        df = self.get_completed_games_chronological(start_date=start_date, end_date=end_date)
        if df.empty:
            return None
        
        # Coverage report for data quality
        self.coverage_report(df)

        # Establish feature schema from first row
        first_feats = self.engineer_features_from_history(df.iloc[0])
        first_feats = self._clean_features(first_feats)  # Clean NaN/inf values

        # If we already have a fitted state, keep its schema unless we're migrating.
        if self.is_fitted and self.feature_columns:
            print(f"📎 Keeping existing {len(self.feature_columns)}-feature schema from state.")
        else:
            # Either no state or we're starting fresh → use the current schema
            self.feature_columns = list(first_feats.keys())

        # If lengths disagree (old state vs new code), reset and adopt the new schema
        try:
            fitted_n = int(getattr(self.scaler, 'mean_', np.array([])).shape[0])
        except Exception:
            fitted_n = 0

        if fitted_n and fitted_n != len(self.feature_columns):
            print(f"⚠️ Feature schema changed (old={fitted_n}, new={len(self.feature_columns)}). Resetting models & scaler.")
            self._reset_models_and_scaler()
            self.feature_columns = list(first_feats.keys())

        # Adaptive warmup for small datasets
        n = len(df)
        if n < self.warmup_n:
            # Use a generous fraction of what we have; ensure at least 50 and < n
            new_warm = max(50, min(self.warmup_n, int(0.6 * n)))
            if new_warm >= n:
                new_warm = max(10, n - 5)  # keep a few for post-warmup predictions
            print(f"🕐 Warmup reduced from {self.warmup_n} → {new_warm} (n={n})")
            self.warmup_n = new_warm

        print(f"📊 Using {len(self.feature_columns)} engineered features")
        print(f"🕐 Warmup: {self.warmup_n} games")

        preds: List[Dict[str, Any]] = []
        coverage_flags: List[bool] = []
        market_roi: List[float] = []
        retrain_count = 0

        for i in range(len(df)):
            row = df.iloc[i]
            self._maybe_monthly_decay(str(row['date'])[:7])

            feats = self.engineer_features_from_history(row)
            feats = self._clean_features(feats)  # Clean NaN/inf values
            x_raw = np.array([[feats.get(c, 0.0) for c in self.feature_columns]], dtype=float)

            # Warmup accumulation
            if i < self.warmup_n:
                self.warmup_X.append(list(x_raw[0]))
                self.warmup_y_home.append(_to_float(row['home_score']))
                self.warmup_y_away.append(_to_float(row['away_score']))
                self.update_team_stats(row)  # update after game
                if i + 1 == self.warmup_n:
                    Xw = np.array(self.warmup_X, dtype=float)
                    # Handle NaN values before fitting
                    nan_mask = np.isnan(Xw).any(axis=1)
                    if nan_mask.any():
                        print(f"⚠️  Removing {nan_mask.sum()} warmup samples with NaN features")
                        valid_idx = ~nan_mask
                        Xw = Xw[valid_idx]
                        warmup_y_home_clean = np.array(self.warmup_y_home)[valid_idx]
                        warmup_y_away_clean = np.array(self.warmup_y_away)[valid_idx]
                    else:
                        warmup_y_home_clean = np.array(self.warmup_y_home)
                        warmup_y_away_clean = np.array(self.warmup_y_away)
                        
                    self.scaler.fit(Xw)
                    Xw_s = self.scaler.transform(Xw)
                    # Fit mean models
                    self.home_model.fit(Xw_s, warmup_y_home_clean)
                    self.away_model.fit(Xw_s, warmup_y_away_clean)
                    # Fit variance heads
                    h_resid = warmup_y_home_clean - self.home_model.predict(Xw_s)
                    a_resid = warmup_y_away_clean - self.away_model.predict(Xw_s)
                    self.home_var_model.fit(Xw_s, np.log(np.maximum(h_resid**2, 1e-6)))
                    self.away_var_model.fit(Xw_s, np.log(np.maximum(a_resid**2, 1e-6)))
                    self.is_fitted = True
                    print('✅ Warmup complete — models initialized')
                continue

            # Optional scaler adaptation
            if self.update_scaler:
                self.scaler.partial_fit(x_raw)
            x = self.scaler.transform(x_raw)

            # Predict team-level means & sigmas
            home_pred, home_sig = self._predict_team_with_sigma(x, is_home=True)
            away_pred, away_sig = self._predict_team_with_sigma(x, is_home=False)
            total_pred = home_pred + away_pred
            sigma_indep = float(np.sqrt(home_sig**2 + away_sig**2))  # Independent team variance

            # === variance boosters for high-volatility contexts ===
            is_high_pf  = 1.0 if (feats.get('park_pf_runs', 1.0) >= 1.05 or feats.get('park_altitude_kft', 0.0) >= 1.0) else 0.0
            strong_wind = 1.0 if feats.get('wind_speed', 0.0) >= 12 else 0.0
            hot         = 1.0 if feats.get('temperature', 72.0) >= 88 else 0.0
            sigma_adj = sigma_indep * (
                1.0
                + self.var_boost['pf']   * is_high_pf
                + self.var_boost['wind'] * strong_wind
                + self.var_boost['heat'] * hot
            )

            # Defensive widening when key features are missing
            missing_penalty = 1.0
            if feats.get('park_pf_runs', 1.0) == 1.0 and feats.get('park_altitude_kft', 0.0) == 0.0:
                missing_penalty += 0.06    # park unknown
            if np.isnan(feats.get('home_vs_rhp_xwoba', np.nan)) or np.isnan(feats.get('away_vs_lhp_xwoba', np.nan)):
                missing_penalty += 0.04    # vs-hand splits missing
            sigma_adj *= missing_penalty

            # Market total for context-aware conformal
            market_total = feats.get('market_total', 9.0)
            
            # Conformal interval (context-aware, clipped to non-negative)
            margin = self._conformal_margin(market_total, sigma_adj)
            lower, upper = max(0.0, total_pred - margin), total_pred + margin

            # Soft cap for egregious widths (optional safety)
            width = upper - lower
            if width > 30.0:
                upper = lower + 30.0

            # Market signal & ROI tracking (stricter edge gate for profitability)
            self._update_prob_calibration(total_pred, market_total, _to_float(row['total_runs']), feats, sigma_adj)
            p_over_est = self.prob_cal.predict(total_pred - market_total, feats, sigma_adj, float(market_total))
            edge = float(p_over_est - 0.5)
            EDGE_MIN = 0.05  # Stricter threshold for betting

            actual_total = _to_float(row['total_runs'])
            in_interval = (lower <= actual_total <= upper)
            coverage_flags.append(in_interval)

            preds.append({
                'game_id': row['game_id'], 'date': row['date'],
                'home_team': row['home_team'], 'away_team': row['away_team'],
                'pred_total': total_pred, 'pred_home': home_pred, 'pred_away': away_pred,
                'actual_total': actual_total, 'lower_80': lower, 'upper_80': upper,
                'market_total': market_total, 'edge': edge, 'sigma_indep': sigma_indep,
            })

            # Update context-aware conformal with realized residual
            self._update_conformal(actual_total, total_pred, market_total, sigma_adj)

            # Incremental updates: bias, mean, variance
            h_true, a_true = _to_float(row['home_score']), _to_float(row['away_score'])
            h_resid = h_true - home_pred
            a_resid = a_true - away_pred
            self.home_bias_mu = (1 - self.bias_alpha) * self.home_bias_mu + self.bias_alpha * h_resid
            self.away_bias_mu = (1 - self.bias_alpha) * self.away_bias_mu + self.bias_alpha * a_resid
            self.home_model.partial_fit(x, [h_true])
            self.away_model.partial_fit(x, [a_true])
            self.home_var_model.partial_fit(x, [np.log(max(h_resid**2, 1e-6))])
            self.away_var_model.partial_fit(x, [np.log(max(a_resid**2, 1e-6))])
            self.home_residuals.append(float(h_resid)); self.away_residuals.append(float(a_resid))
            if len(self.home_residuals) > 1000:
                self.home_residuals = self.home_residuals[-1000:]
                self.away_residuals = self.away_residuals[-1000:]

            # Safer, more selective staking for ROI (stricter criteria)
            if market_total and abs(edge) >= EDGE_MIN:  # Use the stricter threshold
                bet_over = edge > 0
                won = (actual_total > market_total) if bet_over else (actual_total < market_total)
                market_roi.append(0.9091 if won else -1.0)  # -110 pricing

            # Periodic batch retrain (rolling window)
            if (i - self.last_retrain_at >= self.min_retrain_gap) and (i % self.retrain_every == 0):
                j0 = max(0, i - self.batch_window)
                win = df.iloc[j0:i+1]
                Xb = np.array([
                    [self._clean_features(self.engineer_features_from_history(r)).get(c, 0.0) for c in self.feature_columns]
                    for _, r in win.iterrows()
                ], dtype=float)
                Xb_s = self.scaler.transform(Xb)
                ybh = [ _to_float(r['home_score']) for _, r in win.iterrows() ]
                yba = [ _to_float(r['away_score']) for _, r in win.iterrows() ]
                if len(Xb_s) > 50:
                    self.home_batch_model.fit(Xb_s, ybh)
                    self.away_batch_model.fit(Xb_s, yba)
                    retrain_count += 1
                    self.last_retrain_at = i
                    print(f"   🔄 Retrained batch models on {len(Xb_s)} games (retrain #{retrain_count})")

            # Update team aggregates after game
            self.update_team_stats(row)

            # Progress ping every 200
            if (i + 1) % 200 == 0:
                cov = float(np.mean(coverage_flags[-200:])) if len(coverage_flags) >= 20 else 0.0
                mae = float(np.mean([abs(p['actual_total'] - p['pred_total']) for p in preds[-200:]]))
                roi = float(np.mean(market_roi[-100:])) if len(market_roi) >= 20 else 0.0
                print(f"   Game {i+1:,}: Coverage={cov:.1%} | MAE={mae:.2f} | ROI={roi:+.2%}")
                
                # Log missing features for debugging
                if feats:
                    miss = [k for k,v in feats.items() if v is None or (isinstance(v,float) and not np.isfinite(v))]
                    if miss: 
                        print(f"   ⚠️  missing features (sample): {miss[:8]}")

        # Final metrics (last 200 for stability)
        final_cov = float(np.mean(coverage_flags[-200:])) if len(coverage_flags) >= 200 else float(np.mean(coverage_flags) if coverage_flags else 0.0)
        final_mae = float(np.mean([abs(p['actual_total'] - p['pred_total']) for p in preds[-200:]]))
        final_roi = float(np.mean(market_roi[-100:])) if len(market_roi) >= 100 else 0.0

        print('✅ Team-level incremental learning complete!')
        print(f'📊 Predictions: {len(preds):,} | 80% Coverage (recent): {final_cov:.1%} | MAE: {final_mae:.2f} | ROI: {final_roi:+.2%}')

        return {
            'predictions': preds,
            'final_coverage': final_cov,
            'final_mae': final_mae,
            'final_roi': final_roi,
            'retrain_count': retrain_count,
        }

    # ----------------------- Inference -----------------------

    def predict_future_game(self, home_team: str, away_team: str, game_date: str, market_total: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if not self.is_fitted or not self.feature_columns:
            print('❌ Model not trained yet. Run team_level_incremental_learn() first.')
            return None
        mock = {
            'home_team': home_team, 'away_team': away_team,
            'date': game_date, 'opening_total': market_total or 9.0,
            'temperature': 72, 'wind_speed': 5,
            'venue_name': 'Unknown'
        }
        f = self.engineer_features_from_history(mock)
        f = self._clean_features(f)                 # <<< add this line
        x = np.array([[f.get(c, 0.0) for c in self.feature_columns]], dtype=float)
        xs = self.scaler.transform(x)

        h, home_sig = self._predict_team_with_sigma(xs, True)
        a, away_sig = self._predict_team_with_sigma(xs, False)
        tot = h + a
        sigma_total = float(np.sqrt(home_sig**2 + away_sig**2))
        
        # variance boosters (same logic as training)
        is_high_pf  = 1.0 if f.get('park_pf_runs', 1.0) >= 1.05 or f.get('park_altitude_kft', 0.0) >= 1.0 else 0.0
        strong_wind = 1.0 if f.get('wind_speed', 0.0) >= 12 else 0.0
        hot         = 1.0 if f.get('temperature', 72.0) >= 88 else 0.0
        sigma_total = sigma_total * (1.0 + self.var_boost['pf']*is_high_pf + self.var_boost['wind']*strong_wind + self.var_boost['heat']*hot)
        
        # Context-aware conformal margin, clipped to non-negative
        margin = self._conformal_margin(market_total or 9.0, sigma_total)
        low, high = max(0.0, tot - margin), tot + margin
        
        line = market_total or 9.0
        rec = 'OVER' if tot > line + 0.5 else ('UNDER' if tot < line - 0.5 else 'PASS')
        return {
            'pred_total': round(tot, 2), 'pred_home': round(h, 2), 'pred_away': round(a, 2),
            'interval_80': [round(low, 2), round(high, 2)], 'market_total': market_total,
            'recommendation': rec
        }
    
    def simulate_bets_chrono(self, df_preds: pd.DataFrame,
                             ev_min: float = 0.10,        # was 0.08 - stricter gate
                             diff_min: float = 1.10,      # was 1.0 - need clearer model edge
                             kelly_fraction: float = 0.15,# was 0.20 - more conservative Kelly
                             bankroll: float = 100.0,
                             max_bet_frac: float = 0.008, # was 0.01 - smaller max bet
                             max_day_risk_frac: float = 0.03  # cap 3% bankroll per day
                             ) -> pd.DataFrame:
        """
        Chronological bet sim that only uses *past* data to fit an isotonic
        calibrator mapping diff=(pred_total - market_total) -> P(Over).

        Safer gating + risk caps to avoid overbetting.
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        
        if df_preds is None or df_preds.empty:
            print("⚠️  No predictions to simulate bets.")
            return pd.DataFrame()

        df = df_preds.copy()
        # ensure chronology
        if not np.issubdtype(df['date'].dtype, np.datetime64):
            df['date'] = pd.to_datetime(df['date'])
        sort_cols = ['date']
        if 'game_id' in df.columns: sort_cols.append('game_id')
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # online isotonic + logistic on past-only data
        iso = IsotonicRegression(out_of_bounds='clip')
        logit = LogisticRegression(max_iter=1000, C=1.0)
        diffs_hist, outcomes_hist = [], []
        iso_fitted = False
        logit_fitted = False

        equity = bankroll
        rows = []

        current_day = None
        day_staked = 0.0
        day_pnl = 0.0
        day_bets = 0
        max_bets_per_day = 6
        day_stop_loss_frac = 0.02  # stop if -2% on the day

        def american_to_decimal(a):
            a = int(a)
            return 1.0 + (a/100.0 if a > 0 else 100.0/abs(a))

        def kelly_fraction_fn(p, american):
            b = american_to_decimal(american) - 1.0
            return max(0.0, (p*b - (1.0 - p)) / max(b, 1e-9))

        def ev_roi(p, american):
            dec = american_to_decimal(american)
            return p*(dec - 1.0) - (1.0 - p)

        # fallback prob mapping before we have enough history
        def fallback_prob(diff):
            # gentle slope; roughly ±2.5 runs ~ ±0.5 prob swing
            return float(np.clip(0.5 + diff / 5.0, 0.1, 0.9))

        # helper: best odds for this game/date (or -110)
        def best_odds(gid, d):
            try:
                q = text("""
                    SELECT book, total, over_odds, under_odds
                    FROM totals_odds
                    WHERE game_id = :gid AND date = :d
                """)
                tdf = pd.read_sql(q, self.engine, params={'gid': str(gid), 'd': pd.to_datetime(d).date()})
                if tdf.empty:
                    return -110, -110, 'N/A', 'N/A'
                over_row = tdf.loc[tdf['over_odds'].astype(int).idxmax()]
                under_row = tdf.loc[tdf['under_odds'].astype(int).idxmax()]
                return int(over_row['over_odds']), int(under_row['under_odds']), str(over_row['book']), str(under_row['book'])
            except Exception:
                return -110, -110, 'N/A', 'N/A'

        for _, r in df.iterrows():
            d = r['date'].date()
            if current_day != d:
                current_day = d
                day_staked = 0.0
                day_pnl = 0.0
                day_bets = 0

            pred_total = float(r['pred_total'])
            market_total = float(r.get('market_total', 9.0))
            actual_total = float(r.get('actual_total', np.nan))
            diff = pred_total - market_total

            # probability of OVER from ensemble, then shrink
            if iso_fitted:
                try:
                    p_iso = float(iso.predict([diff])[0])
                except Exception:
                    p_iso = fallback_prob(diff)
            else:
                p_iso = fallback_prob(diff)

            if logit_fitted:
                try:
                    p_log = float(logit.predict_proba([[diff]])[0, 1])
                except Exception:
                    p_log = fallback_prob(diff)
            else:
                p_log = fallback_prob(diff)

            p_raw = 0.5 * (p_iso + p_log)  # ensemble

            # shrink toward 0.5: more shrink early & when |diff| is small
            n_hist = len(diffs_hist)
            shrink_size = float(np.clip(1.0 - 1.0 / np.sqrt(max(n_hist, 1)), 0.50, 0.90))  # 0.5..0.9
            shrink_diff = float(np.clip(abs(diff) / 1.20, 0.0, 1.0))                       # 0..1 by 1.2 runs
            shrink_final = 0.5 * shrink_size + 0.5 * shrink_diff                           # blend

            p_over = 0.5 + (p_raw - 0.5) * shrink_final
            p_under = 1.0 - p_over

            # odds
            over_odds, under_odds, book_over, book_under = best_odds(r.get('game_id', ''), r['date'])

            # EVs
            ev_over  = ev_roi(p_over,  over_odds)
            ev_under = ev_roi(p_under, under_odds)

            # pick side
            if ev_over >= ev_under:
                side, use_odds, p_win, ev, book = 'OVER', over_odds, p_over, ev_over, book_over
            else:
                side, use_odds, p_win, ev, book = 'UNDER', under_odds, p_under, ev_under, book_under

            # volatility gating using model sigma
            sigma_indep = float(r.get('sigma_indep', np.nan))
            high_var = bool(sigma_indep > 2.8)
            very_high_var = bool(sigma_indep > 3.3)

            # dynamic thresholds
            diff_cut = diff_min + (0.30 if high_var else 0.0) + (0.20 if very_high_var else 0.0)
            ev_cut   = ev_min + (0.02 if high_var else 0.0)

            # odds quality (avoid heavy juice on totals)
            odds_ok = (use_odds >= -118)

            # huge uncertainty skip
            skip_uncertain = very_high_var and abs(diff) < 1.40

            # daily discipline: stop after bad day or too many bets
            daily_stop = (day_bets >= max_bets_per_day or day_pnl <= -day_stop_loss_frac * equity)

            place = (abs(diff) >= diff_cut) and (ev is not None and ev >= ev_cut) and odds_ok and not skip_uncertain and not daily_stop

            # stake caps
            stake_cap = max_bet_frac * equity
            stake = 0.0
            profit = np.nan
            won = np.nan

            if place and stake_cap > 0:
                f_star = kelly_fraction_fn(p_win, use_odds)
                # scale stake by confidence away from 0.5 (full at ~0.62)
                conf = abs(p_win - 0.5)
                stake_mult = float(np.clip(conf / 0.12, 0.0, 1.0))
                stake = min(stake_cap, kelly_fraction * equity * f_star) * stake_mult

                # daily risk cap
                if day_staked + stake > max_day_risk_frac * equity:
                    stake = max(0.0, max_day_risk_frac * equity - day_staked)

                stake = float(np.round(stake, 2))
                day_staked += stake

                if stake > 0 and np.isfinite(actual_total):
                    hit = (actual_total > market_total) if side == 'OVER' else (actual_total < market_total)
                    if hit:
                        profit = stake * (american_to_decimal(use_odds) - 1.0)
                        won = True
                    else:
                        profit = -stake
                        won = False
                    equity += profit

                    # update daily tracking
                    day_bets += 1
                    day_pnl += profit

            # record row
            rows.append({
                'date': d,
                'game_id': r.get('game_id'),
                'home_team': r.get('home_team'),
                'away_team': r.get('away_team'),
                'pred_total': pred_total,
                'market_total': market_total,
                'diff': diff,
                'side': side,
                'odds': use_odds,
                'book': book,
                'p_win': p_win,
                'ev': ev,
                'stake': stake,
                'result_total': actual_total,
                'won': won,
                'profit': profit if np.isfinite(actual_total) else np.nan,
                'bankroll_after': float(np.round(equity, 2)) if np.isfinite(actual_total) else np.nan,
            })

            # update past-only calibrator AFTER this game settles
            if np.isfinite(actual_total) and market_total > 0:
                diffs_hist.append(diff)
                outcomes_hist.append(1.0 if actual_total > market_total else 0.0)

                if len(diffs_hist) >= 300 and (not iso_fitted or len(diffs_hist) % 75 == 0):
                    try:
                        iso.fit(diffs_hist, outcomes_hist)
                        iso_fitted = True
                    except Exception:
                        iso_fitted = False
                    try:
                        X = np.array(diffs_hist, dtype=float).reshape(-1, 1)
                        y = np.array(outcomes_hist, dtype=int)
                        logit.fit(X, y)
                        logit_fitted = True
                    except Exception:
                        logit_fitted = False

        bets = pd.DataFrame(rows)
        settled = bets[bets['profit'].notna()]
        n_bets = (bets['stake'] > 0).sum()
        hit = float(settled['won'].mean()) if not settled.empty else np.nan
        roi_per_dollar = float(settled['profit'].sum() / max(1.0, settled['stake'].sum())) if not settled.empty else 0.0
        bankroll_roi = (equity - bankroll) / bankroll

        print(f"🎯 Chronological EV Betting Simulation:")
        print(f"EV Bets: {n_bets} | Hit: {hit:.1%} | ROI (per $ staked): {roi_per_dollar:+.2%} | "
              f"Bankroll: ${equity:,.2f} | Bankroll ROI: {bankroll_roi:+.2%}")

        return bets

    def save_state(self, filepath: str = 'incremental_ultra80_state.joblib') -> None:
        state = {
            'scaler': self.scaler,
            'home_model': self.home_model,
            'away_model': self.away_model,
            'home_var_model': self.home_var_model,
            'away_var_model': self.away_var_model,
            'feature_columns': self.feature_columns,
            'home_bias_mu': self.home_bias_mu,
            'away_bias_mu': self.away_bias_mu,
            'conformal_scores': self.conformal_scores,
            'conformal_bucket_scores': self.conformal_bucket_scores,
            'conformal_zscores': self.conformal_zscores,  # NEW: context-aware conformal
            'market_diffs': self.market_diffs,
            'market_outcomes': self.market_outcomes,
            'team_stats': self.team_stats,
            'matchup_history': self.matchup_history,
            'pitcher_stats': self.pitcher_stats,
            'bullpen_usage': self.bullpen_usage,
            'pitcher_handedness': self.pitcher_handedness,
            'umpire_factors': self.umpire_factors,
            'venue_factors': self.venue_factors,
            'is_fitted': self.is_fitted,
            'warmup_n': self.warmup_n,
            'prob_cal': self.prob_cal,
        }
        joblib.dump(state, filepath)
        print(f"💾 Saved state to {filepath}")

    def _reset_models_and_scaler(self):
        """Reset models/scaler if the schema changed"""
        # brand-new scaler & heads; keep your same hyperparams
        self.scaler = StandardScaler()
        self.home_model = SGDRegressor(loss='huber', epsilon=1.0,
                                       learning_rate='adaptive', eta0=0.01, alpha=1e-3,
                                       random_state=42, warm_start=True)
        self.away_model = SGDRegressor(loss='huber', epsilon=1.0,
                                       learning_rate='adaptive', eta0=0.01, alpha=1e-3,
                                       random_state=43, warm_start=True)
        self.home_var_model = SGDRegressor(loss='squared_error',
                                           learning_rate='adaptive', eta0=0.01, alpha=1e-4,
                                           random_state=44, warm_start=True)
        self.away_var_model = SGDRegressor(loss='squared_error',
                                           learning_rate='adaptive', eta0=0.01, alpha=1e-4,
                                           random_state=45, warm_start=True)
        self.is_fitted = False
        self.warmup_X, self.warmup_y_home, self.warmup_y_away = [], [], []

    def _clean_features(self, feats: Dict[str, float]) -> Dict[str, float]:
        """Clean feature dict by replacing NaN/inf with appropriate defaults"""
        cleaned = {}
        for k, v in feats.items():
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                # Use reasonable defaults based on feature type
                if 'era' in k.lower() or 'whip' in k.lower():
                    cleaned[k] = 4.5  # league average ERA/WHIP
                elif 'runs' in k.lower() or 'offense' in k.lower():
                    cleaned[k] = 4.5  # league average runs
                elif 'ratio' in k.lower() or 'momentum' in k.lower():
                    cleaned[k] = 1.0  # neutral ratio
                elif 'volatility' in k.lower() or 'cv' in k.lower():
                    cleaned[k] = 0.2  # moderate volatility
                else:
                    cleaned[k] = 0.0  # default to zero
            else:
                cleaned[k] = float(v)
        return cleaned

    def coverage_report(self, df):
        """Print detailed coverage report for dataset completeness"""
        import numpy as np, pandas as pd
        n = len(df)

        # row-level coverage
        print(f"📊 Coverage Report: {n:,} games loaded")
        
        # Check what recency columns actually exist
        cols = df.columns.tolist()
        recency_cols = [c for c in cols if any(x in c for x in ['_roll', '_l14', '_l7', '_l21'])]
        if recency_cols:
            print(f"📈 Found {len(recency_cols)} recency columns: {recency_cols[:5]}...")

        # simple presence flags for key joins (check if columns exist first)
        flags = {}
        
        # Pitcher rolling stats
        pitcher_cols = [c for c in cols if 'sp_era_roll' in c or 'sp_whip_roll' in c]
        if pitcher_cols:
            flags['pitcher_rolling'] = df[pitcher_cols].notna().any(axis=1)
            
        # Bullpen rolling stats  
        bp_cols = [c for c in cols if 'bp_era_roll' in c]
        if bp_cols:
            flags['bullpen_rolling'] = df[bp_cols].notna().any(axis=1)
            
        # Team offense rolling
        offense_cols = [c for c in cols if 'runs_pg' in c and ('l14' in c or 'l7' in c)]
        if offense_cols:
            flags['offense_rolling'] = df[offense_cols].notna().any(axis=1)
            
        # Park factors
        park_cols = [c for c in cols if 'park_pf' in c]
        if park_cols:
            flags['park_factors'] = df[park_cols].notna().any(axis=1)
            
        # Market totals
        market_cols = [c for c in cols if 'total' in c and any(x in c for x in ['opening', 'market'])]
        if market_cols:
            flags['market_totals'] = df[market_cols].notna().any(axis=1)
            
        for k, s in flags.items():
            have = float(s.mean())
            print(f"  {k:>20}: {have:6.1%} present | missing {int((1-have)*n):4d}")

        # feature-level missingness *before* your _clean_features() fills defaults
        miss = df.isna().mean().sort_values(ascending=False)
        print(f"\n📋 Top 10 columns by missing-rate:")
        top_missing = (miss.head(10) * 100).round(1).astype(str) + '%'
        for col, pct in top_missing.items():
            print(f"  {col:>25}: {pct}")
        print()
        
        # Coverage guardrail (fail fast if critical features <95%)
        critical_checks = {
            'park_pf_runs_3y': flags.get('park_factors', pd.Series([False])).mean(),
            'home_sp_hand': df['home_sp_hand'].notna().mean() if 'home_sp_hand' in df.columns else 0.0,
            'away_sp_hand': df['away_sp_hand'].notna().mean() if 'away_sp_hand' in df.columns else 0.0,
        }
        
        for col, rate in critical_checks.items():
            if rate < 0.95:
                print(f"⛔ {col} below 95% (got {rate:.1%}). Using conservative sigma until backfilled.")
                self.var_boost['pf'] = max(self.var_boost['pf'], 0.18)  # temporary widen

    def load_state(self, filepath: str = 'incremental_ultra80_state.joblib') -> bool:
        try:
            state = joblib.load(filepath)
            self.scaler = state['scaler']
            self.home_model = state['home_model']
            self.away_model = state['away_model']
            self.home_var_model = state['home_var_model']
            self.away_var_model = state['away_var_model']
            self.feature_columns = state.get('feature_columns', [])
            self.home_bias_mu = state.get('home_bias_mu', 0.0)
            self.away_bias_mu = state.get('away_bias_mu', 0.0)
            self.conformal_scores = state.get('conformal_scores', [])
            self.conformal_bucket_scores = state.get('conformal_bucket_scores', {'low': [], 'mid': [], 'high': []})
            self.conformal_zscores = state.get('conformal_zscores', {'low': [], 'mid': [], 'high': []})
            self.market_diffs = state.get('market_diffs', [])
            self.market_outcomes = state.get('market_outcomes', [])
            self.team_stats = state.get('team_stats', {})
            self.matchup_history = state.get('matchup_history', {})
            self.pitcher_stats = state.get('pitcher_stats', {})
            self.bullpen_usage = state.get('bullpen_usage', {})
            self.pitcher_handedness = state.get('pitcher_handedness', {})
            self.umpire_factors = state.get('umpire_factors', {})
            self.venue_factors = state.get('venue_factors', {})
            self.is_fitted = state.get('is_fitted', False)
            self.warmup_n = state.get('warmup_n', 200)
            self.prob_cal = state.get('prob_cal', CalibratorStack())
            print(f"📂 Loaded state from {filepath}")
            return True
        except Exception as e:
            print(f"⚠️  Could not load state: {e}")
            return False



def main():
    print('🧠 ULTRA 80% PREDICTION INTERVAL COVERAGE SYSTEM')
    system = IncrementalUltra80System()
    force_reset = os.getenv('FORCE_RESET', '0') == '1'
    state_loaded = False if force_reset else system.load_state()
    if state_loaded:
        print('🔄 Continuing from previously saved state...')
    elif force_reset:
        print('♻️ FORCE_RESET=1 → ignoring saved state and starting fresh.')

    # Optional training window via env (e.g., train a few months)
    start_date = os.getenv('START_DATE')  # e.g., '2025-05-01'
    end_date   = os.getenv('END_DATE')    # e.g., '2025-08-01'
    results = system.team_level_incremental_learn(start_date=start_date, end_date=end_date)

    if results:
        # Persist model state
        system.save_state()
        # Export backtest predictions
        df_preds = system.predictions_to_dataframe(results['predictions']) if 'predictions' in results else pd.DataFrame()
        if not df_preds.empty:
            system.export_predictions_to_files(df_preds, outdir='outputs', tag='backtest')
            # Optional DB persistence (uncomment when ready):
            # system.persist_predictions_to_db(df_preds, table='team_level_predictions')
        # Optional: produce a future slate if SLATE_DATE is set
        slate_date = os.getenv('SLATE_DATE')  # e.g., '2025-08-26'
        if slate_date and system.is_fitted and system.feature_columns:
            # Auto thresholds + one-pager for the slate
            thr = system.calibrate_thresholds_from_books()
            picks = system.recommend_slate_bets(slate_date, thresholds=thr)   # saves CSV with picks
            onepager_path = system.make_daily_onepager(slate_date, thresholds=thr, max_plays=3)
            
            if not picks.empty:
                print(f"\n💎 RECOMMENDED BETS for {slate_date}:")
                print("="*60)
                for _, r in picks.iterrows():
                    print(f"{r['away_team']} @ {r['home_team']} | {r['best_side']} {r['market_total']} ({r['best_odds']:+d})")
                    print(f"  Trust: {r['trust']:.2f} | EV: {r['ev']:+.1%} | Edge: {r['diff']:+.1f}")
                    print(f"  Book: {r['book']} | Sigma: {r['sigma_indep']:.2f}")
                    print()
                print("="*60)
            else:
                print(f"🔒 No high-confidence bets found for {slate_date}")
        # Show a few historic predictions/outcomes
        if not df_preds.empty:
            # Run EV-based betting simulation
            bets = system.simulate_bets(df_preds,
                                        ev_min=0.05,         # start stricter (≥ +5% EV)
                                        diff_min=0.8,        # need ≥ 0.8 run model edge
                                        kelly_fraction=0.25, # 1/4 Kelly
                                        bankroll=100.0,
                                        max_bet=200.0,
                                        outdir='outputs',
                                        tag='bets_backtest')
            
            # Head/tail
            print('🔎 Sample historic predictions (first 5):')
            print(df_preds.head(5).to_string(index=False))
            print('🔎 Sample historic predictions (last 5):')
            print(df_preds.tail(5).to_string(index=False))
            # Worst misses by absolute error
            df_preds['_abs_err'] = (df_preds['actual_total'] - df_preds['pred_total']).abs()
            worst = df_preds.sort_values('_abs_err', ascending=False).head(10)
            cols = ['date','home_team','away_team','pred_total','actual_total','lower_80','upper_80','market_total','edge','_abs_err']
            print('💥 Worst 10 abs errors:')
            print(worst[cols].to_string(index=False))
            # Daily rollup export
            df_daily = system.daily_rollup_from_predictions(df_preds)
            if not df_daily.empty:
                files_daily = system.export_predictions_to_files(df_daily, outdir='outputs', tag='daily_rollup')
                # Optional DB persistence (uncomment when ready):
                # system.persist_daily_rollup_to_db(df_daily, table='team_level_daily_rollup')
                print('📅 Recent daily rollup (last 10 days):')
                print(df_daily.tail(10).to_string(index=False))
        
        # Add comprehensive betting simulation with chronological integrity  
        print('\n')
        bet_results = system.simulate_bets_chrono(df_preds)
        
        # One example inference
        print('🔮 Example inference (mock):')
        fut = system.predict_future_game('New York Yankees', 'Boston Red Sox', '2025-08-26', 9.5)
        if fut:
            print(f"   Market: {fut['market_total']} | Pred: {fut['pred_total']} | 80% PI: {fut['interval_80']} | Rec: {fut['recommendation']}")




if __name__ == '__main__':
    main()
