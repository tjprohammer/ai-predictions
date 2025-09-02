#!/usr/bin/env python3
"""
Train MLB Hitting Props Calibration Models

This script trains Platt scaling calibration models for hitting props predictions.
The models learn to adjust raw prediction probabilities based on historical outcomes.

Usage:
    python train_hits_calibration.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--retrain-all]
    
Examples:
    python train_hits_calibration.py --start-date 2024-04-01 --end-date 2024-09-30
    python train_hits_calibration.py --retrain-all  # Use all available data
"""

import sys
import os
import argparse
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import joblib

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

class HittingPropsCalibrator:
    """Calibration models for hitting props predictions"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.calibrators = {}
        self.market_stats = {}
        
    def load_training_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load prediction results for training calibration models"""
        
        log.info(f"Loading training data from {start_date} to {end_date}")
        
        date_filter = ""
        params = {}
        
        if start_date and end_date:
            date_filter = "WHERE hp.date BETWEEN :start_date AND :end_date"
            params = {'start_date': start_date, 'end_date': end_date}
        elif start_date:
            date_filter = "WHERE hp.date >= :start_date"
            params = {'start_date': start_date}
        elif end_date:
            date_filter = "WHERE hp.date <= :end_date"
            params = {'end_date': end_date}
        
        query = text(f"""
            SELECT 
                hp.player_id,
                hp.date,
                hp.market,
                hp.line_value,
                hp.prob_over as predicted_prob,
                hp.confidence_score,
                
                -- Actual outcomes from game logs
                CASE hp.market
                    WHEN 'HITS_0.5' THEN CASE WHEN pgl.hits >= 1 THEN 1 ELSE 0 END
                    WHEN 'HITS_1.5' THEN CASE WHEN pgl.hits >= 2 THEN 1 ELSE 0 END
                    WHEN 'HR_0.5' THEN CASE WHEN pgl.home_runs >= 1 THEN 1 ELSE 0 END
                    WHEN 'RBI_0.5' THEN CASE WHEN pgl.rbi >= 1 THEN 1 ELSE 0 END
                    WHEN 'RBI_1.5' THEN CASE WHEN pgl.rbi >= 2 THEN 1 ELSE 0 END
                    WHEN 'TB_1.5' THEN CASE WHEN pgl.total_bases >= 2 THEN 1 ELSE 0 END
                    WHEN 'TB_2.5' THEN CASE WHEN pgl.total_bases >= 3 THEN 1 ELSE 0 END
                    WHEN 'TB_3.5' THEN CASE WHEN pgl.total_bases >= 4 THEN 1 ELSE 0 END
                    ELSE NULL
                END as actual_outcome,
                
                pgl.hits,
                pgl.home_runs,
                pgl.rbi,
                pgl.total_bases,
                pgl.plate_appearances
                
            FROM hitting_props hp
            JOIN player_game_logs pgl 
                ON hp.player_id = pgl.player_id 
                AND hp.date = pgl.date
            {date_filter}
            ORDER BY hp.date, hp.player_id, hp.market
        """)
        
        data = pd.read_sql(query, self.engine, params=params)
        
        # Remove rows with null outcomes
        initial_count = len(data)
        data = data.dropna(subset=['actual_outcome'])
        final_count = len(data)
        
        if initial_count > final_count:
            log.info(f"Removed {initial_count - final_count} rows with missing outcomes")
        
        log.info(f"Loaded {len(data)} prediction-outcome pairs")
        
        return data
    
    def analyze_calibration_data(self, data: pd.DataFrame):
        """Analyze calibration training data"""
        
        log.info("Analyzing calibration data...")
        
        # Market breakdown
        market_stats = data.groupby('market').agg({
            'predicted_prob': ['count', 'mean', 'std'],
            'actual_outcome': ['mean', 'std'],
            'confidence_score': 'mean'
        }).round(4)
        
        log.info("Market statistics:")
        for market in market_stats.index:
            count = market_stats.loc[market, ('predicted_prob', 'count')]
            pred_rate = market_stats.loc[market, ('predicted_prob', 'mean')]
            actual_rate = market_stats.loc[market, ('actual_outcome', 'mean')]
            confidence = market_stats.loc[market, ('confidence_score', 'mean')]
            
            log.info(f"  {market}: {count} samples, "
                    f"pred={pred_rate:.3f}, actual={actual_rate:.3f}, "
                    f"conf={confidence:.3f}")
        
        # Calibration analysis by probability bins
        for market in data['market'].unique():
            market_data = data[data['market'] == market].copy()
            
            if len(market_data) < 50:
                log.warning(f"Insufficient data for {market}: {len(market_data)} samples")
                continue
            
            # Create probability bins
            market_data['prob_bin'] = pd.cut(
                market_data['predicted_prob'], 
                bins=10, 
                labels=[f'0.{i}-0.{i+1}' for i in range(10)]
            )
            
            bin_stats = market_data.groupby('prob_bin').agg({
                'predicted_prob': ['count', 'mean'],
                'actual_outcome': 'mean'
            }).round(3)
            
            log.info(f"\n{market} calibration by probability bin:")
            for bin_name in bin_stats.index:
                if pd.isna(bin_name):
                    continue
                count = bin_stats.loc[bin_name, ('predicted_prob', 'count')]
                pred = bin_stats.loc[bin_name, ('predicted_prob', 'mean')]
                actual = bin_stats.loc[bin_name, ('actual_outcome', 'mean')]
                log.info(f"  {bin_name}: {count} samples, pred={pred:.3f}, actual={actual:.3f}")
    
    def train_market_calibrator(self, data: pd.DataFrame, market: str):
        """Train calibration model for a specific market"""
        
        market_data = data[data['market'] == market].copy()
        
        if len(market_data) < 100:
            log.warning(f"Insufficient data for {market}: {len(market_data)} samples")
            return None, None
        
        log.info(f"Training calibrator for {market} ({len(market_data)} samples)")
        
        X = market_data[['predicted_prob', 'confidence_score']].values
        y = market_data['actual_outcome'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Base classifier for calibration
        base_clf = LogisticRegression(random_state=42)
        
        # Fit calibrated classifier
        calibrator = CalibratedClassifierCV(
            base_clf, 
            method='sigmoid',  # Platt scaling
            cv=3
        )
        
        calibrator.fit(X_train, y_train)
        
        # Evaluate calibration
        y_pred_proba = calibrator.predict_proba(X_test)[:, 1]
        
        metrics = {
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'samples': len(market_data)
        }
        
        log.info(f"  {market} metrics: "
                f"Brier={metrics['brier_score']:.4f}, "
                f"LogLoss={metrics['log_loss']:.4f}, "
                f"AUC={metrics['auc']:.4f}")
        
        return calibrator, metrics
    
    def train_all_calibrators(self, data: pd.DataFrame):
        """Train calibration models for all markets"""
        
        log.info("Training calibration models for all markets...")
        
        markets = data['market'].unique()
        
        for market in markets:
            calibrator, metrics = self.train_market_calibrator(data, market)
            
            if calibrator is not None:
                self.calibrators[market] = calibrator
                self.market_stats[market] = metrics
        
        log.info(f"Trained calibrators for {len(self.calibrators)} markets")
    
    def save_calibrators(self, model_dir: str = 'models_hitting_calibration'):
        """Save trained calibration models"""
        
        model_path = Path(model_dir)
        model_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for market, calibrator in self.calibrators.items():
            filename = model_path / f'{market.lower()}_calibrator_{timestamp}.joblib'
            joblib.dump(calibrator, filename)
            log.info(f"Saved {market} calibrator to {filename}")
        
        # Save market statistics
        stats_file = model_path / f'calibration_stats_{timestamp}.json'
        import json
        with open(stats_file, 'w') as f:
            json.dump(self.market_stats, f, indent=2)
        
        log.info(f"Saved calibration statistics to {stats_file}")
        
        # Save latest symlinks
        for market, calibrator in self.calibrators.items():
            latest_link = model_path / f'{market.lower()}_calibrator_latest.joblib'
            if latest_link.exists():
                latest_link.unlink()
            
            source_file = model_path / f'{market.lower()}_calibrator_{timestamp}.joblib'
            latest_link.symlink_to(source_file.name)
        
        log.info("Created latest symlinks for all calibrators")
    
    def evaluate_existing_predictions(self, data: pd.DataFrame):
        """Evaluate uncalibrated vs calibrated predictions"""
        
        log.info("Evaluating calibration improvement...")
        
        results = []
        
        for market in data['market'].unique():
            market_data = data[data['market'] == market].copy()
            
            if len(market_data) < 50:
                continue
            
            X = market_data[['predicted_prob', 'confidence_score']].values
            y = market_data['actual_outcome'].values
            
            # Uncalibrated metrics
            uncal_brier = brier_score_loss(y, market_data['predicted_prob'])
            uncal_logloss = log_loss(y, market_data['predicted_prob'])
            
            # Calibrated metrics (if calibrator exists)
            if market in self.calibrators:
                cal_probs = self.calibrators[market].predict_proba(X)[:, 1]
                cal_brier = brier_score_loss(y, cal_probs)
                cal_logloss = log_loss(y, cal_probs)
                
                results.append({
                    'market': market,
                    'samples': len(market_data),
                    'uncalibrated_brier': uncal_brier,
                    'calibrated_brier': cal_brier,
                    'brier_improvement': uncal_brier - cal_brier,
                    'uncalibrated_logloss': uncal_logloss,
                    'calibrated_logloss': cal_logloss,
                    'logloss_improvement': uncal_logloss - cal_logloss
                })
        
        if results:
            results_df = pd.DataFrame(results)
            log.info("\nCalibration improvement summary:")
            log.info(results_df.round(4).to_string())
        
        return results

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Train hitting props calibration models')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--retrain-all', action='store_true', 
                       help='Use all available data for training')
    parser.add_argument('--model-dir', type=str, default='models_hitting_calibration',
                       help='Directory to save calibration models')
    
    args = parser.parse_args()
    
    database_url = "postgresql://mlbuser:mlbpass@localhost/mlb"
    
    try:
        calibrator = HittingPropsCalibrator(database_url)
        
        # Determine date range
        start_date = args.start_date
        end_date = args.end_date
        
        if args.retrain_all:
            start_date = None
            end_date = None
            log.info("Training with all available data")
        elif not start_date and not end_date:
            # Default to previous season
            current_year = datetime.now().year
            start_date = f"{current_year - 1}-04-01"
            end_date = f"{current_year - 1}-10-31"
            log.info(f"Using default date range: {start_date} to {end_date}")
        
        # Load training data
        data = calibrator.load_training_data(start_date, end_date)
        
        if data.empty:
            log.error("No training data found")
            sys.exit(1)
        
        # Analyze data
        calibrator.analyze_calibration_data(data)
        
        # Train calibrators
        calibrator.train_all_calibrators(data)
        
        if not calibrator.calibrators:
            log.error("No calibrators were trained successfully")
            sys.exit(1)
        
        # Evaluate improvements
        calibrator.evaluate_existing_predictions(data)
        
        # Save models
        calibrator.save_calibrators(args.model_dir)
        
        log.info("Calibration training completed successfully")
        
    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
