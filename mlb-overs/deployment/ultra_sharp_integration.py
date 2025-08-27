#!/usr/bin/env python3
"""
Ultra Sharp Pipeline Integration for Daily Workflow
=================================================
Integrates the V15 Ultra Sharp Pipeline (smart calibration + ROI filtering) 
into the daily API workflow system.

This module provides a drop-in replacement for the enhanced predictor
that uses the production-ready ultra sharp pipeline with:
- Smart calibration (blend/raw/iso selection)
- ROI-first filtering with z-edge analysis
- Stable categorical encoders
- Per-game uncertainty estimation
"""

import os
import sys
import logging
import subprocess
import time
import pathlib
import shlex
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Add the pipelines directory to path for ultra_sharp_pipeline import
sys.path.append(str(Path(__file__).parent.parent / "pipelines"))

log = logging.getLogger(__name__)

def run_ultra_sharp(cmd, expected_path):
    """
    Robust Ultra Sharp subprocess runner with proper error handling
    
    Args:
        cmd: Command to execute
        expected_path: Path where output file should be created
        
    Returns:
        Tuple of (output_path, execution_time)
        
    Raises:
        RuntimeError: If subprocess fails
        FileNotFoundError: If expected output file not created
    """
    t0 = time.time()
    
    # Use shlex.split for proper command parsing on Windows
    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = cmd
    
    proc = subprocess.run(
        cmd_list, 
        capture_output=True, 
        text=True,
        timeout=300
    )
    
    execution_time = time.time() - t0
    
    if proc.returncode != 0:
        raise RuntimeError(
            f"Ultra Sharp non-zero exit ({proc.returncode}).\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    
    expected_path_obj = pathlib.Path(expected_path)
    if not expected_path_obj.exists():
        raise FileNotFoundError(
            f"Ultra Sharp output not found at {expected_path}.\n"
            f"STDOUT:\n{proc.stdout}\n" 
            f"STDERR:\n{proc.stderr}"
        )
    
    log.info(f"✅ Ultra Sharp completed in {execution_time:.1f}s")
    if proc.stdout:
        log.info(f"Ultra Sharp stdout: {proc.stdout.strip()}")
    
    return str(expected_path), execution_time

class UltraSharpIntegrator:
    """Integrates Ultra Sharp Pipeline into daily workflow"""
    
    def __init__(self, 
                 model_dir: str = "../../models_ultra_v15_smart_calib",
                 db_url: Optional[str] = None):
        """
        Initialize Ultra Sharp integrator
        
        Args:
            model_dir: Path to V15 model directory (relative to pipelines/)
            db_url: Database URL (defaults to env DATABASE_URL)
        """
        self.model_dir = model_dir
        self.db_url = db_url or os.getenv("DATABASE_URL", 
            "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
        
        # Path to ultra sharp pipeline script
        self.pipeline_dir = Path(__file__).parent.parent / "pipelines"
        self.pipeline_script = self.pipeline_dir / "ultra_sharp_pipeline.py"
        
        if not self.pipeline_script.exists():
            raise FileNotFoundError(f"Ultra sharp pipeline not found: {self.pipeline_script}")
        
        log.info(f"Ultra Sharp integrator initialized: model_dir={model_dir}")
    
    def predict_today_games(self, target_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate predictions for today's games using Ultra Sharp Pipeline
        
        Args:
            target_date: Date string in YYYY-MM-DD format
            
        Returns:
            (predictions_df, featured_df, X_df) - Compatible with enhanced predictor interface
        """
        try:
            # Generate temporary output file
            timestamp = datetime.now().strftime("%H%M%S")
            temp_output = self.pipeline_dir.parent.parent / "exports" / f"ultra_sharp_daily_{target_date}_{timestamp}.csv"
            temp_output.parent.mkdir(exist_ok=True, parents=True)
            
            # Run ultra sharp pipeline pricing with robust error handling
            cmd = [
                sys.executable,
                str(self.pipeline_script),
                "price",
                "--db", self.db_url,
                "--date", target_date,
                "--model_dir", self.model_dir,
                "--out", str(temp_output)
            ]
            
            log.info(f"Running Ultra Sharp Pipeline: {' '.join(cmd)}")
            
            # Execute pipeline with robust wrapper
            try:
                output_path, exec_time = run_ultra_sharp(cmd, temp_output)
                log.info(f"✅ Ultra Sharp completed successfully in {exec_time:.1f}s")
            except (RuntimeError, FileNotFoundError) as e:
                log.error(f"Ultra Sharp prediction failed: {e}")
                raise e
            
            # Load predictions
            predictions = pd.read_csv(temp_output)
            log.info(f"Loaded {len(predictions)} predictions from Ultra Sharp Pipeline")
            
            # Clean up temp file
            try:
                temp_output.unlink()
            except:
                pass
            
            # Extract high-confidence plays for logging
            high_conf = predictions[predictions.get('high_confidence', 0) == 1]
            if len(high_conf) > 0:
                log.info(f"🔥 High-confidence plays: {len(high_conf)}/{len(predictions)} games")
                avg_ev = high_conf.get('ev_110', pd.Series([0])).mean()
                avg_z = high_conf.get('z_edge', pd.Series([0])).mean()
                log.info(f"🔥 High-conf metrics: avg_z={avg_z:.2f}, avg_ev={avg_ev:.3f}")
            else:
                log.info("⚠️ No high-confidence predictions for today")
            
            # Create compatible interface - build dummy featured and X DataFrames
            # Since ultra sharp handles its own feature engineering internally
            featured_df = self._create_featured_df(predictions, target_date)
            X_df = self._create_X_df(predictions)
            
            return predictions, featured_df, X_df
            
        except Exception as e:
            log.error(f"Ultra Sharp prediction failed: {e}")
            # Return None to trigger fallback in daily workflow
            return None, None, None
    
    def _create_featured_df(self, predictions: pd.DataFrame, target_date: str) -> pd.DataFrame:
        """Create a dummy featured DataFrame for workflow compatibility"""
        try:
            # Load raw game data to create featured-like structure
            engine = create_engine(self.db_url)
            
            with engine.connect() as conn:
                # Get basic game info to create featured-like DataFrame
                sql = """
                    SELECT 
                        eg.game_id, eg.date, eg.home_team, eg.away_team,
                        eg.market_total, eg.predicted_total_ultra,
                        gc.temperature, gc.wind_speed, gc.humidity,
                        gc.ballpark_run_factor, gc.ballpark_hr_factor
                    FROM enhanced_games eg
                    LEFT JOIN game_conditions gc ON eg.game_id::text = gc.game_id::text
                    WHERE eg.date = %(date)s
                    AND eg.market_total IS NOT NULL
                    AND eg.total_runs IS NULL
                """
                
                featured = pd.read_sql(sql, conn, params={"date": target_date})
                
                # Add computed features that workflow expects
                if len(featured) > 0:
                    featured['ballpark_run_factor'] = featured.get('ballpark_run_factor', 1.0)
                    featured['ballpark_hr_factor'] = featured.get('ballpark_hr_factor', 1.0)
                    featured['temperature'] = featured.get('temperature', 70.0)
                    featured['wind_speed'] = featured.get('wind_speed', 5.0)
                    featured['expected_total'] = featured.get('market_total', 9.0)
                    
                    # Add dummy pitcher stats for variance validation
                    for team in ['home', 'away']:
                        for stat in ['sp_era', 'sp_whip', 'sp_k_per_9', 'sp_bb_per_9']:
                            col = f"{team}_{stat}"
                            if col not in featured.columns:
                                featured[col] = 3.50 if 'era' in stat else (1.25 if 'whip' in stat else 8.0)
                    
                    log.info(f"Created featured DataFrame: {featured.shape}")
                else:
                    log.warning("No game data found for featured DataFrame")
                
                return featured
                
        except Exception as e:
            log.warning(f"Failed to create featured DataFrame: {e}")
            # Return minimal DataFrame
            return pd.DataFrame({
                'game_id': predictions.get('game_id', []),
                'date': predictions.get('date', []),
                'ballpark_run_factor': [1.0] * len(predictions),
                'temperature': [70.0] * len(predictions),
                'expected_total': predictions.get('predicted_total', [9.0] * len(predictions))
            })
    
    def _create_X_df(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Create a dummy X DataFrame for workflow compatibility"""
        # Create minimal feature matrix - ultra sharp handles features internally
        n_games = len(predictions)
        
        return pd.DataFrame({
            'market_total': predictions.get('predicted_total', [9.0] * n_games) - predictions.get('resid_hat', [0.0] * n_games),
            'temperature': [70.0] * n_games,
            'ballpark_factor': [1.0] * n_games,
            'wind_factor': [1.0] * n_games
        })


def integrate_ultra_sharp_predictor(target_date: str, 
                                   model_dir: str = "../../models_ultra_v15_smart_calib",
                                   db_url: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to run Ultra Sharp Pipeline prediction
    
    Args:
        target_date: Date string in YYYY-MM-DD format
        model_dir: Path to V15 model directory
        db_url: Database URL
        
    Returns:
        (predictions_df, featured_df, X_df) - Compatible with enhanced predictor interface
    """
    integrator = UltraSharpIntegrator(model_dir=model_dir, db_url=db_url)
    return integrator.predict_today_games(target_date)


def check_ultra_sharp_available() -> bool:
    """Check if Ultra Sharp Pipeline is available and functional"""
    try:
        pipeline_dir = Path(__file__).parent.parent / "pipelines"
        pipeline_script = pipeline_dir / "ultra_sharp_pipeline.py"
        
        if not pipeline_script.exists():
            return False
            
        # Try importing the module
        sys.path.append(str(pipeline_dir))
        import ultra_sharp_pipeline
        
        return True
        
    except Exception as e:
        log.debug(f"Ultra Sharp Pipeline not available: {e}")
        return False


if __name__ == "__main__":
    # Test the integration
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Ultra Sharp Integration")
    parser.add_argument("--date", required=True, help="Date to predict (YYYY-MM-DD)")
    parser.add_argument("--model_dir", default="../../models_ultra_v15_smart_calib", 
                       help="Model directory")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    predictions, featured, X = integrate_ultra_sharp_predictor(args.date, args.model_dir)
    
    if predictions is not None:
        print(f"✅ Ultra Sharp prediction successful: {len(predictions)} games")
        if 'high_confidence' in predictions.columns:
            hc_count = predictions['high_confidence'].sum()
            print(f"🔥 High-confidence games: {hc_count}")
    else:
        print("❌ Ultra Sharp prediction failed")
