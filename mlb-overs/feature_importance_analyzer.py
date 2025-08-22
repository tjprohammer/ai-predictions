#!/usr/bin/env python3
"""
Feature Importance & Selection System
=====================================
Analyzes feature importance and selects the most predictive features
to maximize over/under picking accuracy.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE, mutual_info_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis for maximizing picking accuracy
    """
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or 'postgresql://mlbuser:mlbpass@localhost/mlb'
        self.engine = create_engine(self.db_url)
        self.models_dir = Path(__file__).parent / "models"
        self.analysis_dir = Path(__file__).parent / "feature_analysis"
        self.analysis_dir.mkdir(exist_ok=True)
        
        # Feature categories for analysis
        self.feature_categories = {
            'pitching': ['era', 'whip', 'k_rate', 'bb_rate', 'pitcher', 'sp_', 'bullpen'],
            'hitting': ['avg', 'obp', 'slg', 'ops', 'woba', 'wrc', 'iso'],
            'environmental': ['temp', 'wind', 'humidity', 'pressure', 'weather'],
            'market': ['total', 'odds', 'line', 'market'],
            'situational': ['venue', 'team', 'home', 'away', 'umpire'],
            'advanced': ['xwoba', 'xera', 'fip', 'xfip', 'war', 'wpa']
        }
    
    def get_comprehensive_data(self, days_back: int = 90) -> pd.DataFrame:
        """Get comprehensive data for feature analysis"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            query = text("""
                SELECT * FROM enhanced_games 
                WHERE total_runs IS NOT NULL 
                AND date >= :start_date AND date <= :end_date
                AND market_total IS NOT NULL
                ORDER BY date DESC
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'start_date': start_date, 'end_date': end_date})
            
            logger.info(f"📊 Retrieved {len(df)} games with {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting data: {e}")
            return pd.DataFrame()
    
    def create_picking_accuracy_target(self, df: pd.DataFrame) -> pd.Series:
        """Create binary target for picking accuracy (over/under relative to market)"""
        # Create binary target: 1 if actual > market (over), 0 if actual <= market (under)
        picking_target = (df['total_runs'] > df['market_total']).astype(int)
        logger.info(f"🎯 Created picking target: {picking_target.mean():.1%} games went over")
        return picking_target
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for analysis"""
        # Exclude non-feature columns
        exclude_cols = ['game_id', 'date', 'total_runs', 'market_total', 'home_team', 'away_team']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        # Handle datetime columns first
        datetime_cols = []
        for col in X.columns:
            if X[col].dtype == 'datetime64[ns]' or 'timestamp' in col.lower() or 'created' in col.lower():
                datetime_cols.append(col)
        
        # Remove datetime columns that can't be easily converted
        X = X.drop(columns=datetime_cols)
        
        # Handle categorical features
        from sklearn.preprocessing import LabelEncoder
        le_dict = {}
        
        for col in X.columns:
            if X[col].dtype == 'object':
                # Skip if it looks like a complex object that can't be encoded
                try:
                    le_dict[col] = LabelEncoder()
                    X[col] = le_dict[col].fit_transform(X[col].fillna('Unknown'))
                except Exception as e:
                    logger.warning(f"Dropping problematic column {col}: {e}")
                    X = X.drop(columns=[col])
            elif X[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                X[col] = X[col].fillna(X[col].median())
            else:
                # Drop any other problematic columns
                logger.warning(f"Dropping column with unsupported dtype {col}: {X[col].dtype}")
                X = X.drop(columns=[col])
        
        logger.info(f"🔧 Preprocessed {len(X.columns)} features (removed {len(datetime_cols)} datetime columns)")
        return X, le_dict
    
    def analyze_correlation_with_picking_success(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze correlation between features and picking success"""
        logger.info("📈 Analyzing feature correlations with picking success...")
        
        correlations = {}
        
        for col in X.columns:
            try:
                # Pearson correlation
                pearson_corr, pearson_p = pearsonr(X[col], y)
                
                # Spearman correlation (for non-linear relationships)
                spearman_corr, spearman_p = spearmanr(X[col], y)
                
                correlations[col] = {
                    'pearson_corr': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_p_value': spearman_p,
                    'abs_pearson': abs(pearson_corr),
                    'abs_spearman': abs(spearman_corr)
                }
                
            except Exception as e:
                logger.warning(f"Could not calculate correlation for {col}: {e}")
                correlations[col] = {
                    'pearson_corr': 0, 'pearson_p_value': 1,
                    'spearman_corr': 0, 'spearman_p_value': 1,
                    'abs_pearson': 0, 'abs_spearman': 0
                }
        
        # Sort by absolute correlation
        sorted_correlations = sorted(correlations.items(), 
                                   key=lambda x: x[1]['abs_pearson'], reverse=True)
        
        logger.info(f"✅ Correlation analysis complete")
        return dict(sorted_correlations)
    
    def random_forest_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Use Random Forest to determine feature importance"""
        logger.info("🌲 Analyzing Random Forest feature importance...")
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Create importance dictionary
        feature_importance = {}
        for i, col in enumerate(X.columns):
            feature_importance[col] = {
                'importance': importances[i],
                'rank': 0  # Will be filled later
            }
        
        # Sort by importance and assign ranks
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1]['importance'], reverse=True)
        
        for rank, (feature, data) in enumerate(sorted_features, 1):
            feature_importance[feature]['rank'] = rank
        
        logger.info(f"✅ Random Forest importance analysis complete")
        return feature_importance
    
    def mutual_information_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze mutual information between features and target"""
        logger.info("🔍 Analyzing mutual information...")
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Create MI dictionary
        mi_importance = {}
        for i, col in enumerate(X.columns):
            mi_importance[col] = {
                'mutual_info': mi_scores[i],
                'rank': 0
            }
        
        # Sort and rank
        sorted_mi = sorted(mi_importance.items(), 
                          key=lambda x: x[1]['mutual_info'], reverse=True)
        
        for rank, (feature, data) in enumerate(sorted_mi, 1):
            mi_importance[feature]['rank'] = rank
        
        logger.info(f"✅ Mutual information analysis complete")
        return mi_importance
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                    target_features: int = 50) -> Dict:
        """Use RFE to select most important features"""
        logger.info(f"🔄 Running Recursive Feature Elimination (target: {target_features} features)...")
        
        # Use Random Forest as the estimator
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Run RFE
        rfe = RFE(estimator, n_features_to_select=target_features, step=1)
        rfe.fit(X, y)
        
        # Get results
        rfe_results = {}
        for i, col in enumerate(X.columns):
            rfe_results[col] = {
                'selected': rfe.support_[i],
                'ranking': rfe.ranking_[i]
            }
        
        logger.info(f"✅ RFE complete: {sum(rfe.support_)} features selected")
        return rfe_results
    
    def analyze_feature_categories(self, correlations: Dict, rf_importance: Dict) -> Dict:
        """Analyze performance by feature category"""
        logger.info("📊 Analyzing feature categories...")
        
        category_analysis = {}
        
        for category, keywords in self.feature_categories.items():
            category_features = []
            
            # Find features matching this category
            for feature in correlations.keys():
                if any(keyword.lower() in feature.lower() for keyword in keywords):
                    category_features.append(feature)
            
            if category_features:
                # Calculate category statistics
                avg_correlation = np.mean([abs(correlations[f]['pearson_corr']) 
                                         for f in category_features])
                avg_rf_importance = np.mean([rf_importance[f]['importance'] 
                                           for f in category_features])
                
                category_analysis[category] = {
                    'feature_count': len(category_features),
                    'avg_correlation': avg_correlation,
                    'avg_rf_importance': avg_rf_importance,
                    'top_features': sorted(category_features, 
                                         key=lambda x: abs(correlations[x]['pearson_corr']), 
                                         reverse=True)[:5]
                }
        
        logger.info(f"✅ Category analysis complete")
        return category_analysis
    
    def create_optimized_feature_set(self, correlations: Dict, rf_importance: Dict, 
                                   mi_importance: Dict, rfe_results: Dict) -> List[str]:
        """Create optimized feature set combining all analysis methods"""
        logger.info("🎯 Creating optimized feature set...")
        
        # Scoring system for features
        feature_scores = {}
        
        for feature in correlations.keys():
            score = 0
            
            # Correlation score (0-25 points)
            corr_rank = len([f for f in correlations.keys() 
                           if abs(correlations[f]['pearson_corr']) > abs(correlations[feature]['pearson_corr'])])
            score += max(0, 25 - (corr_rank / 10))
            
            # Random Forest importance score (0-25 points)
            rf_rank = rf_importance[feature]['rank']
            score += max(0, 25 - (rf_rank / 10))
            
            # Mutual Information score (0-25 points)
            mi_rank = mi_importance[feature]['rank']
            score += max(0, 25 - (mi_rank / 10))
            
            # RFE selection bonus (0-25 points)
            if rfe_results[feature]['selected']:
                score += 25
            else:
                score += max(0, 25 - rfe_results[feature]['ranking'])
            
            feature_scores[feature] = score
        
        # Select top features
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 75 features for optimal performance
        optimized_features = [feature for feature, score in sorted_features[:75]]
        
        logger.info(f"✅ Optimized feature set created: {len(optimized_features)} features")
        return optimized_features
    
    def validate_feature_set(self, X: pd.DataFrame, y: pd.Series, 
                           selected_features: List[str]) -> Dict:
        """Validate the selected feature set"""
        logger.info("✅ Validating optimized feature set...")
        
        # Test with full feature set
        rf_full = RandomForestRegressor(n_estimators=200, random_state=42)
        full_scores = cross_val_score(rf_full, X, y, cv=5, scoring='neg_mean_absolute_error')
        
        # Test with selected features
        X_selected = X[selected_features]
        rf_selected = RandomForestRegressor(n_estimators=200, random_state=42)
        selected_scores = cross_val_score(rf_selected, X_selected, y, cv=5, scoring='neg_mean_absolute_error')
        
        validation_results = {
            'full_feature_mae': -full_scores.mean(),
            'selected_feature_mae': -selected_scores.mean(),
            'improvement': -full_scores.mean() - (-selected_scores.mean()),
            'feature_reduction': f"{len(X.columns)} -> {len(selected_features)}",
            'reduction_percentage': ((len(X.columns) - len(selected_features)) / len(X.columns)) * 100
        }
        
        logger.info(f"🎯 Validation complete:")
        logger.info(f"   Full features MAE: {validation_results['full_feature_mae']:.3f}")
        logger.info(f"   Selected features MAE: {validation_results['selected_feature_mae']:.3f}")
        logger.info(f"   Features reduced by: {validation_results['reduction_percentage']:.1f}%")
        
        return validation_results
    
    def run_complete_analysis(self) -> Dict:
        """Run complete feature importance analysis"""
        logger.info("🚀 Starting complete feature analysis...")
        
        # Get data
        df = self.get_comprehensive_data()
        if df.empty:
            logger.error("No data available for analysis")
            return {}
        
        # Create picking accuracy target
        y = self.create_picking_accuracy_target(df)
        
        # Preprocess features
        X, label_encoders = self.preprocess_features(df)
        
        # Run all analyses
        correlations = self.analyze_correlation_with_picking_success(X, y)
        rf_importance = self.random_forest_feature_importance(X, y)
        mi_importance = self.mutual_information_analysis(X, y)
        rfe_results = self.recursive_feature_elimination(X, y)
        
        # Analyze by categories
        category_analysis = self.analyze_feature_categories(correlations, rf_importance)
        
        # Create optimized feature set
        optimized_features = self.create_optimized_feature_set(
            correlations, rf_importance, mi_importance, rfe_results
        )
        
        # Validate feature set
        validation = self.validate_feature_set(X, y, optimized_features)
        
        # Compile complete results
        complete_analysis = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_games': len(df),
                'total_features': len(X.columns),
                'over_rate': float(y.mean())
            },
            'top_features_by_correlation': list(correlations.keys())[:20],
            'top_features_by_rf_importance': [f for f, data in sorted(rf_importance.items(), 
                                            key=lambda x: x[1]['importance'], reverse=True)][:20],
            'category_analysis': category_analysis,
            'optimized_feature_set': optimized_features,
            'validation_results': validation,
            'detailed_analysis': {
                'correlations': correlations,
                'rf_importance': rf_importance,
                'mutual_information': mi_importance,
                'rfe_results': rfe_results
            }
        }
        
        # Save results
        results_path = self.analysis_dir / f"feature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(complete_analysis, f, indent=2, default=str)
        
        # Save optimized feature list
        features_path = self.models_dir / "optimized_features.json"
        with open(features_path, 'w') as f:
            json.dump(optimized_features, f, indent=2)
        
        logger.info(f"📊 Complete analysis saved to: {results_path}")
        logger.info(f"🎯 Optimized features saved to: {features_path}")
        
        return complete_analysis

def main():
    """Main execution function"""
    analyzer = FeatureImportanceAnalyzer()
    
    print("🔍 Feature Importance & Selection Analysis")
    print("=" * 50)
    print("Goal: Identify the most predictive features for over/under accuracy")
    print()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print("\n🎉 Feature Analysis Complete!")
    if results:
        print(f"📊 Analysis Summary:")
        print(f"   Total Games Analyzed: {results['data_summary']['total_games']}")
        print(f"   Original Features: {results['data_summary']['total_features']}")
        print(f"   Optimized Features: {len(results['optimized_feature_set'])}")
        print(f"   Validation MAE: {results['validation_results']['selected_feature_mae']:.3f}")
        
        print(f"\n🏆 Top 10 Most Predictive Features:")
        for i, feature in enumerate(results['top_features_by_correlation'][:10], 1):
            print(f"   {i:2d}. {feature}")
    
    return results

if __name__ == "__main__":
    main()
