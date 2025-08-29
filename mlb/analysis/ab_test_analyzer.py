#!/usr/bin/env python3
"""
A/B Test Results Analyzer
========================

Quick analysis tool to interpret comprehensive A/B testing results and provide actionable insights.

Usage:
    python ab_test_analyzer.py --results-file comprehensive_ab_test_results_20250829_120000.json
    python ab_test_analyzer.py --latest  # Analyze most recent results
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

class ABTestAnalyzer:
    """Analyzer for comprehensive A/B testing results"""
    
    def __init__(self, results_file: str = None):
        self.results_dir = Path("../../data/ab_test_results")
        
        if results_file:
            self.results_file = Path(results_file)
        else:
            # Find most recent results file
            result_files = list(self.results_dir.glob("comprehensive_ab_test_results_*.json"))
            if not result_files:
                raise FileNotFoundError("No A/B test results found")
            self.results_file = max(result_files, key=os.path.getctime)
        
        # Load results
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
            
    def print_executive_summary(self):
        """Print executive summary of A/B testing results"""
        print("\n" + "="*80)
        print("🎯 ULTRA-80 A/B TESTING EXECUTIVE SUMMARY")
        print("="*80)
        
        test_config = self.results.get('test_config', {})
        print(f"📊 Test Period: {test_config.get('start_date')} to {test_config.get('end_date')}")
        print(f"🎮 Games Analyzed: {test_config.get('total_games', 'N/A')}")
        print(f"⏰ Test Completed: {test_config.get('test_timestamp', 'N/A')}")
        
        if 'recommendations' in self.results:
            print(f"\n🏆 KEY FINDINGS:")
            
            recs = self.results['recommendations']
            
            for test_type, best_config in recs.get('best_configurations', {}).items():
                config_name = best_config['config']
                mae = best_config['mae']
                print(f"   • {test_type.replace('_', ' ').title()}: {config_name} (MAE: {mae:.3f})")
                
            print(f"\n💡 OPTIMIZATION INSIGHTS:")
            for insight in recs.get('optimization_insights', []):
                print(f"   • {insight}")
                
    def analyze_learning_windows(self):
        """Detailed analysis of learning window results"""
        if 'learning_windows' not in self.results:
            return
            
        print(f"\n📈 LEARNING WINDOW DETAILED ANALYSIS")
        print("-" * 50)
        
        windows_data = []
        for window, result in self.results['learning_windows'].items():
            if isinstance(result, dict) and 'mae' in result:
                windows_data.append({
                    'window': window,
                    'mae': result['mae'],
                    'rmse': result.get('rmse', 0),
                    'correlation': result.get('correlation', 0),
                    'over_under_accuracy': result.get('over_under_accuracy', 0),
                    'prediction_count': result.get('prediction_count', 0)
                })
        
        if windows_data:
            df = pd.DataFrame(windows_data)
            df = df.sort_values('mae')
            
            print(f"{'Window':<8} {'MAE':<8} {'RMSE':<8} {'Corr':<8} {'O/U Acc':<8} {'Preds':<8}")
            print("-" * 60)
            
            for _, row in df.iterrows():
                print(f"{row['window']:<8} {row['mae']:<8.3f} {row['rmse']:<8.3f} "
                      f"{row['correlation']:<8.3f} {row['over_under_accuracy']:<8.3f} {row['prediction_count']:<8}")
                      
            best_window = df.iloc[0]['window']
            best_mae = df.iloc[0]['mae']
            print(f"\n🎯 RECOMMENDATION: Use {best_window} learning window (MAE: {best_mae:.3f})")
            
    def analyze_feature_combinations(self):
        """Detailed analysis of feature combination results"""
        if 'feature_combinations' not in self.results:
            return
            
        print(f"\n🧠 FEATURE COMBINATION DETAILED ANALYSIS")
        print("-" * 50)
        
        features_data = []
        for feature_set, result in self.results['feature_combinations'].items():
            if isinstance(result, dict) and 'mae' in result:
                features_data.append({
                    'feature_set': feature_set,
                    'mae': result['mae'],
                    'correlation': result.get('correlation', 0),
                    'over_under_accuracy': result.get('over_under_accuracy', 0)
                })
        
        if features_data:
            df = pd.DataFrame(features_data)
            df = df.sort_values('mae')
            
            print(f"{'Feature Set':<25} {'MAE':<8} {'Correlation':<12} {'O/U Accuracy':<12}")
            print("-" * 70)
            
            for _, row in df.iterrows():
                print(f"{row['feature_set']:<25} {row['mae']:<8.3f} "
                      f"{row['correlation']:<12.3f} {row['over_under_accuracy']:<12.3f}")
                      
            best_features = df.iloc[0]['feature_set']
            best_mae = df.iloc[0]['mae']
            print(f"\n🎯 RECOMMENDATION: Use '{best_features}' feature set (MAE: {best_mae:.3f})")
            
    def analyze_model_architectures(self):
        """Detailed analysis of model architecture results"""
        if 'model_architectures' not in self.results:
            return
            
        print(f"\n🤖 MODEL ARCHITECTURE DETAILED ANALYSIS")
        print("-" * 50)
        
        models_data = []
        for model_type, result in self.results['model_architectures'].items():
            if isinstance(result, dict) and 'mae' in result:
                models_data.append({
                    'model_type': model_type,
                    'mae': result['mae'],
                    'correlation': result.get('correlation', 0),
                    'prediction_count': result.get('prediction_count', 0)
                })
        
        if models_data:
            df = pd.DataFrame(models_data)
            df = df.sort_values('mae')
            
            print(f"{'Model Type':<20} {'MAE':<8} {'Correlation':<12} {'Predictions':<12}")
            print("-" * 60)
            
            for _, row in df.iterrows():
                print(f"{row['model_type']:<20} {row['mae']:<8.3f} "
                      f"{row['correlation']:<12.3f} {row['prediction_count']:<12}")
                      
            best_model = df.iloc[0]['model_type']
            best_mae = df.iloc[0]['mae']
            print(f"\n🎯 RECOMMENDATION: Use '{best_model}' model (MAE: {best_mae:.3f})")
            
    def calculate_improvement_potential(self):
        """Calculate potential improvement from optimizations"""
        print(f"\n📊 IMPROVEMENT POTENTIAL ANALYSIS")
        print("-" * 50)
        
        baseline_mae = None
        best_overall_mae = float('inf')
        
        # Find baseline (assume 14d window with SGD is baseline)
        if 'learning_windows' in self.results and '14d' in self.results['learning_windows']:
            baseline_mae = self.results['learning_windows']['14d'].get('mae')
            
        # Find best overall MAE across all tests
        for test_type, test_results in self.results.items():
            if test_type in ['test_config', 'recommendations']:
                continue
                
            for config, result in test_results.items():
                if isinstance(result, dict) and 'mae' in result:
                    if result['mae'] < best_overall_mae:
                        best_overall_mae = result['mae']
                        
        if baseline_mae and best_overall_mae < float('inf'):
            improvement = ((baseline_mae - best_overall_mae) / baseline_mae) * 100
            print(f"Baseline MAE (14d window): {baseline_mae:.3f}")
            print(f"Best achievable MAE: {best_overall_mae:.3f}")
            print(f"Potential improvement: {improvement:.1f}%")
            
            if improvement > 2:
                print(f"🚀 SIGNIFICANT IMPROVEMENT POTENTIAL detected!")
            elif improvement > 1:
                print(f"✅ Moderate improvement potential available")
            else:
                print(f"📈 Current configuration is near-optimal")
        else:
            print("Could not calculate improvement potential (insufficient data)")
            
    def generate_implementation_plan(self):
        """Generate step-by-step implementation plan"""
        print(f"\n🛠️  IMPLEMENTATION PLAN")
        print("-" * 50)
        
        recs = self.results.get('recommendations', {}).get('best_configurations', {})
        
        print("Step-by-step optimization implementation:")
        print()
        
        step = 1
        
        if 'learning_windows' in recs:
            window = recs['learning_windows']['config']
            mae = recs['learning_windows']['mae']
            print(f"{step}. Update learning window configuration:")
            print(f"   set INCREMENTAL_LEARNING_DAYS={window.replace('d', '')}")
            print(f"   Expected MAE improvement to: {mae:.3f}")
            print()
            step += 1
            
        if 'feature_combinations' in recs:
            features = recs['feature_combinations']['config']
            mae = recs['feature_combinations']['mae']
            print(f"{step}. Optimize feature combination:")
            print(f"   Implement '{features}' feature set")
            print(f"   Expected MAE improvement to: {mae:.3f}")
            print()
            step += 1
            
        if 'model_architectures' in recs:
            model = recs['model_architectures']['config']
            mae = recs['model_architectures']['mae']
            print(f"{step}. Consider model architecture change:")
            print(f"   Switch to '{model}' model type")
            print(f"   Expected MAE improvement to: {mae:.3f}")
            print()
            step += 1
            
        if 'learning_rates' in recs:
            rate = recs['learning_rates']['config']
            mae = recs['learning_rates']['mae']
            print(f"{step}. Tune learning rate:")
            print(f"   Set learning rate to: {rate}")
            print(f"   Expected MAE improvement to: {mae:.3f}")
            print()
            step += 1
            
        print("🎯 Priority: Implement optimizations in the order listed above")
        print("📝 Test each change individually to validate improvements")
        
    def run_complete_analysis(self):
        """Run complete analysis of A/B testing results"""
        self.print_executive_summary()
        self.analyze_learning_windows()
        self.analyze_feature_combinations()
        self.analyze_model_architectures()
        self.calculate_improvement_potential()
        self.generate_implementation_plan()
        
        print(f"\n" + "="*80)
        print(f"📁 Full results available in: {self.results_file}")
        print(f"="*80)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Analyze comprehensive A/B testing results')
    parser.add_argument('--results-file', help='Specific results file to analyze')
    parser.add_argument('--latest', action='store_true', help='Analyze most recent results')
    parser.add_argument('--summary-only', action='store_true', help='Show only executive summary')
    
    args = parser.parse_args()
    
    try:
        if args.latest:
            analyzer = ABTestAnalyzer()
        else:
            analyzer = ABTestAnalyzer(args.results_file)
            
        if args.summary_only:
            analyzer.print_executive_summary()
        else:
            analyzer.run_complete_analysis()
            
    except Exception as e:
        print(f"Error analyzing results: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
