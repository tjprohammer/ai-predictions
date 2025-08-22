#!/usr/bin/env python3
"""
Data Quality Fixer for Enhanced MLB Prediction System
Addresses the critical data quality issues identified in the quality analysis:
1. Starting pitcher data completeness (only 41.8%)
2. Data quality ranges (ERA values of 0.000 are invalid)
3. Feature coverage improvement
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Database and data handling
import psycopg2
import pandas as pd
import numpy as np

class DataQualityFixer:
    def __init__(self, connection_params: Dict[str, str], verbose: bool = False):
        """Initialize the data quality fixer with database connection."""
        self.connection_params = connection_params
        self.verbose = verbose
        self.conn = None
        self.cursor = None
        
        # Track fixes applied
        self.fixes_applied = {
            'era_fixes': 0,
            'whip_fixes': 0,
            'sp_name_fixes': 0,
            'feature_enhancements': 0
        }
        
    def connect_database(self) -> bool:
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.cursor = self.conn.cursor()
            if self.verbose:
                print("✓ Database connection established")
            return True
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return False
            
    def disconnect_database(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            
    def fix_era_values(self, start_date: str, end_date: str) -> int:
        """Fix invalid ERA values (0.000 should be NULL or reasonable estimate)."""
        if self.verbose:
            print(f"\n🔧 Fixing invalid ERA values")
            
        # First, identify games with 0.000 ERA
        check_query = """
        SELECT game_id, date, home_team, away_team, home_sp_name, away_sp_name,
               home_sp_season_era, away_sp_season_era
        FROM enhanced_games 
        WHERE date >= %s AND date <= %s
        AND (home_sp_season_era = 0.000 OR away_sp_season_era = 0.000)
        ORDER BY date DESC
        """
        
        self.cursor.execute(check_query, (start_date, end_date))
        invalid_era_games = self.cursor.fetchall()
        
        if self.verbose:
            print(f"  Found {len(invalid_era_games)} games with 0.000 ERA values")
            
        fixes_count = 0
        
        for game in invalid_era_games:
            game_id, date, home_team, away_team, home_sp, away_sp, home_era, away_era = game
            
            # Strategy 1: Set 0.000 ERA to NULL (better than invalid data)
            updates = []
            if home_era == 0.000:
                updates.append("home_sp_season_era = NULL")
                if self.verbose:
                    print(f"    {date}: {home_team} - {home_sp} ERA 0.000 → NULL")
                    
            if away_era == 0.000:
                updates.append("away_sp_season_era = NULL")
                if self.verbose:
                    print(f"    {date}: {away_team} - {away_sp} ERA 0.000 → NULL")
            
            if updates:
                update_query = f"""
                UPDATE enhanced_games 
                SET {', '.join(updates)}
                WHERE game_id = %s
                """
                self.cursor.execute(update_query, (game_id,))
                fixes_count += 1
                
        self.conn.commit()
        if self.verbose:
            print(f"  ✓ Fixed {fixes_count} invalid ERA values")
            
        return fixes_count
        
    def enhance_starting_pitcher_data(self, start_date: str, end_date: str) -> int:
        """Enhance starting pitcher data by filling missing information."""
        if self.verbose:
            print(f"\n📈 Enhancing starting pitcher data")
            
        # Check for games missing SP names but having other SP data
        missing_names_query = """
        SELECT game_id, date, home_team, away_team, 
               home_sp_name, away_sp_name,
               home_sp_id, away_sp_id,
               home_sp_season_era, away_sp_season_era
        FROM enhanced_games 
        WHERE date >= %s AND date <= %s
        AND (home_sp_name IS NULL OR away_sp_name IS NULL)
        AND (home_sp_id IS NOT NULL OR away_sp_id IS NOT NULL)
        ORDER BY date DESC
        """
        
        self.cursor.execute(missing_names_query, (start_date, end_date))
        missing_name_games = self.cursor.fetchall()
        
        if self.verbose:
            print(f"  Found {len(missing_name_games)} games with missing SP names but have SP IDs")
            
        # Strategy: Create placeholder names from team + position
        fixes_count = 0
        
        for game in missing_name_games:
            game_id, date, home_team, away_team, home_sp_name, away_sp_name, home_sp_id, away_sp_id, home_era, away_era = game
            
            updates = []
            if home_sp_name is None and home_sp_id is not None:
                placeholder_name = f"{home_team}_SP_{home_sp_id}"
                updates.append(f"home_sp_name = '{placeholder_name}'")
                if self.verbose:
                    print(f"    {date}: {home_team} SP name NULL → {placeholder_name}")
                    
            if away_sp_name is None and away_sp_id is not None:
                placeholder_name = f"{away_team}_SP_{away_sp_id}"
                updates.append(f"away_sp_name = '{placeholder_name}'")
                if self.verbose:
                    print(f"    {date}: {away_team} SP name NULL → {placeholder_name}")
            
            if updates:
                update_query = f"""
                UPDATE enhanced_games 
                SET {', '.join(updates)}
                WHERE game_id = %s
                """
                self.cursor.execute(update_query, (game_id,))
                fixes_count += 1
                
        self.conn.commit()
        if self.verbose:
            print(f"  ✓ Enhanced {fixes_count} starting pitcher records")
            
        return fixes_count
        
    def calculate_missing_whip(self, start_date: str, end_date: str) -> int:
        """Calculate WHIP for records that have component stats but missing WHIP."""
        if self.verbose:
            print(f"\n🧮 Calculating missing WHIP values")
            
        # Find games with missing WHIP but have IP, H, BB data
        missing_whip_query = """
        SELECT game_id, date, home_team, away_team,
               home_sp_season_ip, home_sp_season_k, home_sp_h, home_sp_bb,
               away_sp_season_ip, away_sp_season_k, away_sp_h, away_sp_bb,
               home_sp_whip, away_sp_whip
        FROM enhanced_games 
        WHERE date >= %s AND date <= %s
        AND (
            (home_sp_whip IS NULL AND home_sp_season_ip IS NOT NULL AND home_sp_h IS NOT NULL AND home_sp_bb IS NOT NULL) OR
            (away_sp_whip IS NULL AND away_sp_season_ip IS NOT NULL AND away_sp_h IS NOT NULL AND away_sp_bb IS NOT NULL)
        )
        ORDER BY date DESC
        """
        
        self.cursor.execute(missing_whip_query, (start_date, end_date))
        missing_whip_games = self.cursor.fetchall()
        
        if self.verbose:
            print(f"  Found {len(missing_whip_games)} games where WHIP can be calculated")
            
        fixes_count = 0
        
        for game in missing_whip_games:
            game_id, date, home_team, away_team, home_ip, home_k, home_h, home_bb, away_ip, away_k, away_h, away_bb, home_whip, away_whip = game
            
            updates = []
            
            # Calculate WHIP = (Hits + Walks) / Innings Pitched
            if home_whip is None and home_ip and home_h is not None and home_bb is not None and float(home_ip) > 0:
                calculated_whip = (float(home_h) + float(home_bb)) / float(home_ip)
                if 0.5 <= calculated_whip <= 3.0:  # Reasonable range
                    updates.append(f"home_sp_whip = {calculated_whip:.3f}")
                    if self.verbose:
                        print(f"    {date}: {home_team} WHIP calculated: {calculated_whip:.3f}")
                        
            if away_whip is None and away_ip and away_h is not None and away_bb is not None and float(away_ip) > 0:
                calculated_whip = (float(away_h) + float(away_bb)) / float(away_ip)
                if 0.5 <= calculated_whip <= 3.0:  # Reasonable range
                    updates.append(f"away_sp_whip = {calculated_whip:.3f}")
                    if self.verbose:
                        print(f"    {date}: {away_team} WHIP calculated: {calculated_whip:.3f}")
            
            if updates:
                update_query = f"""
                UPDATE enhanced_games 
                SET {', '.join(updates)}
                WHERE game_id = %s
                """
                self.cursor.execute(update_query, (game_id,))
                fixes_count += 1
                
        self.conn.commit()
        if self.verbose:
            print(f"  ✓ Calculated WHIP for {fixes_count} games")
            
        return fixes_count
        
    def enhance_feature_coverage(self, start_date: str, end_date: str) -> int:
        """Enhance feature coverage by deriving additional metrics."""
        if self.verbose:
            print(f"\n📊 Enhancing feature coverage")
            
        # Add derived pitcher performance metrics
        derived_metrics_query = """
        UPDATE enhanced_games 
        SET 
            -- Add pitcher effectiveness ratio (K/BB)
            home_sp_k_bb_ratio = CASE 
                WHEN home_sp_bb > 0 THEN ROUND(home_sp_season_k::numeric / home_sp_bb::numeric, 2)
                ELSE NULL 
            END,
            away_sp_k_bb_ratio = CASE 
                WHEN away_sp_bb > 0 THEN ROUND(away_sp_season_k::numeric / away_sp_bb::numeric, 2)
                ELSE NULL 
            END,
            -- Add innings per start estimate
            home_sp_ip_per_start = CASE 
                WHEN home_sp_season_ip IS NOT NULL THEN ROUND(home_sp_season_ip / 32.0, 1)
                ELSE NULL 
            END,
            away_sp_ip_per_start = CASE 
                WHEN away_sp_season_ip IS NOT NULL THEN ROUND(away_sp_season_ip / 32.0, 1)
                ELSE NULL 
            END
        WHERE date >= %s AND date <= %s
        AND (home_sp_k_bb_ratio IS NULL OR away_sp_k_bb_ratio IS NULL 
             OR home_sp_ip_per_start IS NULL OR away_sp_ip_per_start IS NULL)
        """
        
        # Check if the columns exist first
        try:
            self.cursor.execute("SELECT home_sp_k_bb_ratio FROM enhanced_games LIMIT 1")
        except psycopg2.errors.UndefinedColumn:
            # Add the new columns
            if self.verbose:
                print("  Adding new derived metric columns...")
                
            add_columns_query = """
            ALTER TABLE enhanced_games 
            ADD COLUMN IF NOT EXISTS home_sp_k_bb_ratio NUMERIC,
            ADD COLUMN IF NOT EXISTS away_sp_k_bb_ratio NUMERIC,
            ADD COLUMN IF NOT EXISTS home_sp_ip_per_start NUMERIC,
            ADD COLUMN IF NOT EXISTS away_sp_ip_per_start NUMERIC
            """
            self.cursor.execute(add_columns_query)
            self.conn.commit()
            
        # Now update the derived metrics
        self.cursor.execute(derived_metrics_query, (start_date, end_date))
        fixes_count = self.cursor.rowcount
        self.conn.commit()
        
        if self.verbose:
            print(f"  ✓ Enhanced {fixes_count} games with derived metrics")
            
        return fixes_count
        
    def validate_fixes(self, start_date: str, end_date: str) -> Dict:
        """Validate that the fixes improved data quality."""
        if self.verbose:
            print(f"\n✅ Validating data quality improvements")
            
        # Re-run quality checks
        validation_query = """
        SELECT 
            COUNT(*) as total_games,
            COUNT(home_sp_name) as home_sp_name_count,
            COUNT(away_sp_name) as away_sp_name_count,
            COUNT(CASE WHEN home_sp_season_era > 0.5 AND home_sp_season_era <= 15.0 THEN 1 END) as home_era_valid,
            COUNT(CASE WHEN away_sp_season_era > 0.5 AND away_sp_season_era <= 15.0 THEN 1 END) as away_era_valid,
            COUNT(home_sp_whip) as home_whip_count,
            COUNT(away_sp_whip) as away_whip_count,
            COUNT(CASE WHEN home_sp_season_era = 0.000 THEN 1 END) as home_zero_era,
            COUNT(CASE WHEN away_sp_season_era = 0.000 THEN 1 END) as away_zero_era
        FROM enhanced_games 
        WHERE date >= %s AND date <= %s
        """
        
        self.cursor.execute(validation_query, (start_date, end_date))
        validation_data = self.cursor.fetchone()
        
        total = validation_data[0]
        validation_results = {
            'total_games': total,
            'home_sp_name_coverage': round(validation_data[1] / total * 100, 2),
            'away_sp_name_coverage': round(validation_data[2] / total * 100, 2),
            'home_era_validity': round(validation_data[3] / total * 100, 2),
            'away_era_validity': round(validation_data[4] / total * 100, 2),
            'home_whip_coverage': round(validation_data[5] / total * 100, 2),
            'away_whip_coverage': round(validation_data[6] / total * 100, 2),
            'home_zero_era_count': validation_data[7],
            'away_zero_era_count': validation_data[8]
        }
        
        if self.verbose:
            print(f"  Total games: {total}")
            print(f"  Home SP name coverage: {validation_results['home_sp_name_coverage']}%")
            print(f"  Away SP name coverage: {validation_results['away_sp_name_coverage']}%")
            print(f"  Home ERA validity: {validation_results['home_era_validity']}%")
            print(f"  Away ERA validity: {validation_results['away_era_validity']}%")
            print(f"  Home WHIP coverage: {validation_results['home_whip_coverage']}%")
            print(f"  Away WHIP coverage: {validation_results['away_whip_coverage']}%")
            print(f"  Remaining zero ERA values: {validation_results['home_zero_era_count'] + validation_results['away_zero_era_count']}")
            
        return validation_results
        
    def run_comprehensive_fixes(self, start_date: str = None, end_date: str = None) -> Dict:
        """Run all data quality fixes."""
        if not self.connect_database():
            return {}
            
        # Default date range: last 30 days
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"🔧 Running Comprehensive Data Quality Fixes")
        print(f"📅 Fix Period: {start_date} to {end_date}")
        print("=" * 60)
        
        try:
            # Apply all fixes
            self.fixes_applied['era_fixes'] = self.fix_era_values(start_date, end_date)
            self.fixes_applied['sp_name_fixes'] = self.enhance_starting_pitcher_data(start_date, end_date)
            self.fixes_applied['whip_fixes'] = self.calculate_missing_whip(start_date, end_date)
            self.fixes_applied['feature_enhancements'] = self.enhance_feature_coverage(start_date, end_date)
            
            # Validate improvements
            validation_results = self.validate_fixes(start_date, end_date)
            
            # Summary
            print(f"\n📋 FIX SUMMARY")
            print("=" * 60)
            print(f"🔧 ERA values fixed: {self.fixes_applied['era_fixes']}")
            print(f"📝 SP names enhanced: {self.fixes_applied['sp_name_fixes']}")
            print(f"🧮 WHIP values calculated: {self.fixes_applied['whip_fixes']}")
            print(f"📊 Features enhanced: {self.fixes_applied['feature_enhancements']}")
            
            total_fixes = sum(self.fixes_applied.values())
            print(f"✅ Total fixes applied: {total_fixes}")
            
            # Quality improvement summary
            if validation_results:
                print(f"\n📈 QUALITY IMPROVEMENTS")
                print("=" * 60)
                print(f"SP Name Coverage: {validation_results['home_sp_name_coverage']}%")
                print(f"ERA Validity: {validation_results['home_era_validity']}%")
                print(f"WHIP Coverage: {validation_results['home_whip_coverage']}%")
                
            return {
                'fixes_applied': self.fixes_applied,
                'validation_results': validation_results,
                'total_fixes': total_fixes
            }
            
        except Exception as e:
            print(f"✗ Fix process failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.disconnect_database()
            
        return {}


def main():
    parser = argparse.ArgumentParser(description='Data Quality Fixer for Enhanced MLB Prediction System')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Database connection parameters
    connection_params = {
        'host': 'localhost',
        'database': 'mlb', 
        'user': 'mlbuser',
        'password': 'mlbpass'
    }
    
    # Create fixer and run fixes
    fixer = DataQualityFixer(connection_params, verbose=args.verbose)
    results = fixer.run_comprehensive_fixes(args.start_date, args.end_date)
    
    # Final status
    if results and results.get('total_fixes', 0) > 0:
        print(f"\n✅ SUCCESS: Applied {results['total_fixes']} data quality fixes")
        return 0
    elif results and results.get('total_fixes', 0) == 0:
        print(f"\n✅ COMPLETE: No fixes needed - data quality is good")
        return 0
    else:
        print(f"\n❌ FAILED: Could not complete data quality fixes")
        return 1


if __name__ == "__main__":
    sys.exit(main())
