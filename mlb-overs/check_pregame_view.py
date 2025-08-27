#!/usr/bin/env python3
"""
Sanity check the pregame features view
"""
from sqlalchemy import create_engine, text

def check_pregame_view():
    """Check the pregame features view data quality across different dates"""
    
    engine = create_engine("postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    
    with engine.connect() as conn:
        # Check data by month to see evolution
        print("=== PREGAME FEATURES BY DATE RANGE ===")
        date_ranges = [
            ("2024-04-01", "2024-04-30", "Early 2024"),
            ("2024-07-01", "2024-07-31", "Mid 2024"),  
            ("2025-04-01", "2025-04-30", "Early 2025"),
            ("2025-07-01", "2025-07-31", "Mid 2025"),
        ]
        
        for start_date, end_date, label in date_ranges:
            print(f"\n--- {label} ({start_date} to {end_date}) ---")
            
            # Sample games from this period
            result = conn.execute(text("""
                SELECT game_id, date, 
                       home_sp_era_l3_asof, away_sp_era_l3_asof,
                       home_sp_whip_l3_asof, away_sp_whip_l3_asof,
                       home_runs_pg_14_asof, away_runs_pg_14_asof,
                       home_bp_ip_3d_asof, away_bp_ip_3d_asof
                FROM pregame_features_v1 
                WHERE date BETWEEN :start_date AND :end_date
                  AND (home_sp_era_l3_asof IS NOT NULL OR away_sp_era_l3_asof IS NOT NULL)
                ORDER BY date DESC LIMIT 3
            """), {"start_date": start_date, "end_date": end_date})
            
            for row in result:
                home_era = f"{row.home_sp_era_l3_asof:.2f}" if row.home_sp_era_l3_asof else "N/A"
                away_era = f"{row.away_sp_era_l3_asof:.2f}" if row.away_sp_era_l3_asof else "N/A"
                home_whip = f"{row.home_sp_whip_l3_asof:.2f}" if row.home_sp_whip_l3_asof else "N/A"
                away_whip = f"{row.away_sp_whip_l3_asof:.2f}" if row.away_sp_whip_l3_asof else "N/A"
                home_runs = f"{row.home_runs_pg_14_asof:.1f}" if row.home_runs_pg_14_asof else "N/A"
                away_runs = f"{row.away_runs_pg_14_asof:.1f}" if row.away_runs_pg_14_asof else "N/A"
                home_bp = f"{row.home_bp_ip_3d_asof:.1f}" if row.home_bp_ip_3d_asof else "N/A"
                away_bp = f"{row.away_bp_ip_3d_asof:.1f}" if row.away_bp_ip_3d_asof else "N/A"
                print(f"  {row.game_id} | {row.date}")
                print(f"    ERA L3: {home_era} vs {away_era} | WHIP L3: {home_whip} vs {away_whip}")
                print(f"    Runs L14: {home_runs} vs {away_runs} | BP 3d: {home_bp} vs {away_bp}")
            
            # Feature completeness for this period
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) AS total_games,
                    SUM((home_sp_era_l3_asof IS NOT NULL)::int) AS home_era_count,
                    SUM((away_sp_era_l3_asof IS NOT NULL)::int) AS away_era_count,
                    SUM((home_sp_whip_l3_asof IS NOT NULL)::int) AS home_whip_count,
                    SUM((away_sp_whip_l3_asof IS NOT NULL)::int) AS away_whip_count,
                    SUM((home_runs_pg_14_asof IS NOT NULL)::int) AS home_runs_count,
                    SUM((away_runs_pg_14_asof IS NOT NULL)::int) AS away_runs_count,
                    SUM((home_bp_ip_3d_asof IS NOT NULL)::int) AS home_bp_count,
                    AVG(home_sp_era_l3_asof) AS avg_home_era,
                    AVG(away_sp_era_l3_asof) AS avg_away_era
                FROM pregame_features_v1
                WHERE date BETWEEN :start_date AND :end_date
            """), {"start_date": start_date, "end_date": end_date})
            
            row = result.fetchone()
            if row.total_games > 0:
                print(f"    Games: {row.total_games} | ERA coverage: {100*row.home_era_count/row.total_games:.0f}%/{100*row.away_era_count/row.total_games:.0f}%")
                print(f"    WHIP coverage: {100*row.home_whip_count/row.total_games:.0f}%/{100*row.away_whip_count/row.total_games:.0f}%")
                print(f"    Runs coverage: {100*row.home_runs_count/row.total_games:.0f}%/{100*row.away_runs_count/row.total_games:.0f}%")
                if row.avg_home_era:
                    print(f"    Avg ERA: {row.avg_home_era:.2f}/{row.avg_away_era:.2f}")
            else:
                print(f"    No games found in this period")
        
        # Overall summary
        print("\n=== OVERALL FEATURE COMPLETENESS ===")
        result = conn.execute(text("""
            SELECT 
                COUNT(*) AS total_games,
                SUM((home_sp_era_l3_asof IS NOT NULL)::int) AS home_era_count,
                SUM((away_sp_era_l3_asof IS NOT NULL)::int) AS away_era_count,
                SUM((home_sp_whip_l3_asof IS NOT NULL)::int) AS home_whip_count,
                SUM((away_sp_whip_l3_asof IS NOT NULL)::int) AS away_whip_count,
                SUM((home_runs_pg_14_asof IS NOT NULL)::int) AS home_runs_count,
                SUM((away_runs_pg_14_asof IS NOT NULL)::int) AS away_runs_count,
                MIN(date) AS earliest_date,
                MAX(date) AS latest_date
            FROM pregame_features_v1
        """))
        
        row = result.fetchone()
        print(f"Date range: {row.earliest_date} to {row.latest_date}")
        print(f"Total games: {row.total_games}")
        print(f"Home ERA L3: {row.home_era_count} ({100*row.home_era_count/row.total_games:.1f}%)")
        print(f"Away ERA L3: {row.away_era_count} ({100*row.away_era_count/row.total_games:.1f}%)")
        print(f"Home WHIP L3: {row.home_whip_count} ({100*row.home_whip_count/row.total_games:.1f}%)")
        print(f"Away WHIP L3: {row.away_whip_count} ({100*row.away_whip_count/row.total_games:.1f}%)")
        print(f"Home Runs L14: {row.home_runs_count} ({100*row.home_runs_count/row.total_games:.1f}%)")
        print(f"Away Runs L14: {row.away_runs_count} ({100*row.away_runs_count/row.total_games:.1f}%)")

if __name__ == "__main__":
    check_pregame_view()
