"""
Inspects the collected historical data for a specific date (basic version without ERA).
"""
import pandas as pd
import argparse
import statsapi

def inspect_data_for_date(data_path: str, target_date: str):
    """
    Loads the historical data and displays games for a specific date.
    """
    print(f"Loading historical data from {data_path}...")
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date']).dt.date

    print("\n--- DataFrame Columns ---")
    print(df.columns.tolist())

    print(f"Filtering for games on {target_date}...")
    target_games = df[df['date'] == pd.to_datetime(target_date).date()]

    if target_games.empty:
        print(f"No games found for {target_date}.")
        return

    print(f"\n--- Games on {target_date} ---")
    
    for _, game in target_games.iterrows():
        home_sp_id = game['home_sp_id']
        away_sp_id = game['away_sp_id']

        home_sp_name = "N/A"
        if pd.notna(home_sp_id):
            try:
                home_sp_name = statsapi.lookup_player(int(home_sp_id))[0]['fullName']
            except Exception:
                home_sp_name = f"ID: {int(home_sp_id)}"

        away_sp_name = "N/A"
        if pd.notna(away_sp_id):
            try:
                away_sp_name = statsapi.lookup_player(int(away_sp_id))[0]['fullName']
            except Exception:
                away_sp_name = f"ID: {int(away_sp_id)}"

        market_total = game.get('market_total', 'N/A')
        print(
            f"Matchup: {game['away_team']} @ {game['home_team']}\n"
            f"  Score: {game['away_score']} - {game['home_score']} (Total: {game['total_runs']}, O/U: {market_total})\n"
            f"  Away Pitcher: {away_sp_name}\n"
            f"  Home Pitcher: {home_sp_name}\n"
        )

def main():
    parser = argparse.ArgumentParser(description="Inspect historical MLB game data (basic version).")
    parser.add_argument("--date", type=str, default="2025-08-11", help="Date to inspect (YYYY-MM-DD).")
    parser.add_argument("--data-path", type=str, default="S:/Projects/AI_Predictions/mlb-overs/data/historical_games_2025.parquet", help="Path to historical data.")
    args = parser.parse_args()

    inspect_data_for_date(args.data_path, args.date)

if __name__ == "__main__":
    main()
