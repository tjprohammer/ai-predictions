"""
Starting pitchers game logs + Statcast.
"""

# TODO: adapt from real_data_only_collector.py / pitcher_vs_team_analysis.py

def load_pitchers(date=None):
    return None

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--start")
    p.add_argument("--end")
    args = p.parse_args()
    print(f"[ingestors.pitchers_last10] stub ingest start={args.start} end={args.end}")
    sys.exit(0)
