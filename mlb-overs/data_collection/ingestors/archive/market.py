"""
Market line snapshots open→close.
"""

def load_market(date=None):
    return None

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", action="store_true")
    args = p.parse_args()
    print(f"[ingestors.odds_totals] stub snapshot={args.snapshot}")
    sys.exit(0)
