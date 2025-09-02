import argparse
from .hitprops_predictor import run as run_predict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    out = run_predict(args.date)
    if out is not None:
        print(f"[hitprops] wrote {len(out)} rows for {args.date}")

if __name__ == "__main__":
    main()
