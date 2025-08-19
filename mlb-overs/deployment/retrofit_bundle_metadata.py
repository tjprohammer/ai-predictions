#!/usr/bin/env python3
"""
Retrofit Bundle Metadata
========================
One-off script to add missing training metadata to an existing model bundle,
with atomic save + backup and robust date handling.
"""

import argparse
import joblib
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import tempfile
import shutil

# Optional: better month arithmetic
try:
    from dateutil.relativedelta import relativedelta
    HAS_REL = True
except Exception:
    HAS_REL = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("retrofit_bundle")

REQUIRED_KEYS = [
    "label_definition",
    "training_period",
    "evaluation_metrics",
    "bias_correction",
    "training_feature_snapshot",
]

def _estimate_training_period(train_iso: str) -> dict:
    # parse ISO, tolerate trailing 'Z'
    try:
        dt = datetime.fromisoformat(train_iso.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.utcnow()
    end = dt.date()
    if HAS_REL:
        start = (dt - relativedelta(months=4)).date()
    else:
        start = (dt - timedelta(days=120)).date()
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_rows": 0,
        "val_rows": 0,
        "note": "Estimated period for retrofitted bundle",
    }

def _atomic_dump(obj, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(target.parent), suffix=".joblib") as tmp:
        tmp_path = Path(tmp.name)
        joblib.dump(obj, tmp_path, compress=3)
    tmp_path.replace(target)

def retrofit_bundle_metadata(bundle_path: Path, dry_run: bool = False,
                            start_override: str = None, end_override: str = None) -> bool:
    if not bundle_path.exists():
        log.error(f"❌ Bundle not found: {bundle_path}")
        return False

    log.info(f"📦 Loading bundle: {bundle_path}")
    b = joblib.load(bundle_path)

    before_keys = set(b.keys())
    added = []

    # 1) label_definition
    if "label_definition" not in b:
        b["label_definition"] = (
            "Regression target: final total_runs from legitimate_game_features.total_runs"
        )
        added.append("label_definition")

    # 2) training_period (estimate or override)
    if "training_period" not in b:
        period = _estimate_training_period(b.get("training_date", datetime.now().isoformat()))
        if start_override:
            period["start"] = start_override
        if end_override:
            period["end"] = end_override
        b["training_period"] = period
        added.append("training_period")

    # 3) evaluation_metrics placeholder
    if "evaluation_metrics" not in b:
        b["evaluation_metrics"] = {
            "MAE": None,
            "RMSE": None,
            "R2": None,
            "note": "Not available for retrofitted bundle; will be populated on next training",
        }
        added.append("evaluation_metrics")

    # 4) bias_correction default
    if "bias_correction" not in b:
        b["bias_correction"] = 0.0
        added.append("bias_correction")

    # 5) training_feature_snapshot (only if we know columns)
    if "training_feature_snapshot" not in b:
        feat_cols = b.get("feature_columns")
        if isinstance(feat_cols, (list, tuple)) and len(feat_cols) > 0:
            b["training_feature_snapshot"] = pd.DataFrame(columns=list(feat_cols))
            added.append("training_feature_snapshot")
        else:
            log.warning("⚠️  feature_columns missing or empty; skipping empty snapshot creation")

    # 6) versioning + feature_sha
    if "trainer_version" not in b:
        b["trainer_version"] = "v2.1_retrofitted"
        added.append("trainer_version")
    b.setdefault("schema_version", "1")
    if "feature_sha" not in b and isinstance(b.get("feature_columns"), (list, tuple)):
        b["feature_sha"] = pd.util.hash_pandas_object(pd.Index(b["feature_columns"])).sum().item()
        added.append("feature_sha")

    # Summary
    after_keys = set(b.keys())
    unchanged = sorted(list(after_keys - set(added)))
    log.info(f"Added keys: {added or 'None'}")
    log.info(f"Unchanged keys: {unchanged[:10]}{' …' if len(unchanged) > 10 else ''}")

    if dry_run:
        log.info("🧪 Dry run: not writing changes.")
        return True

    # Backup + atomic save
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = bundle_path.with_name(bundle_path.stem + f".bak.{ts}" + bundle_path.suffix)
    try:
        shutil.copy2(bundle_path, backup_path)
        log.info(f"🛟 Backup written: {backup_path.name}")
    except Exception as e:
        log.warning(f"Could not create backup: {e}")

    try:
        _atomic_dump(b, bundle_path)
        log.info("✅ Bundle retrofitted and saved atomically.")
        return True
    except Exception as e:
        log.error(f"❌ Failed to save updated bundle: {e}")
        return False

def main():
    ap = argparse.ArgumentParser(description="Retrofit missing training metadata into a model bundle")
    ap.add_argument("--bundle", type=Path, default=Path("../models/legitimate_model_latest.joblib"))
    ap.add_argument("--dry-run", action="store_true", help="Do not write changes")
    ap.add_argument("--start", dest="start_override", help="Override training period start (YYYY-MM-DD)")
    ap.add_argument("--end", dest="end_override", help="Override training period end (YYYY-MM-DD)")
    args = ap.parse_args()
    retrofit_bundle_metadata(args.bundle, args.dry_run, args.start_override, args.end_override)

if __name__ == "__main__":
    main()
