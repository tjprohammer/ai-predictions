import pandas as pd
import pytest

from src.features.contracts import STRIKEOUTS_META_COLUMNS, TOTALS_META_COLUMNS, validate_columns


def test_validate_columns_accepts_complete_frame():
    frame = pd.DataFrame([{column: 1 for column in TOTALS_META_COLUMNS}])
    validate_columns(frame, TOTALS_META_COLUMNS, "totals")


def test_validate_columns_raises_for_missing_columns():
    with pytest.raises(ValueError):
        validate_columns(pd.DataFrame([{"game_id": 1}]), TOTALS_META_COLUMNS, "totals")


def test_validate_columns_accepts_strikeout_meta_frame():
    frame = pd.DataFrame([{column: 1 for column in STRIKEOUTS_META_COLUMNS}])
    validate_columns(frame, STRIKEOUTS_META_COLUMNS, "strikeouts")