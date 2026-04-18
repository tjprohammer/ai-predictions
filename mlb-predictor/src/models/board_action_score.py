"""Calibrated binary \"action\" score for team best-bet cards from a model fit on ``prediction_outcomes_daily``.

Inference-only at board time; training lives in ``train_board_action_score``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.best_bets import BEST_BET_MARKET_KEYS, to_float
from src.utils.logging import get_logger
from src.utils.settings import get_settings

log = get_logger(__name__)

_MODEL_SUBDIR = "board_action_score"
_ARTIFACT_NAME = "action_classifier.joblib"
_META_NAME = "action_classifier_meta.json"

# Reload artifact from disk when the file changes (retrain) — avoids stale in-process cache on long-lived uvicorn.
_pipeline_cache: dict[str, Any] = {"mtime": None, "pipeline": None}

_MARKET_KEYS_ORDER = tuple(BEST_BET_MARKET_KEYS)
_MARKET_TO_IDX = {k: i for i, k in enumerate(_MARKET_KEYS_ORDER)}
_N_MARKETS = max(len(_MARKET_KEYS_ORDER), 1)


def _float01(x: Any, default: float = 0.0) -> float:
    v = to_float(x)
    if v is None:
        return default
    return float(v)


def feature_vector_from_card(card: dict[str, Any]) -> np.ndarray:
    """Same feature order as training (single row)."""
    mk = str(card.get("market_key") or "")
    m_idx = _MARKET_TO_IDX.get(mk, 0) / float(_N_MARKETS)
    mp = _float01(card.get("model_probability"))
    nv = _float01(card.get("no_vig_probability"))
    push = _float01(card.get("push_probability")) or 0.0
    lp = max(0.0, 1.0 - (mp or 0.0) - push) if mp is not None else 0.0
    wev = _float01(card.get("weighted_ev"))
    pe = _float01(card.get("probability_edge"))
    it = card.get("input_trust") or {}
    trust = _float01(it.get("score")) if isinstance(it, dict) else 0.0
    gcp = _float01(card.get("game_certainty_pct"))
    if gcp is not None and gcp > 1.0:
        gcp = gcp / 100.0
    pos = 1.0 if card.get("positive") else 0.0
    return np.array(
        [[mp or 0.0, nv or 0.0, lp, wev, pe, trust, gcp or 0.0, pos, m_idx]],
        dtype=np.float64,
    )


def _parse_meta(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {}
    try:
        if isinstance(meta, float) and meta != meta:  # NaN
            return {}
    except Exception:
        pass
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except json.JSONDecodeError:
            return {}
    return {}


def feature_vector_from_outcome_row(row: dict[str, Any]) -> np.ndarray:
    """Build the same vector from a DB row + meta_payload (matches ``_build_best_bet_outcomes``)."""
    meta = _parse_meta(row.get("meta_payload"))
    mk = str(row.get("market") or "")
    m_idx = _MARKET_TO_IDX.get(mk, 0) / float(_N_MARKETS)
    mp = _float01(row.get("probability"))
    # Stored no-vig / market implied for the recommended side (see product_surfaces best-bet rows).
    nv = _float01(row.get("market_line"))
    lp = _float01(row.get("opposite_probability"))
    wev = _float01(meta.get("weighted_ev"))
    pe = _float01(meta.get("probability_edge"))
    trust = _float01(meta.get("input_trust_score"))
    gcp = _float01(meta.get("game_certainty_pct"))
    if gcp is not None and gcp > 1.0:
        gcp = gcp / 100.0
    pos = 1.0 if meta.get("is_board_green_pick") or meta.get("is_green_pick") else 0.0
    return np.array(
        [[mp or 0.0, nv or 0.0, lp or 0.0, wev, pe, trust, gcp or 0.0, pos, m_idx]],
        dtype=np.float64,
    )


def artifact_paths() -> tuple[Path, Path]:
    root = get_settings().model_dir / _MODEL_SUBDIR
    return root / _ARTIFACT_NAME, root / _META_NAME


def _load_pipeline():
    """Load sklearn Pipeline; None if missing or unloadable. Cached by artifact mtime."""
    path, meta_path = artifact_paths()
    if not path.exists():
        _pipeline_cache["mtime"] = None
        _pipeline_cache["pipeline"] = None
        return None
    try:
        mtime = float(path.stat().st_mtime)
    except OSError:
        mtime = None
    if (
        mtime is not None
        and _pipeline_cache["mtime"] == mtime
        and _pipeline_cache["pipeline"] is not None
    ):
        return _pipeline_cache["pipeline"]
    try:
        import joblib

        pipe = joblib.load(path)
    except Exception as exc:
        log.warning("board_action_score: could not load %s: %s", path, exc)
        _pipeline_cache["mtime"] = None
        _pipeline_cache["pipeline"] = None
        return None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            feats = meta.get("feature_names")
            if feats and tuple(feats) != training_feature_names():
                log.warning("board_action_score: feature metadata mismatch; still using model")
        except OSError:
            pass
    _pipeline_cache["mtime"] = mtime
    _pipeline_cache["pipeline"] = pipe
    return pipe


def training_feature_names() -> tuple[str, ...]:
    return (
        "model_probability",
        "no_vig_implied",
        "loss_probability",
        "weighted_ev",
        "probability_edge",
        "input_trust_score",
        "game_certainty_pct",
        "positive",
        "market_norm_idx",
    )


def predict_action_win_probability(card: dict[str, Any]) -> float | None:
    """Return P(win) for this card, or None if no model."""
    pipe = _load_pipeline()
    if pipe is None:
        return None
    X = feature_vector_from_card(card)
    try:
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)
            return float(proba[0, 1])
        pred = pipe.predict(X)
        return float(pred[0])
    except Exception as exc:
        log.warning("board_action_score predict failed: %s", exc)
        return None


def maybe_attach_action_score(card: dict[str, Any]) -> dict[str, Any]:
    """Copy card with ``action_score`` / ``action_score_model`` when a model is available."""
    mk = str(card.get("market_key") or "")
    if mk not in BEST_BET_MARKET_KEYS:
        return card
    p = predict_action_win_probability(card)
    if p is None:
        return card
    out = dict(card)
    out["action_score"] = round(float(p), 4)
    out["action_score_model"] = "board_action_logistic_v1"
    return out


def clear_loaded_model_cache() -> None:
    _pipeline_cache["mtime"] = None
    _pipeline_cache["pipeline"] = None
