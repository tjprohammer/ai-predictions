"""Run every training lane in one process for the dashboard \"Retrain Models\" action.

Training used to invoke seven separate subprocesses; a single subprocess avoids extra
spawn overhead and keeps one continuous training phase (still one long step — use
``make start-app-stable`` when developing with uvicorn --reload).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from src.models.train_first5_totals import main as train_first5_totals_main
from src.models.train_hits import main as train_hits_main
from src.models.train_hr import main as train_hr_main
from src.models.train_inning1_nrfi import main as train_inning1_nrfi_main
from src.models.train_strikeouts import main as train_strikeouts_main
from src.models.train_board_action_score import main as train_board_action_score_main
from src.models.train_total_bases import main as train_total_bases_main
from src.models.train_totals import main as train_totals_main
from src.utils.logging import get_logger

log = get_logger(__name__)

_TRAINERS: Sequence[tuple[str, Callable[[], int]]] = (
    ("totals", train_totals_main),
    ("first5_totals", train_first5_totals_main),
    ("inning1_nrfi", train_inning1_nrfi_main),
    ("hits", train_hits_main),
    ("hr", train_hr_main),
    ("total_bases", train_total_bases_main),
    ("strikeouts", train_strikeouts_main),
)


def main() -> int:
    log.warning(
        "RetrainModels — training all lanes in one process (often 15–45+ minutes total). "
        "Do not stop the API. With uvicorn --reload, use `make start-app-stable`."
    )
    for lane, trainer in _TRAINERS:
        log.info("RetrainModels — starting train_%s", lane)
        result = trainer()
        code = 0 if result is None else int(result)
        if code != 0:
            log.error("RetrainModels — train_%s exited with code %s", lane, code)
            return code
        log.info("RetrainModels — finished train_%s", lane)
    log.info("RetrainModels — board_action_score (from outcomes, optional)")
    try:
        action_code = train_board_action_score_main()
        if action_code not in (0, None):
            log.warning("RetrainModels — train_board_action_score exited with code %s", action_code)
    except Exception as exc:
        log.warning("RetrainModels — train_board_action_score failed (non-fatal): %s", exc)
    log.info("RetrainModels — all training lanes finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
