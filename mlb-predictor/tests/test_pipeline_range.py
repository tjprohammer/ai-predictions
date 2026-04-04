import datetime as dt

from src.utils import pipeline_range as pipeline_range_module


def test_pipeline_range_includes_strikeout_steps(monkeypatch):
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        pipeline_range_module,
        "resolve_date_range",
        lambda _args: (dt.date(2026, 4, 2), dt.date(2026, 4, 2)),
    )
    monkeypatch.setattr(
        pipeline_range_module,
        "_run_step",
        lambda module_name, *args: calls.append((module_name, list(args))),
    )
    monkeypatch.setattr(
        pipeline_range_module.sys,
        "argv",
        ["pipeline_range.py", "--start-date", "2026-04-02", "--end-date", "2026-04-02"],
    )

    result = pipeline_range_module.main()

    assert result == 0
    assert "src.features.strikeouts_builder" in [module for module, _ in calls]
    assert "src.models.predict_strikeouts" in [module for module, _ in calls]
