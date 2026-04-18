from src.models import retrain_models


def test_retrain_models_stops_on_first_nonzero_exit(monkeypatch):
    called: list[str] = []

    def ok():
        called.append("ok")
        return 0

    def bad():
        called.append("bad")
        return 2

    monkeypatch.setattr(
        retrain_models,
        "_TRAINERS",
        (("first", ok), ("second", bad), ("third", ok)),
    )
    monkeypatch.setattr(retrain_models, "train_board_action_score_main", lambda: 0)
    assert retrain_models.main() == 2
    assert called == ["ok", "bad"]


def test_retrain_models_runs_all_when_success(monkeypatch):
    called: list[str] = []

    def step(name: str):
        def _inner() -> int:
            called.append(name)
            return 0

        return _inner

    monkeypatch.setattr(
        retrain_models,
        "_TRAINERS",
        (("a", step("a")), ("b", step("b"))),
    )
    monkeypatch.setattr(retrain_models, "train_board_action_score_main", lambda: 0)
    assert retrain_models.main() == 0
    assert called == ["a", "b"]
