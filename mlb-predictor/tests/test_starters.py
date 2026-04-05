import pandas as pd

from src.ingestors import starters as starters_module


def test_starters_parses_sqlite_string_last_start(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        starters_module,
        "resolve_date_range",
        lambda _args: (starters_module.date(2026, 4, 4), starters_module.date(2026, 4, 4)),
    )
    monkeypatch.setattr(
        starters_module,
        "iter_schedule_games",
        lambda *_args: [
            {
                "gamePk": 100,
                "officialDate": "2026-04-04",
                "gameDate": "2026-04-04T23:10:00Z",
                "teams": {
                    "home": {"team": {"id": 10}, "probablePitcher": {"id": 77}},
                    "away": {"team": {"id": 20}},
                },
            }
        ],
    )
    monkeypatch.setattr(
        starters_module,
        "query_df",
        lambda *_args, **_kwargs: pd.DataFrame([{"pitcher_id": 77, "last_game_date": "2026-03-30"}]),
    )
    monkeypatch.setattr(starters_module, "team_dimension_row", lambda team_id: {"team_abbr": f"T{team_id}"})
    monkeypatch.setattr(starters_module, "player_dimension_row", lambda player_id: {"player_id": player_id})
    monkeypatch.setattr(starters_module, "record_ingest_event", lambda **_kwargs: None)
    monkeypatch.setattr(
        starters_module,
        "run_sql",
        lambda sql, params=None: captured.setdefault("run_sql_calls", []).append((sql, params)),
    )

    def fake_upsert_rows(table_name, rows, conflict_columns):
        captured[table_name] = {"rows": rows, "conflict_columns": conflict_columns}
        return len(rows)

    monkeypatch.setattr(starters_module, "upsert_rows", fake_upsert_rows)
    monkeypatch.setattr(
        starters_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: type("Args", (), {"start_date": None, "end_date": None, "target_date": None})(),
    )

    result = starters_module.main()

    assert result == 0
    assert captured["run_sql_calls"][0][1] == {
        "start_date": starters_module.date(2026, 4, 4),
        "end_date": starters_module.date(2026, 4, 4),
        "is_probable": True,
    }
    starter_rows = captured["pitcher_starts"]["rows"]
    assert starter_rows[0]["days_rest"] == 5


def test_starters_rest_lookup_ignores_probable_rows(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        starters_module,
        "resolve_date_range",
        lambda _args: (starters_module.date(2026, 4, 4), starters_module.date(2026, 4, 4)),
    )
    monkeypatch.setattr(starters_module, "iter_schedule_games", lambda *_args: [])

    def fake_query_df(query, *_args, **_kwargs):
        captured["query"] = query
        return pd.DataFrame(columns=["pitcher_id", "last_game_date"])

    monkeypatch.setattr(starters_module, "query_df", fake_query_df)
    monkeypatch.setattr(starters_module, "record_ingest_event", lambda **_kwargs: None)
    monkeypatch.setattr(starters_module, "run_sql", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(starters_module, "upsert_rows", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        starters_module.argparse.ArgumentParser,
        "parse_args",
        lambda self: type("Args", (), {"start_date": None, "end_date": None, "target_date": None})(),
    )

    assert starters_module.main() == 0
    assert "COALESCE(is_probable, FALSE) = FALSE" in captured["query"]