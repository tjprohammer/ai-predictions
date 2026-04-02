import importlib


app_module = importlib.import_module("src.api.app")


def test_sql_bind_list_populates_named_parameters():
    params: dict[str, int] = {}

    placeholders = app_module._sql_bind_list("player_id", [17, 42], params)

    assert placeholders == ":player_id_0, :player_id_1"
    assert params == {"player_id_0": 17, "player_id_1": 42}


def test_sqlite_helper_fragments_use_portable_casts():
    assert app_module._sql_real("metric", dialect="sqlite") == "CAST(metric AS REAL)"
    assert app_module._sql_integer("slot", dialect="sqlite") == "CAST(slot AS INTEGER)"
    assert app_module._sql_year("game_date", dialect="sqlite") == "CAST(strftime('%Y', game_date) AS INTEGER)"
    assert app_module._sql_order_nulls_last("lineup_slot") == "CASE WHEN lineup_slot IS NULL THEN 1 ELSE 0 END, lineup_slot ASC"


def test_sqlite_boolean_helper_normalizes_text_values():
    fragment = app_module._sql_boolean("flag_text", dialect="sqlite")

    assert "'true'" in fragment
    assert "'false'" in fragment
    assert "THEN 1" in fragment
    assert "THEN 0" in fragment