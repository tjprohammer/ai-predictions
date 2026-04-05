import importlib


app_module = importlib.import_module("src.api.app")


def test_html_shell_routes_disable_caching():
    responses = [
        app_module.index(),
        app_module.hot_hitters_page(),
        app_module.results_page(),
        app_module.doctor_page(),
        app_module.totals_page(),
        app_module.pitchers_page(),
        app_module.game_page(),
    ]

    for response in responses:
        assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate, max-age=0"
        assert response.headers["pragma"] == "no-cache"
        assert response.headers["expires"] == "0"