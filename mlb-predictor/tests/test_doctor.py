from datetime import date

from src.utils import doctor


def test_doctor_json_serializes_dates(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor,
        "app_module",
        type(
            "AppModule",
            (),
            {
                "_doctor_payload": staticmethod(
                    lambda *args, **kwargs: {
                        "target_date": date(2026, 4, 5),
                        "overall": {"status": "ok"},
                        "checks": [],
                        "runtime": {},
                    }
                )
            },
        )(),
    )
    monkeypatch.setattr(
        doctor,
        "build_parser",
        lambda: type(
            "Parser",
            (),
            {
                "parse_args": staticmethod(
                    lambda: type(
                        "Args",
                        (),
                        {
                            "target_date": None,
                            "source_health_hours": 24,
                            "pipeline_limit": 5,
                            "update_history_limit": 5,
                            "json": True,
                        },
                    )()
                )
            },
        )(),
    )

    result = doctor.main()
    captured = capsys.readouterr()

    assert result == 0
    assert '"target_date": "2026-04-05"' in captured.out
