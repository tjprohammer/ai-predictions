from src.models.hr_reasoning import build_hr_reasoning_lines


def test_hr_reasoning_includes_power_signals():
    lines = build_hr_reasoning_lines(
        {
            "hr_per_pa_blended": 0.05,
            "xwoba_14": 0.42,
            "park_hr_factor": 1.1,
            "opposing_starter_barrel_pct": 0.1,
        }
    )
    text = " ".join(lines).lower()
    assert "% of pa" in text or "park" in text


def test_hr_reasoning_fallback_when_sparse():
    lines = build_hr_reasoning_lines({"hr_per_pa_blended": 0.01})
    assert len(lines) >= 1
