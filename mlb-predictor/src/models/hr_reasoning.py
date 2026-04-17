"""Human-readable drivers for home-run probability (product copy, not SHAP)."""

from __future__ import annotations

from typing import Any


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _wind_from_compass_16(deg: float) -> str:
    """Meteorological: direction the wind blows *from* (degrees clockwise from N)."""
    labels = (
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    )
    x = float(deg) % 360.0
    idx = int((x + 11.25) / 22.5) % 16
    return labels[idx]


def build_hr_reasoning_lines(row: dict[str, Any]) -> list[str]:
    """Short bullets explaining why the model leans HR / no-HR for this plate appearance context."""
    lines: list[str] = []
    hrb = _f(row.get("hr_per_pa_blended"))
    if hrb is not None:
        # ``hrb`` is HR per PA as a rate in [0, 1] (e.g. 0.037 ≈ 3.7% of PA).
        pct = hrb * 100.0
        if hrb >= 0.055:
            lines.append(f"Elite HR rate in rolling blend (~{pct:.1f}% of PA).")
        elif hrb >= 0.038:
            lines.append(f"Strong HR rate recently (~{pct:.1f}% of PA).")
        elif hrb >= 0.026:
            lines.append(f"Solid HR rate vs league baseline (~{pct:.1f}% of PA).")

    xw = _f(row.get("xwoba_14"))
    if xw is not None and xw >= 0.400:
        lines.append(f"Hot quality-of-contact window (xwOBA ~{xw:.3f} last 14d).")

    hh = _f(row.get("hard_hit_pct_14"))
    if hh is not None and hh >= 0.48:
        lines.append(f"High hard-hit rate in the two-week lookback (~{hh * 100:.0f}%).")

    park = _f(row.get("park_hr_factor"))
    if park is not None and park >= 1.08:
        lines.append("HR-friendly park environment.")
    elif park is not None and park <= 0.94:
        lines.append("Pitcher-friendly park suppresses long balls.")

    barrel = _f(row.get("opposing_starter_barrel_pct"))
    if barrel is not None and barrel >= 0.095:
        lines.append("Opposing starter has allowed elevated barrel contact.")

    op_hr9 = _f(row.get("opposing_starter_hr_per_9"))
    if op_hr9 is not None and op_hr9 >= 1.65:
        lines.append(f"Opposing starter has allowed elevated HR volume (~{op_hr9:.2f} HR/9, blended recent starts).")
    elif op_hr9 is not None and op_hr9 <= 0.85:
        lines.append(f"Opposing starter has suppressed HRs (~{op_hr9:.2f} HR/9, blended recent starts).")

    wind = _f(row.get("wind_speed_mph"))
    wdir = _f(row.get("wind_direction_deg"))
    temp = _f(row.get("temperature_f"))
    if wind is not None and wind >= 6 and wdir is not None:
        lines.append(f"Wind ~{wind:.0f} mph from the {_wind_from_compass_16(wdir)} (carry depends on how that aligns with the park).")
    elif wind is not None and wind >= 12 and temp is not None and temp >= 72:
        lines.append("Warm air plus double-digit wind — small boost to fly-ball carry.")

    if not lines:
        lines.append("Model blends HR/PA trends, batted-ball quality, park, and matchup context.")

    return lines[:7]
