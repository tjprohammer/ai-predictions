import numpy as np

def kelly_fraction(p: float, american: int) -> float:
    if american is None:
        return 0.0
    if american >= 100:
        b = american / 100.0
    else:
        b = 100.0 / abs(american)
    q = 1 - p
    return max(0.0, (b*p - q)/b)
