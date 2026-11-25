"""Stock price forecasts from price history, computed classically and with quantum amplitude estimation.

Also contains the original European option pricing study (classical.py, quantum.py, params.py).
"""

from .classical import (
    black_scholes_call,
    discretised_call_price,
    grid_and_pmf,
    mc_call_price,
)
from .params import OptionParams

__all__ = [
    "OptionParams",
    "black_scholes_call",
    "discretised_call_price",
    "grid_and_pmf",
    "mc_call_price",
]
