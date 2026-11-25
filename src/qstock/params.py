"""Option and model parameters shared by the classical and quantum code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptionParams:
    """European call under geometric Brownian motion.

    Defaults follow the example used in the Qiskit Finance tutorials so results
    can be sanity-checked against the literature.

    Attributes:
        S0: Spot price today.
        K: Strike price.
        r: Risk-free rate (continuous compounding).
        vol: Annualised volatility.
        T: Time to maturity in years.
        n_std: Truncation width of the S_T distribution, in standard deviations
            around its mean. The quantum circuit can only load a finite grid.
    """

    S0: float = 2.0
    K: float = 1.9
    r: float = 0.05
    vol: float = 0.4
    T: float = 40 / 365
    n_std: float = 3.0

    @property
    def mu(self) -> float:
        """Mean of log(S_T)."""
        return float((self.r - 0.5 * self.vol**2) * self.T + np.log(self.S0))

    @property
    def sigma(self) -> float:
        """Standard deviation of log(S_T)."""
        return float(self.vol * np.sqrt(self.T))

    @property
    def discount(self) -> float:
        return float(np.exp(-self.r * self.T))

    @property
    def bounds(self) -> tuple[float, float]:
        """Truncation interval [low, high] for S_T used by the discretisation."""
        mean = np.exp(self.mu + self.sigma**2 / 2)
        std = np.sqrt((np.exp(self.sigma**2) - 1) * np.exp(2 * self.mu + self.sigma**2))
        low = max(0.0, float(mean - self.n_std * std))
        high = float(mean + self.n_std * std)
        if not low < self.K < high:
            raise ValueError(
                f"Strike {self.K} must lie inside the truncation interval "
                f"({low:.3f}, {high:.3f}); adjust n_std or the parameters."
            )
        return low, high
