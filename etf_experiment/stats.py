"""Weighted means and household-clustered bootstrap intervals.

The unit of resampling is the household, never the household-condition row. A
household contributes fifteen conditions that are internally correlated, so
drawing rows independently would understate the interval by a wide margin.
Every interval in the report comes from ``cluster_bootstrap`` via ``weighted_ci``.

The default seed matches the one recorded in the experiment plan, so re-running
a table reproduces the published numbers exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DRAWS = 1000
SEED = 20261021
NEAR_ZERO = 1e-4


@dataclass(frozen=True)
class Estimate:
    point: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return (self.low > 0 and self.high > 0) or (self.low < 0 and self.high < 0)

    def as_columns(self, prefix: str) -> dict[str, str]:
        return {
            f"{prefix}_估计": f"{self.point:.6f}",
            f"{prefix}_下限": f"{self.low:.6f}",
            f"{prefix}_上限": f"{self.high:.6f}",
        }


def wmean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * values) / np.sum(weights))


def weighted_ci(
    values: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    draws: int = DRAWS,
) -> Estimate:
    """Weighted mean of ``values`` with a 95% percentile bootstrap over households."""
    n = len(values)
    if n == 0:
        raise ValueError("empty group")
    draw = rng.integers(0, n, size=(draws, n))
    sampled = values[draw]
    sampled_w = weights[draw]
    estimates = np.sum(sampled_w * sampled, axis=1) / np.sum(sampled_w, axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return Estimate(wmean(values, weights), float(low), float(high))


def weighted_ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    draws: int = DRAWS,
) -> tuple[Estimate | None, float, float]:
    """Bootstrap of ratio-of-weighted-means.

    Returns ``(None, num, den)`` when the denominator is near zero, because the
    ratio is then not identifiable and a bound would be meaningless.
    """
    num, den = wmean(numerator, weights), wmean(denominator, weights)
    if abs(den) < NEAR_ZERO:
        return None, num, den
    n = len(numerator)
    draw = rng.integers(0, n, size=(draws, n))
    sampled_w = weights[draw]
    estimates = np.sum(sampled_w * numerator[draw], axis=1) / np.sum(sampled_w * denominator[draw], axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return Estimate(num / den, float(low), float(high)), num, den


def rng_for(*parts: object) -> np.random.Generator:
    """A generator whose stream depends only on the cell it is used for.

    The derivation is a deterministic fold over the part names, so it is stable
    across processes and Python versions -- ``hash()`` is not, because string
    hashing is randomised per interpreter and would make every rerun produce
    different intervals. The fold is also what keeps a cell's interval unchanged
    when another cell is added to the same table.
    """
    digest = 0
    for part in parts:
        folded = sum(map(ord, str(part)))
        digest = (digest * 1_000_003 + folded) % (2**31 - 1)
    return np.random.default_rng(SEED + digest)
