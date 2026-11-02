"""The A–F product menu, the idle-cash amounts, and the fee arithmetic.

Money handling is deliberately explicit. The experiment's key claim is that a
1,000-yuan threshold binds some households and not others, and that a 0.5% fee
scales with the amount, so every rule that changes the *investable* amount lives
here, in one place, with the rounding mode written out.

All amounts are compared in yuan with a 1e-9 tolerance and reported to the cent,
matching the prompts the models actually saw (``{X:,.2f}``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KEYS = ("A", "B", "C", "D", "E", "F")

# Idle cash by financial-asset bracket, from the CFPS 2018 calibration.
# The old schedule is kept because the first two rounds ran on it and the
# results are compared across the two in the report.
IDLE_X = {
    "new": {"b1": 1_500.0, "b2": 9_100.0, "b3": 30_000.0, "b4": 53_000.0, "b5": 88_000.0},
    "old": {"b1": 2_500.0, "b2": 15_000.0, "b3": 62_500.0, "b4": 175_000.0, "b5": 312_500.0},
}

BASE_THRESHOLD = 1_000
OPEN_FEE = 200
SUBSCRIPTION_RATE = 0.005
DEPOSIT_RATE = 0.015


@dataclass(frozen=True)
class Menu:
    """One product menu with the terms that condition i applies to it."""

    criteria: dict[str, str]
    threshold: int = BASE_THRESHOLD
    open_fee: int = 0
    subscription: float = 0.0


def idle_amounts(schedule: str, brackets: np.ndarray) -> np.ndarray:
    table = IDLE_X[schedule]
    return np.array([table[str(bracket)] for bracket in brackets], dtype=float)


def round_half_up(value: float, decimals: int = 2) -> float:
    """Round half away from zero, as the plan specifies for fee amounts."""
    factor = 10**decimals
    return float(np.floor(abs(value) * factor + 0.5) / factor * np.sign(value))


def net_after_subscription(x: float, rate: float = SUBSCRIPTION_RATE) -> float:
    """Amount actually invested when the fee is charged on the gross amount."""
    return round_half_up(x / (1 + rate), 2)


def subscription_fee(x: float, rate: float = SUBSCRIPTION_RATE) -> float:
    return round_half_up(x - net_after_subscription(x, rate), 2)


def split_deposit_etf(x: float, rate: float = DEPOSIT_RATE) -> tuple[float, float]:
    """Condition 15: deposit P grows to exactly X, the rest buys the ETF now.

    ``P = ceil(X / (1+rate))`` to the cent, then ``I = X - P``. The ceiling makes
    the deposit leg cover X on its own, so the portfolio cannot fall below X
    unless the ETF leg loses value.
    """
    p = float(np.ceil(round(x / (1 + rate) * 100, 6)) / 100)
    return round(p, 2), round(x - p, 2)


def investable(condition: int, x: np.ndarray, etf_leg: np.ndarray | None = None) -> np.ndarray:
    """Yuan that can actually be placed after this condition's fees and threshold.

    Zero means the household would pick a product it cannot buy. The convention
    in the plan is to keep the raw probability and record zero executable amount,
    rather than renormalising across the feasible options.
    """
    if condition == 15 and etf_leg is not None:
        return etf_leg.copy()
    amount = x.astype(float).copy()
    if condition == 3:
        amount = amount - OPEN_FEE
    elif condition == 4:
        amount = np.array([net_after_subscription(v) for v in amount])
    threshold = {5: 100, 6: 10_000}.get(condition, BASE_THRESHOLD)
    return np.where(amount + 1e-9 >= threshold, amount, 0.0)


def blocked_mask(condition: int, x: np.ndarray, schedule: str = "new") -> np.ndarray:
    """Households whose idle cash cannot reach this condition's minimum."""
    threshold = {5: 100, 6: 10_000}.get(condition, BASE_THRESHOLD)
    return x + 1e-9 < threshold


def describe_amount(value: float, decimals: int = 2) -> str:
    """Thousands-separated amount at a fixed number of decimals.

    The idle-cash X is always rendered with zero decimals (it is a whole number
    on both schedules), while the deposit/ETF legs of condition 15 and all fee
    amounts use two. This distinction is load-bearing: it is what the models
    actually saw, so the round half-up helpers above are also kept separate from
    the formatting here.
    """
    return f"{value:,.{decimals}f}"


def describe_cash(value: float) -> str:
    """The idle-cash X as rendered in every prompt: whole yuan, no decimals."""
    return describe_amount(value, 0)
