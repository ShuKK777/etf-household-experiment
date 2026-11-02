"""Outcome metrics and how the six product probabilities map onto them.

Every published number is one of these metrics or a difference of them, so the
definitions live in one place: if "equity ETF" ever means something other than
D+E+F, both the baseline table and the clause table must change together or they
stop being comparable.

The raw six-way array is kept intact. Aggregating at load time would make the
per-product columns of table 1 unrecoverable, which is exactly the bug that
makes two tables in the same report quietly disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .options import KEYS, investable

EQUITY_KEYS = ("D", "E", "F")
BOND_ETF_KEYS = ("C", "D", "E", "F")


@dataclass(frozen=True)
class Probabilities:
    """Six product probabilities, shaped ``(households, conditions, 6)``."""

    array: np.ndarray

    def __post_init__(self) -> None:
        if self.array.ndim != 3 or self.array.shape[2] != len(KEYS):
            raise ValueError(f"expected (households, conditions, 6), got {self.array.shape}")

    def __getitem__(self, key: str) -> np.ndarray:
        return self.array[:, :, KEYS.index(key)]

    def sum_of(self, keys: tuple[str, ...]) -> np.ndarray:
        return np.sum(self.array[:, :, [KEYS.index(key) for key in keys]], axis=2)

    @property
    def deposit(self) -> np.ndarray:
        return self["A"]

    @property
    def money_fund(self) -> np.ndarray:
        return self["B"]

    @property
    def bond(self) -> np.ndarray:
        return self["C"]

    @property
    def equity(self) -> np.ndarray:
        return self.sum_of(EQUITY_KEYS)

    @property
    def bond_or_equity(self) -> np.ndarray:
        return self.sum_of(BOND_ETF_KEYS)

    @property
    def leave_deposit(self) -> np.ndarray:
        return 1.0 - self.deposit

    def metrics(self, capacity: np.ndarray) -> dict[str, np.ndarray]:
        """The six series the tables report, each ``(households, conditions)``."""
        return {
            "离开存款": self.leave_deposit,
            "货基": self.money_fund,
            "纯债": self.bond,
            "纯债或ETF": self.bond_or_equity,
            "权益ETF": self.equity,
            "ETF金额": capacity * self.equity,
        }

    def products(self) -> dict[str, np.ndarray]:
        return {key: self[key] for key in KEYS}


def capacity_matrix(x: np.ndarray, etf_leg: np.ndarray | None = None) -> np.ndarray:
    """Executable yuan per condition, shaped ``(households, 15)``.

    A household that cannot meet a condition's minimum contributes zero, keeping
    the "purchase not completed" mass visible instead of folding it into a
    feasible-looking amount.
    """
    return np.vstack([investable(condition, x, etf_leg) for condition in range(1, 16)]).T


def blocked_by_threshold(x: np.ndarray) -> np.ndarray:
    """Households shut out of the 10,000-yuan minimum.

    On the new schedule both b1 and b2 fall below it; on the old one only b1 did,
    because b2's 15,000 yuan already cleared the line.
    """
    return x + 1e-9 < 10_000
