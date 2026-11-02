"""Building the two arms of the #6 sentence placebo.

The two-tick round appends one sentence to the prompt. That sentence carries three
things at once: content (the order failed), length (a sentence was added) and
repetition (your choice was restated). This module separates content from the other
two by constructing a control that is length- and structure-matched.

The length and vocabulary constraints are asserted, not assumed. If someone edits
either sentence so that it stops being comparable, ``build`` raises rather than
producing an arm that quietly tests something else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEED = 20261026
BRACKET_COUNTS = {"b1": 3_000, "b2": 2_600}
AMOUNTS = {"b1": "1,500", "b2": "9,100"}

FAILURE = (
    "上一轮选择未成交：所选产品净申购金额至少 10,000 元，现有 {amount} 元不足，"
    "资金未动，目前没有持有该产品。"
)
# Same character count as FAILURE, no content word from the failure arm, and the
# same "上一轮选择已…" opening so that restating the choice is held constant.
CONTROL = (
    "上一轮选择已保存：系统已记下本次勾选，页面维持原来样子，记录没有其他改动，"
    "请继续查看当前页面即可，本次记录只作留存。"
)

BANNED = ("未成交", "没买成", "门槛", "10,000", "10000", "不足", "起购", "资金", "余额", "持有", "产品", "元")


@dataclass(frozen=True)
class Arm:
    index: int
    bracket: str
    kind: str          # "failure" or "placebo"
    sentence: str


def _self_check() -> None:
    failure = FAILURE.format(amount=AMOUNTS["b1"])
    longer = FAILURE.format(amount=AMOUNTS["b2"])
    if not (len(failure) == len(longer) == len(CONTROL)):
        raise ValueError(
            f"arms are not length-matched: failure {len(failure)}, "
            f"failure with 9,100 {len(longer)}, control {len(CONTROL)}"
        )
    if any(word in CONTROL for word in BANNED):
        raise ValueError("the control contains a word that only the failure arm should carry")
    if not CONTROL.startswith("上一轮选择已") or not failure.startswith("上一轮选择"):
        raise ValueError("arms no longer share the restatement opening")


def build(indexes_by_bracket: dict[str, list[int]], seed: int = SEED) -> list[Arm]:
    """Split each bracket in half with a fixed seed and tag every household.

    Splitting within bracket rather than across the panel keeps the two arms
    balanced on idle cash, which is the dimension that drives the whole #6 result.
    """
    _self_check()
    rng = np.random.default_rng(seed)
    arms: list[Arm] = []
    for bracket, indexes in indexes_by_bracket.items():
        expected = BRACKET_COUNTS[bracket]
        unique = sorted(set(indexes))
        if len(unique) != expected:
            raise ValueError(f"{bracket} has {len(unique)} households, expected {expected}")
        order = rng.permutation(np.array(unique, dtype=int))
        half = len(order) // 2
        for position, index in enumerate(order):
            kind = "failure" if position < half else "placebo"
            sentence = FAILURE.format(amount=AMOUNTS[bracket]) if kind == "failure" else CONTROL
            arms.append(Arm(index=int(index), bracket=bracket, kind=kind, sentence=sentence))
    if len(arms) != sum(BRACKET_COUNTS.values()):
        raise ValueError(f"expected {sum(BRACKET_COUNTS.values())} households, got {len(arms)}")
    return arms


def arms_from_tasks(task_rows: list[dict], seed: int = SEED) -> list[Arm]:
    """Collect the b1/b2 households of condition 6 and split them."""
    by_bracket: dict[str, list[int]] = {"b1": [], "b2": []}
    for row in task_rows:
        bracket = row["meta"]["asset_bracket"]
        if bracket in by_bracket:
            by_bracket[bracket].append(int(row["meta"]["sample_index"]))
    return build(by_bracket, seed=seed)
