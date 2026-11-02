"""Subgroup labels: the coded level names and their Chinese presentation.

Table 3 cuts the panel seven ways. The codes are the stored values; the labels
are what the report prints. Keeping them adjacent to the field names prevents the
classic failure where a subgroup silently disappears because a code was renamed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ASSET = {
    "b1": "b1 不足 1 万元", "b2": "b2 1 万至不足 5 万元", "b3": "b3 5 万至不足 20 万元",
    "b4": "b4 20 万至不足 50 万元", "b5": "b5 50 万元及以上",
}
DEBT_AMOUNT = {
    "none": "无债务", "unknown": "未填写",
    "d1": "d1 0 至 1 万元", "d2": "d2 1 万至 2 万元", "d3": "d3 2 万至 5 万元",
    "d4": "d4 5 万至 10 万元", "d5": "d5 10 万至 20 万元", "d6": "d6 20 万至 30 万元",
    "d7": "d7 30 万至 50 万元", "d8": "d8 50 万至 100 万元", "d9": "d9 100 万至 200 万元",
    "d10": "d10 200 万至 500 万元", "d11": "d11 500 万元以上",
}
DEBT_TYPE = {
    "none": "无债务", "unknown": "未填写", "housing": "住房",
    "education": "教育", "medical": "医疗", "consumer_credit": "消费信贷",
}
CASH_FLOW = {"surplus": "有结余", "balance": "收支相当", "deficit": "收入不够支出", "unknown": "未填写"}
RISK = {
    "high": "高风险", "slightly_high": "略高", "average": "平均",
    "slightly_low": "略低", "none": "不愿承担任何风险", "unknown": "未填写",
}
LITERACY = {
    "neither_correct": "两题全错", "interest_only": "仅利息题正确",
    "inflation_only": "仅通胀题正确", "both_correct": "全对", "unknown": "未知",
}
TRUST = {"trust": "通常信任他人", "depends": "视情况而定", "distrust": "通常不信任他人", "unknown": "未填写"}

# field name -> (stored column in the panel, printed labels)
DIMENSIONS = {
    "asset": ("asset_bracket", ASSET, "资产档"),
    "debt_amount": ("extended.debt_amount_bracket", DEBT_AMOUNT, "债务金额档"),
    "debt_type": ("extended.debt_type", DEBT_TYPE, "债务类型"),
    "cash_flow": ("extended.income_expenditure_status", CASH_FLOW, "收支"),
    "risk": ("extended.risk_preference", RISK, "风险偏好"),
    "literacy": ("extended.financial_literacy", LITERACY, "金融知识"),
    "trust": ("extended.general_trust", TRUST, "信任"),
}

# Conditions worth cutting by each dimension, chosen to answer the plan's
# section 4 questions rather than to maximise the number of rows.
FOCUS: dict[str, tuple[str, ...]] = {
    "asset": ("#3−#1", "#4−#1", "#6−#1"),
    "debt_amount": ("#7−#1", "#8−#1", "#9−#1"),
    "debt_type": ("#7−#1", "#8−#1", "#9−#1"),
    "cash_flow": ("#7−#1", "#8−#1"),
    "risk": ("#2−#1", "#10−#1", "#11−#1"),
    "literacy": ("#2−#1", "#10−#1", "#11−#1"),
    "trust": ("#14−#1",),
}


def label(field: str, level: str) -> str:
    if field == "debt_type" and "|" in level:
        return "、".join(DEBT_TYPE.get(part, part) for part in level.split("|"))
    _, labels, _ = DIMENSIONS[field]
    return labels.get(level, level)


@dataclass(frozen=True)
class Groups:
    """The panel cut seven ways, in the same household order as the outcomes."""

    arrays: dict[str, np.ndarray]

    def __getitem__(self, field: str) -> np.ndarray:
        return self.arrays[field]

    def levels(self, field: str) -> list[str]:
        """Coded levels in label order, then any composite level not in the table.

        Composite debt types such as ``housing|consumer_credit`` are real stored
        values but have no label of their own, so they are appended after the
        single types rather than dropped.
        """
        _, labels, _ = DIMENSIONS[field]
        known = [level for level in labels if (self.arrays[field] == level).any()]
        extra = sorted(set(self.arrays[field]) - set(labels))
        return known + [level for level in extra if level != ""]

    def mask(self, field: str, level: str) -> np.ndarray:
        return self.arrays[field] == level


def from_records(records: list[dict], *, asset_from_idle: bool = False) -> Groups:
    """Build the seven cuts from panel rows.

    ``asset_from_idle`` recomputes the asset bracket from the idle-cash schedule
    when the stored bracket is the pre-calibration one; the new schedule uses the
    bracket recorded at sampling.
    """
    arrays: dict[str, list[str]] = {field: [] for field in DIMENSIONS}
    for record in records:
        stored = record["profile"]
        arrays["asset"].append(record["asset_bracket"])
        for field, (column, _, _) in DIMENSIONS.items():
            if field == "asset":
                continue
            arrays[field].append(stored[column])
    return Groups({field: np.array(values) for field, values in arrays.items()})
