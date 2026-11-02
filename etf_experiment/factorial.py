"""The 162-cell factorial: five factors, fully crossed, on the same 10,000 households.

The fifteen-condition round changes one clause at a time. That design cannot tell
whether effects add up: a fee and a threshold shown together may interact, and with
one-clause-at-a-time data the interaction is unidentifiable. Crossing every level of
five factors (3 fees x 3 thresholds x 3 liquidity x 3 information x 2 rates = 162
cells) makes each factor's marginal effect estimable *averaged over* all
combinations of the others, which is what the factorial table reports.

Two extra arms are scored alongside but are not part of the 162:

- a **length placebo**: the dividend wording padded with a sentence that adds no
  fact, measuring how much probability a merely longer option takes;
- an **ETF-label arm**: D-F rewritten as "mostly deposit plus a small ETF", whose
  combined one-year return (~1.6%) is *below* the money fund's 2.0%, so a rational
  household should not prefer it. Whatever preference it draws is the label.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .conditions import BASE_CRITERIA

FEES = ("none", "open_200", "sub_0.5")
THRESHOLDS = (1_000, 100, 10_000)
LIQUIDITY = ("t1", "redeem_fee", "lock")
INFO = ("none", "drawdown", "upside")
RATES = (1.5, 0.95)

BASELINE_LEVEL = {"fee": "none", "threshold": 1_000, "liquidity": "t1", "info": "none", "rate": 1.5}

LABELS = {
    "none": "免费", "open_200": "开户费200元", "sub_0.5": "申购费0.5%",
    "1000": "门槛1000元", "100": "门槛100元", "10000": "门槛10000元",
    "t1": "T+1免费赎回", "redeem_fee": "30日内赎回费", "lock": "30日锁定",
    "drawdown": "回撤信息", "upside": "三年正收益",
    "1.5": "存款1.5%", "0.95": "存款0.95%",
}

DIVIDEND_WITH_DISTRIBUTION = (
    "红利低波 ETF：一年参考总收益 6.0%（含分红），其中相当于投入本金 4% 的现金分四季派发，"
    "非额外收益；压力年度总回报 −12%"
)
# At least as long as the dividend wording, no added fact: it only restates what the
# option already says. This is what turns "the effect of the dividend sentence" into
# "the effect of a longer sentence".
DIVIDEND_LENGTH_PLACEBO = (
    "红利低波 ETF：一年参考总收益 6.0%，压力年度总回报 −12%。"
    "以上只重述该产品的参考收益和压力回报，到账仍为 T+1，不另设分红、费用或锁定期。"
)

COMBO_D = "存款＋小额红利低波 ETF：组合一年参考收益约 1.57%，给定压力情景约 +1.30%"
COMBO_E = "存款＋小额宽基指数 ETF：组合一年参考收益约 1.60%，给定压力情景约 +1.11%"
COMBO_F = "存款＋小额科技成长 ETF：组合一年参考收益约 1.66%，给定压力情景约 +0.89%"

OPENING = {
    "none": "首次开通基金投资账户免费",
    "open_200": "首次开通基金投资账户需付一次性开户费 200 元（仅选择纯债基金或三只 ETF 时收取）",
    "sub_0.5": "首次开通基金投资账户免费",
}

INFO_CLAUSE = {
    "none": "",
    "drawdown": "情景给定的最大回撤：纯债基金 4%、红利低波 ETF 30%、宽基指数 ETF 45%、科技成长 ETF 65%。",
    "upside": "情景给定的三年持有正收益比例：纯债基金 98%、红利低波 ETF 85%、宽基指数 ETF 78%、科技成长 ETF 72%。",
}


@dataclass(frozen=True)
class Cell:
    kind: str          # "factorial", "placebo" or "label"
    fee: str
    threshold: int
    liquidity: str
    info: str
    rate: float

    def as_record(self) -> dict:
        return {
            "kind": self.kind, "fee": self.fee, "threshold": self.threshold,
            "liquidity": self.liquidity, "info": self.info, "rate": self.rate,
        }


def cells() -> list[Cell]:
    """The 162 crossed cells plus the two extra arms, in a stable order."""
    rows = [
        Cell("factorial", fee, threshold, liquidity, info, rate)
        for fee, threshold, liquidity, info, rate in product(FEES, THRESHOLDS, LIQUIDITY, INFO, RATES)
    ]
    rows.append(Cell("placebo", "none", 1_000, "t1", "placebo", 1.5))
    rows.append(Cell("label", "none", 1_000, "t1", "label", 1.5))
    return rows


def price_clause(fee: str, liquidity: str) -> str:
    """The fee-and-redemption sentence for one combination.

    Written out rather than composed: the two conditions share a single sentence and
    the models saw specific interleavings, so generating it from flags would risk
    drifting from the text that was actually scored.
    """
    if fee == "sub_0.5":
        if liquidity == "t1":
            return "申购费为净申购金额的 0.5%，赎回费率为 0，赎回后 T+1 到账。"
        if liquidity == "redeem_fee":
            return (
                "申购费为净申购金额的 0.5%；持有不足 30 日赎回按赎回金额收取 0.5%，"
                "满 30 日后赎回费率为 0，赎回后 T+1 到账。"
            )
        return (
            "申购费为净申购金额的 0.5%；持有不足 30 日不可赎回，"
            "满 30 日后可赎回且赎回费率为 0，赎回后 T+1 到账。"
        )
    if liquidity == "t1":
        return "申购赎回费率为 0，赎回后 T+1 到账。"
    if liquidity == "redeem_fee":
        return (
            "申购费率为 0；持有不足 30 日赎回按赎回金额收取 0.5%，"
            "满 30 日后赎回费率为 0，赎回后 T+1 到账。"
        )
    return "申购费率为 0；持有不足 30 日不可赎回，满 30 日后可赎回且赎回费率为 0，赎回后 T+1 到账。"


def clause(cell: Cell) -> str:
    return (
        "纯债基金、红利低波 ETF、宽基指数 ETF、科技成长 ETF："
        f"净申购金额至少 {cell.threshold:,} 元，{price_clause(cell.fee, cell.liquidity)}"
        f"尚未开通基金或证券账户，{OPENING[cell.fee]}，无需证券账户。{INFO_CLAUSE[cell.info]}"
        "费用从这笔资金内扣；选定后余额全部投入该安排。"
        "基金与 ETF 的回报为一年含分红总回报的情景设定，非保证收益；存款按所列到期本息条款执行。"
    )


def label_clause(x: str, deposit: str, etf_leg: str) -> str:
    return (
        "纯债基金仍为 1,000 元起购、申购赎回费率 0、T+1 到账。"
        f"D–F 改为组合：现在将 {deposit} 元存为一年期存款（年利率 1.5%，到期保本保息），"
        f"其余 {etf_leg} 元立即买入对应 ETF；"
        "ETF 允许 1 元起购、零申赎费、T+1 到账。存款部分持有到期，ETF 子账户不使用杠杆，"
        f"最多损失投入的 {etf_leg} 元；组合收益按全部 {x} 元计算。"
        "尚未开通基金或证券账户，首次开通基金投资账户免费，无需证券账户。"
        "费用从这笔资金内扣；选定后余额全部投入该安排。"
        "基金与 ETF 的回报为一年含分红总回报的情景设定，非保证收益；存款按所列到期本息条款执行。"
    )


def rate_text(rate: float) -> str:
    return "1.5" if rate == 1.5 else "0.95"


def menu(cell: Cell) -> dict[str, str]:
    """The six options for this cell, in A–F order."""
    criteria = dict(BASE_CRITERIA)
    criteria["A"] = f"一年期银行存款：年利率 {rate_text(cell.rate)}%，持有到期保本保息"
    if cell.kind == "placebo":
        criteria["D"] = DIVIDEND_LENGTH_PLACEBO
    elif cell.kind == "label":
        criteria["D"], criteria["E"], criteria["F"] = COMBO_D, COMBO_E, COMBO_F
    return criteria


def clause_for(cell: Cell, x: str, deposit: str, etf_leg: str) -> str:
    if cell.kind == "label":
        return label_clause(x, deposit, etf_leg)
    if cell.kind == "placebo":
        return clause(Cell("factorial", "none", 1_000, "t1", "none", 1.5))
    return clause(cell)


def instructions(x: str) -> str:
    return f"以该家庭身份，为题目给定的 {x} 元闲置资金选择一种安排，仅选一项。"


def self_check() -> None:
    """The length placebo only measures length if it is not shorter than the original."""
    if len(DIVIDEND_LENGTH_PLACEBO) < len(DIVIDEND_WITH_DISTRIBUTION) - 4:
        raise ValueError(
            "length placebo is shorter than the dividend text: "
            f"{len(DIVIDEND_LENGTH_PLACEBO)} vs {len(DIVIDEND_WITH_DISTRIBUTION)}"
        )
    if len(cells()) != 164:
        raise ValueError(f"expected 162 cells plus 2 extra arms, got {len(cells())}")


def treatment_matrix(cell: Cell) -> list[list[float]]:
    """Which of the six alternatives each of nine treatments applies to.

    Nine columns in this order: opening fee, subscription fee, threshold 100,
    threshold 10,000, redemption fee, lock-up, drawdown info, upside info, deposit
    rate. The first eight attach to products C-F only; the rate attaches to the
    deposit only.
    """
    affected = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    deposit = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    product_flags = [
        cell.fee == "open_200",
        cell.fee == "sub_0.5",
        cell.threshold == 100,
        cell.threshold == 10_000,
        cell.liquidity == "redeem_fee",
        cell.liquidity == "lock",
        cell.info == "drawdown",
        cell.info == "upside",
    ]
    rows = []
    for position in range(6):
        row = [affected[position] * float(flag) for flag in product_flags]
        row.append(deposit[position] * float(cell.rate == 0.95))
        rows.append(row)
    return rows
