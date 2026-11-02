"""The fifteen product conditions.

Condition 1 is the baseline; every other condition changes exactly one thing and
is compared against it. Conditions 12 and 13 differ only by one added sentence
about inflation, which is why the report always reports both #13−#1 (the joint
effect) and #13−#12 (the inflation sentence alone).

The terms are written as whole sentences rather than assembled from flags. That
is intentional: the experiment's finding is that wording moves probabilities as
much as economics does, so the exact strings are part of the treatment and must
not be generated in a way that silently drifts.
"""

from __future__ import annotations

from dataclasses import dataclass

from .options import Menu, describe_amount, describe_cash, split_deposit_etf

# ---------------------------------------------------------------- product text
DEPOSIT_15 = "一年期银行存款：年利率 1.5%，持有到期保本保息"
DEPOSIT_095 = "一年期银行存款：年利率 0.95%，持有到期保本保息"
MONEY_FUND = "货币基金：一年参考总收益 2.0%，低风险不保本，1 元起购、随取随用，无需新开基金账户"
BOND = "纯债基金：一年参考总收益 3.5%，压力年度总回报 −1.5%"
DIVIDEND = "红利低波 ETF：一年参考总收益 6.0%，压力年度总回报 −12%"
BROAD = "宽基指数 ETF：一年参考总收益 8.0%，压力年度总回报 −25%"
TECH = "科技成长 ETF：一年参考总收益 12.0%，压力年度总回报 −40%"

DIVIDEND_QUARTERLY = (
    "红利低波 ETF：一年参考总收益 6.0%（含分红），"
    "其中相当于投入本金 4% 的现金分四季派发，非额外收益；压力年度总回报 −12%"
)
COMBO_DIVIDEND = "存款＋小额红利低波 ETF：组合一年参考收益约 1.57%，给定压力情景约 +1.30%"
COMBO_BROAD = "存款＋小额宽基指数 ETF：组合一年参考收益约 1.60%，给定压力情景约 +1.11%"
COMBO_TECH = "存款＋小额科技成长 ETF：组合一年参考收益约 1.66%，给定压力情景约 +0.89%"

BASE_CRITERIA = {
    "A": DEPOSIT_15, "B": MONEY_FUND, "C": BOND,
    "D": DIVIDEND, "E": BROAD, "F": TECH,
}

# ---------------------------------------------------------------- clause text
TAIL = (
    "费用从这笔资金内扣；选定后余额全部投入该安排。"
    "基金与 ETF 的回报为一年含分红总回报的情景设定，非保证收益；"
    "存款按所列到期本息条款执行。所有条款均为实验设定。"
)
OPEN_FREE = "实验设定：尚未开通基金或证券账户，首次开通基金投资账户免费，无需证券账户。"
TERMS_1000 = (
    "纯债基金、红利低波 ETF、宽基指数 ETF、科技成长 ETF："
    "净申购金额至少 1,000 元，申购赎回费率为 0，赎回后 T+1 到账。"
)
FUNDS = "纯债基金、红利低波 ETF、宽基指数 ETF、科技成长 ETF"

EVENT = "家中现有闲置资金 {x} 元。这笔钱可自主安排，以下方式都能在手机上办理。"
EVENT_RESERVED = (
    "家中现有闲置资金 {x} 元。未来 3 个月必要开支与到期还款已另行预留。"
    "这笔钱可自主安排，以下方式都能在手机上办理。"
)

# ---------------------------------------------------------------- conditions
NAMES = {
    1: "基准",
    2: "三年正收益比例",
    3: "开户费 200 元",
    4: "申购费 0.5%",
    5: "起购 100 元",
    6: "起购 10,000 元",
    7: "30 日内赎回费",
    8: "30 日锁定期",
    9: "告知生活费已预留",
    10: "回撤信息",
    11: "按季分红",
    12: "存款降息",
    13: "降息＋通胀",
    14: "资金托管",
    15: "大部分存款＋小部分 ETF",
}
CONDITIONS = tuple(range(1, 16))


@dataclass(frozen=True)
class Condition:
    number: int
    name: str
    event: str
    clause: str
    menu: Menu

    @property
    def criteria(self) -> dict[str, str]:
        return self.menu.criteria


def _menu(**overrides: str) -> Menu:
    criteria = dict(BASE_CRITERIA)
    criteria.update(overrides)
    return Menu(criteria=criteria)


def build(number: int, x: float) -> Condition:
    """The state text and menu for one condition at one idle-cash amount.

    ``x`` only matters for condition 15, where the deposit and ETF legs are
    written into the clause as concrete amounts.
    """
    if number not in NAMES:
        raise ValueError(f"condition {number} does not exist")
    event = EVENT.format(x=describe_cash(x))
    menu = _menu()
    clause = f"{TERMS_1000}{OPEN_FREE}{TAIL}"

    if number == 1:
        pass
    elif number == 2:
        clause = (
            f"{TERMS_1000}{OPEN_FREE}"
            "情景给定的三年持有正收益比例：纯债基金 98%、红利低波 ETF 85%、"
            "宽基指数 ETF 78%、科技成长 ETF 72%；"
            f"这不是实测历史统计。{TAIL}"
        )
    elif number == 3:
        clause = (
            f"{FUNDS}：净申购金额至少 1,000 元，申购赎回费率为 0，赎回后 T+1 到账。"
            "实验设定：尚未开通基金或证券账户，首次开通基金投资账户需付一次性开户费 200 元"
            f"（仅选择纯债基金或三只 ETF 时收取），无需证券账户。{TAIL}"
        )
        menu = Menu(criteria=dict(BASE_CRITERIA), open_fee=200)
    elif number == 4:
        clause = (
            f"{FUNDS}：净申购金额至少 1,000 元，"
            f"申购费为净申购金额的 0.5%，赎回费率为 0，赎回后 T+1 到账。{OPEN_FREE}{TAIL}"
        )
        menu = Menu(criteria=dict(BASE_CRITERIA), subscription=0.005)
    elif number == 5:
        clause = (
            f"{FUNDS}：净申购金额至少 100 元，"
            f"申购赎回费率为 0，赎回后 T+1 到账。{OPEN_FREE}{TAIL}"
        )
        menu = Menu(criteria=dict(BASE_CRITERIA), threshold=100)
    elif number == 6:
        clause = (
            f"{FUNDS}：净申购金额至少 10,000 元，"
            f"申购赎回费率为 0，赎回后 T+1 到账。{OPEN_FREE}{TAIL}"
        )
        menu = Menu(criteria=dict(BASE_CRITERIA), threshold=10_000)
    elif number == 7:
        clause = (
            f"{FUNDS}：净申购金额至少 1,000 元，申购费率为 0；"
            "持有不足 30 日赎回按赎回金额收取 0.5%，满 30 日后赎回费率为 0，赎回后 T+1 到账。"
            f"{OPEN_FREE}{TAIL}"
        )
    elif number == 8:
        clause = (
            f"{FUNDS}：净申购金额至少 1,000 元，申购费率为 0；"
            "持有不足 30 日不可赎回，满 30 日后可赎回且赎回费率为 0，赎回后 T+1 到账。"
            f"{OPEN_FREE}{TAIL}"
        )
    elif number == 9:
        event = EVENT_RESERVED.format(x=describe_cash(x))
    elif number == 10:
        clause = (
            f"{TERMS_1000}{OPEN_FREE}"
            "情景给定的最大回撤：纯债基金 4%、红利低波 ETF 30%、"
            "宽基指数 ETF 45%、科技成长 ETF 65%；"
            f"这不是实测历史统计。{TAIL}"
        )
    elif number == 11:
        menu = _menu(D=DIVIDEND_QUARTERLY)
    elif number == 12:
        menu = _menu(A=DEPOSIT_095)
    elif number == 13:
        clause = f"{TERMS_1000}{OPEN_FREE}{TAIL}另给定同期物价年涨幅为 2.5%。"
        menu = _menu(A=DEPOSIT_095)
    elif number == 14:
        clause = (
            f"{TERMS_1000}{OPEN_FREE}"
            "纯债基金和三只 ETF 的资产由独立托管机构保管；"
            f"托管不保证投资收益，也不承担市场亏损。{TAIL}"
        )
    elif number == 15:
        deposit, etf_leg = split_deposit_etf(x)
        clause = (
            "纯债基金仍按基准：1,000 元起购、申购赎回费率 0、T+1 到账。"
            f"D–F 改为组合产品：现在将 {describe_amount(deposit)} 元存为一年期存款"
            "（年利率 1.5%，到期保本保息），"
            f"其余 {describe_amount(etf_leg)} 元立即买入对应 ETF；"
            "ETF 允许 1 元起购、零申赎费、T+1 到账。"
            f"存款部分持有到期，ETF 子账户不使用杠杆，最多损失投入的 {describe_amount(etf_leg)} 元；"
            f"以下组合收益均按全部 {describe_cash(x)} 元计算。{OPEN_FREE}{TAIL}"
        )
        menu = _menu(D=COMBO_DIVIDEND, E=COMBO_BROAD, F=COMBO_TECH)

    return Condition(number=number, name=NAMES[number], event=event, clause=clause, menu=menu)


def instruction(x: float) -> str:
    return f"以该家庭身份，为题目给定的 {describe_cash(x)} 元闲置资金选择一种安排，仅选一项。"
