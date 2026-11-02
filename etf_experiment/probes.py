"""Text probes: how much of an observed effect is the wording.

The fifteen-condition round changes the economics of a clause. But the same clause
can be written in more than one way, and this round measures how far those ways move
the answer. That is not a side issue -- if a paraphrase worth 4.7 percentage points
sits next to a fee worth 1.4, then a claim about the fee is unsupported unless the
wording is held fixed.

Two families of probe:

**Wording and position** (scored by the encoder, 10,000 households each):
  - ``baseline_ref``   the baseline clause, re-rendered to confirm the harness
  - ``bond_rename``    纯债基金 -> 债券型基金, a pure vocabulary swap
  - ``etf_as_index``   ETF -> 指数基金
  - ``etf_first``      move the three ETFs ahead of the bond fund in the clause
  - ``rewrite``        same facts, different sentences: the paraphrasing benchmark

**Position** (``position_01`` .. ``position_15``): the same question scored twice,
once in fixed A–F order and once in each household's own randomised order. The
difference is what option order alone is worth, and it doubles as the check that the
engine's batching did not permute probabilities.

A third family, the **teacher contrast**, asks a much larger model for one letter on
1,000 households and ten clauses, to see whether the directions a small decision model
gives are recoverable by a model that was never fine-tuned on this task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conditions import BASE_CRITERIA

POSITION_INDICES = tuple(range(1, 16))
WORDING_NAMES = ("baseline_ref", "bond_rename", "etf_first", "etf_as_index", "rewrite")

BOND_AS_BOND_FUND = "债券型基金"


@dataclass(frozen=True)
class Probe:
    name: str
    clause: str
    criteria: dict[str, str]


def _baseline_clause() -> str:
    from .conditions import build

    return build(1, 30_000.0).clause


def baseline() -> Probe:
    return Probe("baseline_ref", _baseline_clause(), dict(BASE_CRITERIA))


def bond_rename() -> Probe:
    """Swap 纯债基金 for 债券型基金 in the clause and in option C."""
    clause = _baseline_clause().replace("纯债基金", BOND_AS_BOND_FUND)
    criteria = dict(BASE_CRITERIA)
    criteria["C"] = criteria["C"].replace("纯债基金", BOND_AS_BOND_FUND)
    return Probe("bond_rename", clause, criteria)


def etf_first() -> Probe:
    """List the three ETFs before the bond fund. Fact set unchanged."""
    clause = _baseline_clause().replace(
        "纯债基金、红利低波 ETF、宽基指数 ETF、科技成长 ETF",
        "红利低波 ETF、宽基指数 ETF、科技成长 ETF、纯债基金",
    )
    return Probe("etf_first", clause, dict(BASE_CRITERIA))


def etf_as_index() -> Probe:
    """ETF -> 指数基金, in the clause and in options D-F."""
    clause = _baseline_clause().replace("ETF", "指数基金")
    criteria = {
        key: value.replace("ETF", "指数基金") if key in "DEF" else value
        for key, value in BASE_CRITERIA.items()
    }
    return Probe("etf_as_index", clause, criteria)


def rewrite() -> Probe:
    """The same facts said differently. Deliberately verbose, deliberately informal.

    This is the benchmark that sets the noise floor: no fact changes, so whatever it
    moves is attributable to phrasing alone.
    """
    clause = (
        "纯债基金、红利低波 ETF、宽基指数 ETF 和科技成长 ETF 的申购起点都是 1,000 元，"
        "申购与赎回均不另收费，赎回资金在 T+1 到账。"
        "该户还没有基金账户，也没有证券账户，这次开立基金账户不收钱。"
        "如果产生费用，从这笔闲置资金里面扣；选定之后，剩余金额全部买入所选产品。"
        "基金和 ETF 写出的是含分红的一年情景收益，并不承诺兑付；存款则按到期还本付息办理。"
    )
    criteria = {
        "A": "一年期银行存款：年利率 1.5%，存满一年后取回本金和利息",
        "B": "货币基金：按情景一年总收益 2.0%，风险低且不保证本金，1 元即可买，需要时可取，不必新开基金账户",
        "C": "纯债基金：按情景一年总收益 3.5%，压力情形下一年总回报 −1.5%",
        "D": "红利低波 ETF：按情景一年总收益 6.0%，压力情形下一年总回报 −12%",
        "E": "宽基指数 ETF：按情景一年总收益 8.0%，压力情形下一年总回报 −25%",
        "F": "科技成长 ETF：按情景一年总收益 12.0%，压力情形下一年总回报 −40%",
    }
    return Probe("rewrite", clause, criteria)


BUILDERS = {
    "baseline_ref": baseline,
    "bond_rename": bond_rename,
    "etf_first": etf_first,
    "etf_as_index": etf_as_index,
    "rewrite": rewrite,
}


def build(name: str) -> Probe:
    if name not in BUILDERS:
        raise KeyError(f"unknown wording probe {name}")
    return BUILDERS[name]()


def all_names() -> list[str]:
    return list(POSITION_INDICES and ()) + list(WORDING_NAMES)


def task_names() -> list[str]:
    """Every probe to run: fifteen position probes plus five wording probes."""
    return [f"position_{index:02d}" for index in POSITION_INDICES] + list(WORDING_NAMES)


def fixed_order(criteria: dict[str, str]) -> dict[str, str]:
    """The A–F presentation, as opposed to a household's randomised order."""
    return {key: criteria[key] for key in "ABCDEF"}


def reorder(criteria: dict[str, str], order: tuple[str, ...]) -> dict[str, str]:
    return {key: criteria[key] for key in order}


@dataclass(frozen=True)
class TeacherCondition:
    name: str
    cell: object   # a factorial.Cell


def teacher_conditions() -> list[TeacherCondition]:
    """Ten cells: the baseline plus nine single-clause deviations."""
    from .factorial import Cell

    specs = [
        ("baseline", "none", 1_000, "t1", "none", 1.5),
        ("open_200", "open_200", 1_000, "t1", "none", 1.5),
        ("sub_0.5", "sub_0.5", 1_000, "t1", "none", 1.5),
        ("thr_100", "none", 100, "t1", "none", 1.5),
        ("thr_10000", "none", 10_000, "t1", "none", 1.5),
        ("redeem", "none", 1_000, "redeem_fee", "none", 1.5),
        ("lock", "none", 1_000, "lock", "none", 1.5),
        ("drawdown", "none", 1_000, "t1", "drawdown", 1.5),
        ("upside", "none", 1_000, "t1", "upside", 1.5),
        ("rate_095", "none", 1_000, "t1", "none", 0.95),
    ]
    return [TeacherCondition(name, Cell("factorial", fee, threshold, liquidity, info, rate))
            for name, fee, threshold, liquidity, info, rate in specs]


TEACHER_QUOTA = {"b1": 300, "b2": 260, "b3": 240, "b4": 120, "b5": 80}


def teacher_sample(asset_brackets: np.ndarray, seed: int = 20261021) -> np.ndarray:
    """1,000 households, 300/260/240/120/80 by bracket, same seed as the main panel.

    The quota keeps the small high-asset brackets represented; a proportional draw
    would leave b5 with roughly 28 households.
    """
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for bracket, count in TEACHER_QUOTA.items():
        pool = np.flatnonzero(asset_brackets == bracket)
        if len(pool) < count:
            raise ValueError(f"{bracket} has only {len(pool)} households, need {count}")
        chosen.extend(sorted(int(pool[i]) for i in rng.choice(len(pool), size=count, replace=False)))
    return np.array(sorted(chosen))


def teacher_prompt(state: dict, question: dict) -> str:
    """One chat turn asking for a single letter, no reasoning.

    The instruction suppresses explanation because the measurement is the probability
    of each letter as the next token, not the content of a reply.
    """
    criteria = question["criteria"]
    body = (
        f"人物：{state['人物']}\n"
        f"家庭收支（年度分档）：{state['家庭收支（年度分档）']}\n"
        f"当前事件：{state['当前事件']}\n"
        f"已知条款：{state['已知条款']}\n"
        f"题目：{question['instructions']}\n"
        + "\n".join(f"{key}. {criteria[key]}" for key in "ABCDEF")
    )
    return (
        "<|im_start|>system\n"
        "你在替这一户做一次性选择。不要思考，不要解释。下一个词只能是 A、B、C、D、E、F 中的一个字母。"
        "<|im_end|>\n"
        f"<|im_start|>user\n{body}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
