"""Building the second-tick tasks that write a realised outcome back into state.

The two-tick design asks the same household the same question twice. Tick 1 is the
already-completed fifteen-condition run; tick 2 keeps everything identical except
one appended sentence stating what happened to the first choice. That sentence, and
the amount it may change, is the entire treatment.

Which sentence a household gets depends on its first-tick argmax, so this module
must be fed the tick-1 files rather than recomputed from probabilities: the argmax
is what the household actually saw itself choose.
"""

from __future__ import annotations

from dataclasses import dataclass

from .options import describe_amount, describe_cash, net_after_subscription, subscription_fee

CONDITIONS = {4: "c04", 6: "c06", 8: "c08"}
IDLE_CASH = {"b1": 1_500.0, "b2": 9_100.0, "b3": 30_000.0, "b4": 53_000.0, "b5": 88_000.0}
BLOCKED_BRACKETS = ("b1", "b2")


@dataclass(frozen=True)
class Outcome:
    sentence: str
    question_amount: float


def _cash(value: float) -> str:
    return describe_cash(value)


def condition_6(bracket: str, choice: str) -> Outcome:
    amount = IDLE_CASH[bracket]
    shown = _cash(amount)
    if choice == "A":
        return Outcome(f"上一轮选择已成交：{shown} 元已存入一年期存款，持有至到期。", amount)
    if choice == "B":
        return Outcome(f"上一轮选择已成交：{shown} 元已买入货币基金，可随时取出。", amount)
    if choice in "CDEF" and bracket in BLOCKED_BRACKETS:
        return Outcome(
            f"上一轮选择未成交：所选产品净申购金额至少 10,000 元，"
            f"现有 {shown} 元不足，资金未动，目前没有持有该产品。",
            amount,
        )
    if choice in "CDEF":
        return Outcome(f"上一轮选择已成交：{shown} 元已投入所选产品，申购费 0 元。", amount)
    raise ValueError(f"unexpected choice {choice}")


def condition_8(bracket: str, choice: str) -> Outcome:
    amount = IDLE_CASH[bracket]
    shown = _cash(amount)
    if choice == "A":
        return Outcome(f"上一轮选择已成交：{shown} 元已存入一年期存款，未锁定，到期取回本息。", amount)
    if choice == "B":
        return Outcome(f"上一轮选择已成交：{shown} 元已买入货币基金，未锁定，可随时取出。", amount)
    if choice in "CDEF":
        return Outcome(
            f"上一轮选择已成交：{shown} 元已投入所选产品。"
            "未来 30 日不能赎回，满 30 日后才能免费赎回。",
            amount,
        )
    raise ValueError(f"unexpected choice {choice}")


def condition_4(bracket: str, choice: str) -> Outcome:
    """The fee case. Unlike the others, this one changes the question's amount.

    A household that chose to invest now holds the *net* amount, so the second
    question has to offer that reduced sum. A household that stayed in cash or the
    money fund was never charged and keeps the original amount.
    """
    gross = IDLE_CASH[bracket]
    shown = _cash(gross)
    if choice == "A":
        return Outcome(f"上一轮选择已成交：{shown} 元已存入一年期存款，未产生申购费。", gross)
    if choice == "B":
        return Outcome(f"上一轮选择已成交：{shown} 元已买入货币基金，未产生申购费，可随时取出。", gross)
    if choice in "CDEF":
        net = net_after_subscription(gross)
        fee = subscription_fee(gross)
        return Outcome(
            f"上一轮选择已成交：申购费 {describe_amount(fee)} 元已从资金中扣除，"
            f"净投入 {describe_amount(net)} 元已买入所选产品。"
            f"现在可重新安排的金额是 {describe_amount(net)} 元。",
            net,
        )
    raise ValueError(f"unexpected choice {choice}")


BUILDERS = {4: condition_4, 6: condition_6, 8: condition_8}


def outcome(condition: int, bracket: str, choice: str) -> Outcome:
    if condition not in BUILDERS:
        raise ValueError(f"condition {condition} has no two-tick arm")
    if bracket not in IDLE_CASH:
        raise ValueError(f"unknown bracket {bracket}")
    if choice not in "ABCDEF" or len(choice) != 1:
        raise ValueError(f"choice must be a single option letter, got {choice!r}")
    return BUILDERS[condition](bracket, choice)


def apply_to_payload(payload: dict, condition: int, bracket: str, choice: str) -> dict:
    """Append the outcome to the event and update the amount the question asks about.

    Returns a new payload; the input task row is not mutated, so tick 1 stays intact
    and the two can be diffed.
    """
    import copy

    tick2 = copy.deepcopy(payload)
    result = outcome(condition, bracket, choice)
    state = tick2["model_input"]["state"]
    question = tick2["model_input"]["questions"]["action"]

    before = _cash(IDLE_CASH[bracket])
    if f"闲置资金 {before} 元" not in state["当前事件"]:
        raise ValueError(f"event does not mention the idle cash {before}; wrong household?")
    state["当前事件"] = state["当前事件"] + result.sentence

    if result.question_amount != IDLE_CASH[bracket]:
        old = f"为题目给定的 {before} 元"
        if question["instructions"].count(old) != 1:
            raise ValueError("question amount phrase is missing or ambiguous")
        question["instructions"] = question["instructions"].replace(
            old, f"为题目给定的 {_cash(result.question_amount)} 元", 1
        )

    tick2.setdefault("meta", {})
    tick2["meta"].update({
        "tick": "2",
        "tick1_option": choice,
        "outcome": result.sentence,
        "question_amount": _cash(result.question_amount),
    })
    return tick2
