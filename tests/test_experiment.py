"""Tests for the parts where a silent bug would corrupt a published number.

The most important test is ``test_condition_text_matches_frozen_inputs``: it pins
the refactored template against the exact strings the models were scored on. If a
wording edit slips in, that test fails rather than quietly changing the treatment.

Run with ``python -m pytest tests`` from the repository root.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from etf_experiment import options
from etf_experiment.conditions import build, instruction
from etf_experiment.io import read_jsonl
from etf_experiment.stats import weighted_ci

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PANEL = ROOT / "data" / "10000户样本_20261022" / "家庭数据.jsonl"
FROZEN_INPUTS = ROOT / "inputs"


# ---------------------------------------------------------------- money rules
@pytest.mark.parametrize(
    "x, net, fee",
    [(1_500, 1_492.54, 7.46), (9_100, 9_054.73, 45.27), (30_000, 29_850.75, 149.25),
     (53_000, 52_736.32, 263.68), (88_000, 87_562.19, 437.81)],
)
def test_subscription_fee_matches_plan_table(x, net, fee):
    assert options.net_after_subscription(x) == pytest.approx(net)
    assert options.subscription_fee(x) == pytest.approx(fee)
    # The fee is charged on the gross amount, so it is slightly above 0.5% of net.
    assert fee == pytest.approx(x - net, abs=0.005)


def test_combo_split_covers_x_with_the_deposit_leg():
    for x in (1_500, 9_100, 30_000, 53_000, 88_000):
        deposit, etf = options.split_deposit_etf(x)
        assert deposit * 1.015 >= x - 1e-6, "deposit leg must mature to at least X"
        assert deposit + etf == pytest.approx(x, abs=0.005)


@pytest.mark.parametrize("condition,threshold", [(1, 1_000), (5, 100), (6, 10_000)])
def test_threshold_binds_only_below_the_minimum(condition, threshold):
    x = np.array([threshold - 1.0, float(threshold), threshold + 1_000.0])
    amount = options.investable(condition, x)
    assert amount[0] == 0.0
    assert amount[1] > 0 and amount[2] > 0


def test_open_fee_can_push_a_small_account_below_the_minimum():
    # 1,000 - 200 = 800 falls under the 1,000 floor that would otherwise clear.
    assert options.investable(3, np.array([1_000.0]))[0] == 0.0


def test_cash_rendering_is_whole_yuan_but_legs_are_cents():
    assert options.describe_cash(62_500.0) == "62,500"
    assert options.describe_amount(923.64) == "923.64"


# ---------------------------------------------------------------- templates
@pytest.mark.skipif(not FIXTURES.is_dir(), reason="frozen text fixtures are not shipped")
def test_condition_text_matches_frozen_inputs():
    """Every condition must render byte-identically to what the models saw.

    The fixtures hold the full ``state`` and ``questions`` of one household per
    condition, captured from the actual scoring run. They are small on purpose
    (31 KB) so the check travels with the repository: a wording edit that would
    silently change the treatment fails here instead of after a 150,000-row rerun.
    """
    for number in range(1, 16):
        path = FIXTURES / f"c{number:02d}.json"
        if not path.is_file():
            pytest.skip(f"{path.name} not present")
        frozen = json.loads(path.read_text(encoding="utf-8"))
        state = frozen["state"]
        x = float(state["当前事件"].split("闲置资金 ")[1].split(" 元")[0].replace(",", ""))
        spec = build(number, x)
        assert state["当前事件"] == spec.event, f"condition {number} event drifted"
        assert state["已知条款"] == spec.clause, f"condition {number} clause drifted"
        assert frozen["questions"]["action"]["instructions"] == instruction(x)
        assert frozen["questions"]["action"]["criteria"] == spec.criteria


def test_every_condition_keeps_six_options():
    for number in range(1, 16):
        spec = build(number, 30_000.0)
        assert set(spec.criteria) == set("ABCDEF")


def test_condition_13_is_condition_12_plus_one_sentence():
    twelve = build(12, 30_000.0)
    thirteen = build(13, 30_000.0)
    assert thirteen.clause.startswith(twelve.clause)
    assert thirteen.clause[len(twelve.clause):] == "另给定同期物价年涨幅为 2.5%。"


# ---------------------------------------------------------------- statistics
def test_cluster_bootstrap_is_reproducible():
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    values = np.random.default_rng(1).random(500)
    weights = np.ones(500)
    first = weighted_ci(values, weights, rng_a, draws=200)
    second = weighted_ci(values, weights, rng_b, draws=200)
    assert first == second


def test_interval_brackets_the_point_estimate():
    values = np.random.default_rng(2).random(1_000)
    estimate = weighted_ci(values, np.ones(1_000), np.random.default_rng(3), draws=500)
    assert estimate.low <= estimate.point <= estimate.high


# ---------------------------------------------------------------- panel
@pytest.mark.skipif(not PANEL.is_file(), reason="panel is not shipped")
def test_panel_shape_and_weights():
    records = read_jsonl(PANEL)
    assert len(records) == 10_000
    weights = np.array([float(record["normalized_weight"]) for record in records])
    assert weights.sum() == pytest.approx(10_000, abs=1e-6)
    orders = {tuple(record["display_order"]) for record in records}
    assert all(sorted(order) == list("ABCDEF") for order in orders)


# ---------------------------------------------------------------- portability
def test_no_machine_specific_paths_in_package():
    """The package must not bake in an absolute home or checkout path.

    This is the regression test for the thing that makes a repository
    uncloneable: a path like /home/<user>/... or /Users/<name>/... that worked on
    the machine where the experiment ran. Every location has to come from
    ``paths.get_paths()`` so a clone runs wherever it lands.
    """
    import re

    package = Path(__file__).resolve().parents[1] / "etf_experiment"
    pattern = re.compile(r"/(?:home|Users|root|workspace)/")
    offenders = []
    for source in package.rglob("*.py"):
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{source.relative_to(package)}:{number}: {line.strip()}")
    assert not offenders, "machine-specific paths found:\n" + "\n".join(offenders)


def test_paths_resolve_without_environment():
    """Unconfigured paths must still resolve, and relative to the checkout."""
    from etf_experiment import paths as paths_module

    saved = {key: os.environ.pop(key) for key in list(os.environ) if key.startswith(("ETF_", "MAS_ENGINE"))}
    try:
        resolved = paths_module.get_paths()
    finally:
        os.environ.update(saved)
    root = Path(__file__).resolve().parents[1]
    assert resolved.root == root
    assert resolved.data == root / "data"
    assert resolved.scorer_dir.is_dir()
    assert resolved.engine_python.is_file()


# ---------------------------------------------------------------- placebo arms
def test_placebo_arms_are_length_matched_and_content_free():
    """The control must differ from the treatment only in meaning.

    If the two sentences drift apart in length, the arm stops isolating content and
    the +1.6 to +2.4pp contrast in the report would be uninterpretable.
    """
    from etf_experiment.placebo import AMOUNTS, CONTROL, FAILURE

    failure = FAILURE.format(amount=AMOUNTS["b1"])
    assert len(failure) == len(FAILURE.format(amount=AMOUNTS["b2"])) == len(CONTROL)
    for word in ("未成交", "门槛", "10,000", "不足", "起购", "资金", "持有", "产品", "元"):
        assert word not in CONTROL, f"control leaks treatment vocabulary: {word}"
    assert failure.startswith("上一轮选择") and CONTROL.startswith("上一轮选择已")


def test_placebo_split_is_balanced_within_bracket():
    from etf_experiment.placebo import BRACKET_COUNTS, build

    indexes = {"b1": list(range(3_000)), "b2": list(range(3_000, 5_600))}
    arms = build(indexes)
    assert len(arms) == sum(BRACKET_COUNTS.values())
    for bracket, expected in BRACKET_COUNTS.items():
        subset = [arm for arm in arms if arm.bracket == bracket]
        assert len(subset) == expected
        failure = sum(1 for arm in subset if arm.kind == "failure")
        assert failure == expected // 2, f"{bracket}: {failure} of {expected} in the failure arm"


# ---------------------------------------------------------------- two-tick arms
@pytest.mark.parametrize(
    "condition,bracket,choice,expect_net",
    [(4, "b1", "C", True), (4, "b5", "F", True), (4, "b1", "A", False),
     (6, "b1", "C", False), (6, "b3", "C", False), (6, "b2", "A", False),
     (8, "b4", "E", False)],
)
def test_two_tick_only_changes_the_amount_on_the_fee_arm(condition, bracket, choice, expect_net):
    """Condition 4 reduces the sum re-offered; the others leave it alone."""
    from etf_experiment.options import IDLE_X, net_after_subscription
    from etf_experiment.two_tick import outcome

    result = outcome(condition, bracket, choice)
    if expect_net:
        assert result.question_amount == pytest.approx(net_after_subscription(IDLE_X["new"][bracket]))
    else:
        assert result.question_amount == pytest.approx(IDLE_X["new"][bracket])


def test_two_tick_marks_the_unaffordable_arm_as_unsuccessful():
    """b1 cannot buy under a 10,000 floor, so the sentence must say so."""
    from etf_experiment.two_tick import outcome

    assert "未成交" in outcome(6, "b1", "C").sentence
    assert "未成交" not in outcome(6, "b3", "C").sentence
    assert "未成交" not in outcome(6, "b1", "A").sentence
