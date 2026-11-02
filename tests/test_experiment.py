"""Tests for the parts where a silent bug would corrupt a published number.

The most important test is ``test_condition_text_matches_frozen_inputs``: it pins
the refactored template against the exact strings the models were scored on. If a
wording edit slips in, that test fails rather than quietly changing the treatment.

Run with ``python -m pytest tests`` from the repository root.
"""

from __future__ import annotations

import json
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
