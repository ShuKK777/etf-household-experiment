"""Building the three published tables from per-household probabilities.

The tables are pure functions of (probabilities, weights, groups, capacity), so
they rebuild from archived jsonl without a GPU and are byte-stable for a fixed
seed. That is what makes the numbers in the report auditable.

Table 1  baseline      what the six products get, overall and by subgroup
Table 2  clause impact each condition minus the baseline, plus the blocked mass
Table 3  heterogeneity the same differences cut by each household dimension
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import groups as group_module
from .groups import DIMENSIONS, FOCUS, Groups, label
from .io import read_jsonl
from .metrics import Probabilities, blocked_by_threshold, capacity_matrix
from .options import KEYS
from .stats import rng_for, weighted_ci, weighted_ratio_ci

CONDITIONS = 15
BASELINE_CONTRASTS = [(number, 1) for number in range(2, 16)] + [(13, 12)]

CONDITION_NAMES = {
    1: "基准", 2: "三年正收益比例", 3: "开户费 200 元", 4: "申购费 0.5%",
    5: "起购 100 元", 6: "起购 10,000 元", 7: "30 日内赎回费", 8: "30 日锁定期",
    9: "告知生活费已预留", 10: "回撤信息", 11: "按季分红", 12: "存款降息",
    13: "降息＋通胀", 14: "资金托管", 15: "大部分存款＋小部分 ETF",
}
METRIC_ORDER = ("离开存款", "货基", "纯债", "纯债或ETF", "权益ETF", "ETF金额")


# ---------------------------------------------------------------- loading
def load_probabilities(results_dir: Path, households: int = 10_000) -> Probabilities:
    """Read the fifteen per-condition files into one array.

    Missing or non-ok rows raise rather than being skipped: a partially finished
    run must not produce a table that looks complete.
    """
    array = np.full((households, CONDITIONS, len(KEYS)), np.nan)
    paths = sorted(Path(results_dir).glob("c*.jsonl"))
    if len(paths) != CONDITIONS:
        raise SystemExit(f"expected {CONDITIONS} condition files in {results_dir}, found {len(paths)}")
    for path in paths:
        for row in read_jsonl(path):
            if row.get("status") != "ok":
                raise SystemExit(f"unresolved row {row.get('agent_id')} in {path}")
            array[int(row["meta_sample_index"]), int(row["meta_condition"]) - 1] = [
                float(row["probabilities"][key]) for key in KEYS
            ]
    if np.isnan(array).any():
        missing = int(np.isnan(array).any(axis=2).sum())
        raise SystemExit(f"{missing} household-conditions have no decision")
    return Probabilities(array)


def load_panel_frame(path: Path, households: int = 10_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Idle cash, the condition-15 ETF leg, and the sampling weight, in stored order.

    The ETF leg is stored separately because condition 15 splits X into a deposit
    that grows to X and a small ETF position; its executable amount is that leg,
    not X.
    """
    records = read_jsonl(path)
    if len(records) != households:
        raise SystemExit(f"expected {households} households in {path}, found {len(records)}")
    x = np.array([float(record["assigned_idle_X_cny"]) for record in records])
    etf_leg = np.array([record["combo_etf_I_cents"] / 100 for record in records])
    weights = np.array([float(record["normalized_weight"]) for record in records])
    return x, etf_leg, weights


def idle_cash_from_brackets(asset: np.ndarray, schedule: str = "new") -> np.ndarray:
    """Recompute X from the asset bracket under either calibration schedule."""
    from .options import IDLE_X

    table = IDLE_X[schedule]
    return np.array([table[str(bracket)] for bracket in asset])


def combo_legs(x: np.ndarray) -> np.ndarray:
    """The condition-15 deposit and ETF legs implied by a schedule of X values."""
    from .options import split_deposit_etf

    return np.array([split_deposit_etf(value)[1] for value in x])


# ---------------------------------------------------------------- table 1
def baseline_table(
    probabilities: Probabilities,
    capacity: np.ndarray,
    weights: np.ndarray,
    groups: Groups,
) -> list[dict]:
    metrics = probabilities.metrics(capacity)
    rows: list[dict] = []
    slices: list[tuple[np.ndarray, str, str]] = [
        (np.ones(len(weights), dtype=bool), "全体", "全体")
    ]
    for field in DIMENSIONS:
        slices += [(groups.mask(field, level), label(field, level), DIMENSIONS[field][2])
                   for level in groups.levels(field)]

    for mask, shown, heading in slices:
        n = int(mask.sum())
        if n == 0:
            continue
        rng = rng_for("table1", heading, shown)
        row: dict[str, object] = {"分组": heading, "组别": shown, "户数": n}
        for key in KEYS:
            row.update(weighted_ci(probabilities[key][mask, 0], weights[mask], rng).as_columns(key))
        for name in METRIC_ORDER:
            row.update(weighted_ci(metrics[name][mask, 0], weights[mask], rng).as_columns(name))
        ratio, num, den = weighted_ratio_ci(
            probabilities.equity[mask, 0], probabilities.bond_or_equity[mask, 0], weights[mask], rng
        )
        row["ETF占纯债及ETF_估计"] = "" if ratio is None else f"{ratio.point:.6f}"
        row["ETF占纯债及ETF_下限"] = "" if ratio is None else f"{ratio.low:.6f}"
        row["ETF占纯债及ETF_上限"] = "" if ratio is None else f"{ratio.high:.6f}"
        row["ETF分子"] = f"{num:.6f}"
        row["纯债及ETF分母"] = f"{den:.6f}"
        rows.append(row)
    return rows


# ---------------------------------------------------------------- table 2
def clause_table(
    probabilities: Probabilities,
    capacity: np.ndarray,
    weights: np.ndarray,
    *,
    blocked: np.ndarray | None = None,
) -> list[dict]:
    metrics = probabilities.metrics(capacity)
    rows: list[dict] = []
    for left, right in BASELINE_CONTRASTS:
        rng = rng_for("table2", left, right)
        title = f"#{left}−#{right}"
        row: dict[str, object] = {"对比": title, "条件": CONDITION_NAMES[left], "户数": len(weights)}
        for key in KEYS:
            delta = probabilities[key][:, left - 1] - probabilities[key][:, right - 1]
            row.update(weighted_ci(delta, weights, rng).as_columns(key))
        for name in METRIC_ORDER:
            delta = metrics[name][:, left - 1] - metrics[name][:, right - 1]
            row.update(weighted_ci(delta, weights, rng).as_columns(name))
        rows.append(row)

    if blocked is not None and blocked.any():
        rng = rng_for("table2", "blocked")
        estimate = weighted_ci(probabilities.bond_or_equity[blocked, 5], weights[blocked], rng)
        zero = {"估计": "0.000000", "下限": "0.000000", "上限": "0.000000"}
        rows.append(
            {
                "对比": "#6 购买未达成",
                "条件": "起购 10,000 元时达不到门槛的 C–F 概率，可执行金额记 0",
                "户数": int(blocked.sum()),
                "纯债或ETF_估计": f"{estimate.point:.6f}",
                "纯债或ETF_下限": f"{estimate.low:.6f}",
                "纯债或ETF_上限": f"{estimate.high:.6f}",
                "ETF金额_估计": zero["估计"],
                "ETF金额_下限": zero["下限"],
                "ETF金额_上限": zero["上限"],
            }
        )
    return rows


# ---------------------------------------------------------------- table 3
def heterogeneity_table(
    probabilities: Probabilities,
    capacity: np.ndarray,
    weights: np.ndarray,
    groups: Groups,
) -> list[dict]:
    metrics = probabilities.metrics(capacity)
    rows: list[dict] = []
    for field, contrasts in FOCUS.items():
        if field == "asset":
            slices = [(groups.mask(field, level), label(field, level), "")
                      for level in groups.levels(field)]
        else:
            # Debt and flow are cut within each asset bracket, because the same
            # debt level means something different at 5,000 and 500,000 yuan.
            slices = []
            for bracket in groups.levels("asset"):
                within = groups.mask("asset", bracket)
                combined = np.where(within, groups[field], "")
                for level in groups.levels(field):
                    slices.append((combined == level, label(field, level), label("asset", bracket)))

        for mask, shown, bracket_shown in slices:
            n = int(mask.sum())
            if n == 0:
                continue
            for title in contrasts:
                left, right = _parse(title)
                rng = rng_for("table3", field, shown, bracket_shown, title)
                record: dict[str, object] = {
                    "分组": DIMENSIONS[field][2], "资产档": bracket_shown,
                    "组别": shown, "户数": n, "对比": title,
                }
                for name in METRIC_ORDER:
                    delta = metrics[name][mask, left - 1] - metrics[name][mask, right - 1]
                    record.update(weighted_ci(delta, weights[mask], rng).as_columns(name))
                rows.append(record)

    # The plan asks for #13−#12 inside every dimension, not only for literacy.
    for field in DIMENSIONS:
        for level in groups.levels(field):
            mask = groups.mask(field, level)
            n = int(mask.sum())
            if n == 0:
                continue
            rng = rng_for("table3", field, level, "#13−#12")
            record = {
                "分组": DIMENSIONS[field][2], "资产档": "", "组别": label(field, level),
                "户数": n, "对比": "#13−#12",
            }
            for name in METRIC_ORDER:
                delta = metrics[name][mask, 12] - metrics[name][mask, 11]
                record.update(weighted_ci(delta, weights[mask], rng).as_columns(name))
            rows.append(record)
    return rows


def _parse(title: str) -> tuple[int, int]:
    left, right = title.replace("#", "").split("−")
    return int(left), int(right)


def build_all(
    results_dir: Path,
    panel: Path,
    out_dir: Path,
    *,
    households: int = 10_000,
    schedule: str = "new",
) -> dict[str, int]:
    """Rebuild the three tables.

    ``schedule`` picks the idle-cash calibration. The stored panel carries the old
    amounts, so rebuilding a new-schedule table has to recompute X from the asset
    bracket instead of reading ``assigned_idle_X_cny``.
    """
    from .io import write_csv

    probabilities = load_probabilities(results_dir, households)
    records = read_jsonl(panel)[:households]
    frame_x, stored_leg, weights = load_panel_frame(panel, households)
    groups = group_module.from_records(records)

    if schedule == "new":
        x = idle_cash_from_brackets(groups["asset"], "new")
        etf_leg = combo_legs(x)
    else:
        x = frame_x
        etf_leg = stored_leg

    capacity = capacity_matrix(x, etf_leg)
    blocked = blocked_by_threshold(x)

    baseline = baseline_table(probabilities, capacity, weights, groups)
    clauses = clause_table(probabilities, capacity, weights, blocked=blocked)
    heterogeneity = heterogeneity_table(probabilities, capacity, weights, groups)

    write_csv(out_dir / "基准选择.csv", baseline)
    write_csv(out_dir / "条款影响.csv", clauses)
    write_csv(out_dir / "人群差异.csv", heterogeneity)
    return {
        "schedule": schedule,
        "baseline_rows": len(baseline),
        "clause_rows": len(clauses),
        "group_rows": len(heterogeneity),
    }
