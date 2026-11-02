"""Scoring the probes and turning them into the two published tables.

Position probes need something the engine path cannot give: the same question scored
under two option orders, with the pre-softmax scores kept. The scores matter because a
probability difference conflates "the model likes this product more" with "the model
likes whatever is listed first"; comparing scores before the softmax separates them.

The wording probes follow the same pattern as the factorial: build the clause, score
10,000 households in fixed A–F order, subtract the re-rendered baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import probes
from .direct import encoder
from .io import read_csv, read_jsonl, write_csv
from .options import describe_cash
from .persona import from_record, income_and_spending, persona
from .stats import SEED, weighted_ci, wmean, rng_for

# The published table splits the three equity ETFs and drops "离开存款" (which is
# just the negative of "存款"), so the row set matches what a reader can cross-check.
# Order matters for reproducing the published file byte for byte: the three equity
# ETFs come first and their sum comes last, which is how the original table reads.
METRICS = ("存款", "货币基金", "纯债", "红利低波", "宽基", "科技成长", "权益ETF")
POSITION_METRICS = ("存款", "货币基金", "纯债", "红利低波", "宽基", "科技成长")


def load_people(panel: Path) -> list[dict]:
    people = []
    for position, record in enumerate(read_jsonl(panel)):
        profile = from_record(record)
        people.append({
            "index": position,
            "asset": record["asset_bracket"],
            "x": describe_cash(float(record["assigned_idle_X_cny"])),
            "persona": persona(profile),
            "flow": income_and_spending(profile),
            "order": tuple(record["display_order"]),
        })
    return people


def state_for(person: dict, clause: str) -> dict:
    return {
        "人物": person["persona"],
        "家庭收支（年度分档）": person["flow"],
        "当前事件": f"家中现有闲置资金 {person['x']} 元。这笔钱可自主安排，以下方式都能在手机上办理。",
        "已知条款": clause,
    }


def question_for(criteria: dict[str, str], x: str) -> dict:
    return {"action": {"type": "choice",
                       "instructions": f"以该家庭身份，为题目给定的 {x} 元闲置资金选择一种安排，仅选一项。",
                       "criteria": criteria}}


def outcomes(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "存款": probabilities[:, 0],
        "货币基金": probabilities[:, 1],
        "纯债": probabilities[:, 2],
        "权益ETF": probabilities[:, 3] + probabilities[:, 4] + probabilities[:, 5],
        "红利低波": probabilities[:, 3],
        "宽基": probabilities[:, 4],
        "科技成长": probabilities[:, 5],
        "离开存款": 1 - probabilities[:, 0],
    }


def score_wording(agent, people: list[dict], probe: probes.Probe) -> dict[str, np.ndarray]:
    """Score one wording probe for every household, in fixed A–F order."""
    probabilities = np.zeros((len(people), 6), np.float32)
    scores = np.zeros((len(people), 6), np.float32)
    by_asset: dict[str, list[dict]] = {}
    for person in people:
        by_asset.setdefault(person["asset"], []).append(person)
    for group in by_asset.values():
        head = group[0]
        states = [state_for(person, probe.clause) for person in group]
        criteria = probes.fixed_order(probe.criteria)
        answers = agent.predict_batch(states, question_for(criteria, head["x"]), batch_size=64)
        for person, answer in zip(group, answers):
            action = answer["answers"]["action"]
            probabilities[person["index"]] = [action["probabilities"][k] for k in "ABCDEF"]
            scores[person["index"]] = [action["scores"][k] for k in "ABCDEF"]
    return {"probabilities": probabilities, "scores": scores}


def score_position(agent, people: list[dict], condition: int, inputs: Path, by_order: Path):
    """Score one condition in fixed order, and pair it with the randomised-order run.

    The randomised half is read from an existing run rather than re-scored: it must be
    the identical scoring pass that produced the main results, or the two halves would
    differ by more than option order.
    """
    rows = read_jsonl(inputs / f"c{condition:02d}.jsonl")
    fixed_probs = np.zeros((len(people), 6), np.float32)
    fixed_scores = np.zeros((len(people), 6), np.float32)

    buckets: dict[str, dict] = {}
    for row in rows:
        criteria = {key: row["questions"]["action"]["criteria"][key] for key in "ABCDEF"}
        # A deterministic key: string hashing is randomised per interpreter, so
        # ``hash()`` here would split the same question into different buckets on
        # different runs and silently change the batching.
        key = "|".join(criteria[key] for key in "ABCDEF") + "||" + row["questions"]["action"]["instructions"]
        buckets.setdefault(key, {"criteria": criteria, "rows": [],
                                 "questions": {"action": {**row["questions"]["action"],
                                                          "criteria": criteria}}})
        buckets[key]["rows"].append(row)

    for bucket in buckets.values():
        answers = agent.predict_batch(
            [row["state"] for row in bucket["rows"]], bucket["questions"], batch_size=64
        )
        for row, answer in zip(bucket["rows"], answers):
            index = int(row["meta"]["sample_index"])
            action = answer["answers"]["action"]
            fixed_probs[index] = [action["probabilities"][k] for k in "ABCDEF"]
            fixed_scores[index] = [action["scores"][k] for k in "ABCDEF"]

    random_probs = np.zeros_like(fixed_probs)
    random_scores = np.zeros_like(fixed_scores)
    for row in read_jsonl(by_order / f"c{condition:02d}.jsonl"):
        index = int(row["meta_sample_index"])
        random_scores[index] = [row["scores"][k] for k in "ABCDEF"]
        random_probs[index] = [row["probabilities"][k] for k in "ABCDEF"]

    return {
        "fixed_probs": fixed_probs, "fixed_scores": fixed_scores,
        "random_probs": random_probs, "random_scores": random_scores,
    }


def score_all(
    *,
    panel: Path,
    model_dir: Path,
    out_dir: Path,
    inputs: Path,
    by_order: Path,
    gpu: int = 0,
    names: list[str] | None = None,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    people = load_people(panel)
    wanted = probes.task_names() if names is None else names
    written = []
    with encoder(model_dir, gpu) as agent:
        for name in wanted:
            path = out_dir / f"{name}.npz"
            if path.is_file():
                continue
            if name.startswith("position_"):
                condition = int(name.split("_")[1])
                result = score_position(agent, people, condition, inputs, by_order)
                np.savez_compressed(path, **result)
            else:
                result = score_wording(agent, people, probes.build(name))
                np.savez_compressed(path, **result)
            written.append(name)
    return written


# ---------------------------------------------------------------- tables
def wording_table(out_dir: Path, weights: np.ndarray) -> list[dict]:
    """Each wording probe minus the re-rendered baseline, in long form.

    One row per (probe, metric). Long form rather than wide because the published
    table is long: it keeps `指标` as a filterable column and lets the interval stay
    next to the estimate it belongs to.
    """
    base = outcomes(np.load(out_dir / "baseline_ref.npz")["probabilities"].astype(float))
    rows = []
    for name in probes.WORDING_NAMES[1:]:
        path = out_dir / f"{name}.npz"
        if not path.is_file():
            continue
        probe = outcomes(np.load(path)["probabilities"].astype(float))
        for metric in METRICS:
            # A fresh generator per metric: the published intervals were drawn that
            # way, so reusing one stream would shift every interval after the first.
            estimate = weighted_ci(
                probe[metric] - base[metric], weights, np.random.default_rng(20261021)
            )
            rows.append({
                "探针": name,
                "指标": metric,
                "差值": f"{estimate.point:.6f}",
                "下限": f"{estimate.low:.6f}",
                "上限": f"{estimate.high:.6f}",
            })
    return rows


def position_table(out_dir: Path, weights: np.ndarray) -> list[dict]:
    """Score and probability differences between randomised and fixed option order.

    Two subtleties that change the numbers:

    1. The sign is randomised minus fixed, so a positive value means the household's
       own order *raised* that option relative to the canonical A–F listing.
    2. The gap is averaged over the fifteen conditions **per household first**, then
       the absolute value is taken. Taking |difference| per condition and averaging
       afterwards gives a systematically larger number, because it removes the
       cancellation between conditions; the published table is the per-household
       version.
    """
    names = [f"position_{index:02d}" for index in probes.POSITION_INDICES]
    score_gap = np.zeros((len(weights), 6))
    prob_gap = np.zeros((len(weights), 6))
    flips = np.zeros(len(weights))
    total = 0
    for name in names:
        path = out_dir / f"{name}.npz"
        if not path.is_file():
            continue
        data = np.load(path)
        fixed_scores = data["fixed_scores"].astype(float)
        random_scores = data["random_scores"].astype(float)
        fixed_probs = data["fixed_probs"].astype(float)
        random_probs = data["random_probs"].astype(float)
        score_gap += random_scores - fixed_scores
        prob_gap += random_probs - fixed_probs
        flips += (random_probs.argmax(axis=1) != fixed_probs.argmax(axis=1)).astype(float)
        total += 1
    if total == 0:
        return []
    score_gap /= total
    prob_gap /= total
    flips /= total

    rows = []
    labels = ("存款", "货币基金", "纯债", "红利低波", "宽基", "科技成长")
    for index, name in enumerate(labels):
        score_estimate = weighted_ci(score_gap[:, index], weights, np.random.default_rng(20261021))
        prob_estimate = weighted_ci(prob_gap[:, index], weights, np.random.default_rng(20261022))
        rows.append({
            "选项": name,
            "随机顺序减固定_分数": f"{score_estimate.point:.6f}",
            "分数下限": f"{score_estimate.low:.6f}",
            "分数上限": f"{score_estimate.high:.6f}",
            "随机顺序减固定_概率": f"{prob_estimate.point:.6f}",
            "概率下限": f"{prob_estimate.low:.6f}",
            "概率上限": f"{prob_estimate.high:.6f}",
            "平均绝对分数差": f"{wmean(np.abs(score_gap[:, index]), weights):.6f}",
        })

    # The headline finding of this table: how often the top choice changes when the
    # order changes. Averaged per household across the fifteen conditions.
    flip_estimate = weighted_ci(flips, weights, np.random.default_rng(20261023))
    rows.append({
        "选项": "argmax翻转率",
        "随机顺序减固定_分数": "",
        "分数下限": "",
        "分数上限": "",
        "随机顺序减固定_概率": f"{flip_estimate.point:.6f}",
        "概率下限": f"{flip_estimate.low:.6f}",
        "概率上限": f"{flip_estimate.high:.6f}",
        "平均绝对分数差": f"{wmean(np.mean(np.abs(score_gap), axis=1), weights):.6f}",
    })
    return rows


def write_tables(out_dir: Path, weights: np.ndarray, table_dir: Path) -> dict[str, int]:
    table_dir.mkdir(parents=True, exist_ok=True)
    wording = wording_table(out_dir, weights)
    positions = position_table(out_dir, weights)
    if wording:
        write_csv(table_dir / "词汇与改写.csv", wording)
    if positions:
        write_csv(table_dir / "位置效应.csv", positions)
    return {"wording_rows": len(wording), "position_rows": len(positions)}
