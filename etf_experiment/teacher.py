"""The teacher contrast: does a larger model recover the directions a small one gives?

The small decision models are fine-tuned on this exact task. A 27B instruction-tuned
model reading the same prompt is not, so agreement is evidence that a direction is a
property of the situation rather than of one checkpoint's training.

The 1,000 household subset is drawn with the same seed and a fixed per-bracket quota
(300/260/240/120/80), so the small models can be read off the factorial cells for the
exact same households. That is what makes the comparison paired rather than
independent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import probes
from .stats import wmean

# Ten named conditions, each mapped to the factorial cell holding the same treatment.
CELLS = {
    "baseline": 0, "rate_095": 1, "drawdown": 2, "upside": 4,
    "redeem": 6, "lock": 12, "thr_100": 18, "thr_10000": 36,
    "open_200": 54, "sub_0.5": 108,
}

# Pre-specified sign of each named probability. Empty means no prior, so the row is
# reported without a verdict rather than being scored as a hit or miss.
EXPECTED = {
    ("open_200", "C到F"): -1,
    ("sub_0.5", "C到F"): -1,
    ("thr_10000", "C到F"): -1,
    ("thr_10000_b1", "C到F"): -1,
    ("thr_100", "C到F"): 1,
    ("redeem", "权益ETF"): -1,
    ("lock", "权益ETF"): -1,
    ("drawdown", "权益ETF"): -1,
    ("upside", "权益ETF"): 1,
    ("rate_095", "存款"): -1,
    ("rate_095", "货币基金"): 1,
}

METRICS = ("存款", "货币基金", "纯债", "权益ETF", "C到F")


def metrics(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "存款": probabilities[:, 0],
        "货币基金": probabilities[:, 1],
        "纯债": probabilities[:, 2],
        "权益ETF": probabilities[:, 3] + probabilities[:, 4] + probabilities[:, 5],
        "C到F": probabilities[:, 2] + probabilities[:, 3] + probabilities[:, 4] + probabilities[:, 5],
    }


def load_teacher(directory: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Merge the per-GPU shards, restoring panel order.

    Shards are written by GPU, so their row order follows the shard split rather than
    the panel; sorting by ``sample_index`` is what lets the teacher be joined to the
    factorial cells household by household.
    """
    shards = [np.load(path) for path in sorted(Path(directory).glob("gpu*.npz"))]
    if not shards:
        raise FileNotFoundError(f"no gpu*.npz in {directory}")
    order = np.concatenate([shard["sample_index"] for shard in shards])
    teacher = np.concatenate([shard["probabilities"] for shard in shards], axis=0)
    names = [str(value) for value in shards[0]["conditions"]]
    rank = np.argsort(order)
    return order[rank], teacher[rank], names


def compare(
    teacher_order: np.ndarray,
    teacher: np.ndarray,
    names: list[str],
    factorial_dir: Path,
    panel_records: list[dict],
) -> list[dict]:
    """Rows for both models, with the pre-specified verdict where one exists."""
    assets = np.array([record["asset_bracket"] for record in panel_records])[teacher_order]
    weights = np.array([float(record["normalized_weight"]) for record in panel_records])[teacher_order]

    small = np.stack([
        np.load(Path(factorial_dir) / f"cell_{CELLS[name]:03d}.npz")["probabilities"][teacher_order]
        for name in names
    ], axis=1)

    groups = {"全体": np.ones(len(teacher_order), dtype=bool), "b1": assets == "b1"}
    rows = []
    for model, cube in (("27B", teacher), ("sharp-322m", small)):
        base = metrics(cube[:, names.index("baseline")])
        for name in names:
            got = metrics(cube[:, names.index(name)])
            for group, mask in groups.items():
                group_weight = weights[mask]
                for label in METRICS:
                    level = wmean(got[label][mask], group_weight)
                    delta = 0.0 if name == "baseline" else wmean((got[label] - base[label])[mask], group_weight)
                    key = (f"{name}_b1" if group == "b1" and name == "thr_10000" else name, label)
                    expected = EXPECTED.get(key, "")
                    verdict = ""
                    if expected != "" and name != "baseline":
                        verdict = "是" if np.sign(delta) == expected and abs(delta) > 1e-6 else "否"
                    rows.append({
                        "模型": model, "条件": name, "组别": group, "指标": label,
                        "水平": f"{level:.6f}", "相对基准": f"{delta:.6f}",
                        "预期符号": expected, "符号符合": verdict, "户数": int(mask.sum()),
                    })
    return rows


def summary(rows: list[dict]) -> dict[str, int]:
    """How many pre-specified signs each model got right."""
    hits: dict[str, int] = {}
    for row in rows:
        if row["符号符合"] == "":
            continue
        hits.setdefault(row["模型"], 0)
        if row["符号符合"] == "是":
            hits[row["模型"]] += 1
    return hits


def teacher_sample_for(panel_records: list[dict]) -> np.ndarray:
    """The 1,000 household indices, from the same quota draw the teacher run used."""
    assets = np.array([record["asset_bracket"] for record in panel_records])
    return probes.teacher_sample(assets)
