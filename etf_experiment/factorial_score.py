"""Scoring the 162-cell factorial by calling the encoder directly.

This is the high-volume path: 164 cells x 10,000 households is 1.64 million
decisions, so each request carries no HTTP overhead. The engine path exists for the
rounds that need world state; this one does not, and at this volume a per-request
round trip dominates the forward pass.

Households are grouped by asset bracket before scoring, because the idle-cash
amount -- and therefore the rendered question -- depends only on the bracket. That
turns 10,000 requests per cell into 5 distinct prompts, each scored for the
households in its bracket.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import factorial
from .direct import encoder
from .io import read_jsonl
from .options import describe_amount, describe_cash
from .persona import from_record, income_and_spending, persona

TOTAL_CELLS = len(factorial.cells())


def load_people(panel: Path) -> list[dict]:
    """One record per household with the fields the prompt needs, pre-rendered."""
    people = []
    for position, record in enumerate(read_jsonl(panel)):
        profile = from_record(record)
        people.append({
            "index": position,
            "asset": record["asset_bracket"],
            "x": describe_cash(float(record["assigned_idle_X_cny"])),
            "deposit": describe_amount(record["combo_deposit_P_cents"] / 100),
            "etf_leg": describe_amount(record["combo_etf_I_cents"] / 100),
            "persona": persona(profile),
            "flow": income_and_spending(profile),
        })
    return people


def state_for(person: dict, clause: str) -> dict:
    return {
        "人物": person["persona"],
        "家庭收支（年度分档）": person["flow"],
        "当前事件": f"家中现有闲置资金 {person['x']} 元。这笔钱可自主安排，以下方式都能在手机上办理。",
        "已知条款": clause,
    }


def questions_for(cell: factorial.Cell, x: str) -> dict:
    return {
        "action": {
            "type": "choice",
            "instructions": factorial.instructions(x),
            "criteria": factorial.menu(cell),
        }
    }


def score_cell(agent, people: list[dict], cell: factorial.Cell) -> dict[str, np.ndarray]:
    """Score one cell for every household, returning probabilities and scores."""
    by_asset: dict[str, list[dict]] = {}
    for person in people:
        by_asset.setdefault(person["asset"], []).append(person)

    probabilities = np.zeros((len(people), 6), np.float32)
    scores = np.zeros((len(people), 6), np.float32)
    choice = np.empty(len(people), dtype="U1")

    for group in by_asset.values():
        head = group[0]
        clause = factorial.clause_for(cell, head["x"], head["deposit"], head["etf_leg"])
        states = [state_for(person, clause) for person in group]
        answers = agent.predict_batch(states, questions_for(cell, head["x"]), batch_size=64)
        for person, answer in zip(group, answers):
            action = answer["answers"]["action"]
            probabilities[person["index"]] = [action["probabilities"][k] for k in "ABCDEF"]
            scores[person["index"]] = [action["scores"][k] for k in "ABCDEF"]
            choice[person["index"]] = action["choice"]

    return {"probabilities": probabilities, "scores": scores, "choice": choice}


def score_cells(
    *,
    panel: Path,
    model_dir: Path,
    out_dir: Path,
    gpu: int = 0,
    only: list[int] | None = None,
) -> list[int]:
    """Score the requested cells, one npz per cell. Resumable: existing files are kept."""
    factorial.self_check()
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = factorial.cells()
    wanted = list(range(len(catalog))) if only is None else list(only)
    people = load_people(panel)

    written = []
    with encoder(model_dir, gpu) as agent:
        for index in wanted:
            path = out_dir / f"cell_{index:03d}.npz"
            if path.is_file():
                continue
            cell = catalog[index]
            result = score_cell(agent, people, cell)
            np.savez_compressed(
                path,
                probabilities=result["probabilities"],
                scores=result["scores"],
                choice=result["choice"],
                **cell.as_record(),
            )
            written.append(index)
    return written


def spec_of(npz_path: Path) -> factorial.Cell:
    """Recover the treatment combination recorded inside a cell file."""
    with np.load(npz_path, allow_pickle=True) as data:
        return factorial.Cell(
            kind=str(data["kind"]),
            fee=str(data["fee"]),
            threshold=int(data["threshold"]),
            liquidity=str(data["liquidity"]),
            info=str(data["info"]),
            rate=float(data["rate"]),
        )


def load_cells(directory: Path) -> list[dict]:
    """Read every cell file into memory, with its specification attached."""
    out = []
    for path in sorted(Path(directory).glob("cell_*.npz")):
        with np.load(path, allow_pickle=True) as data:
            out.append({
                "path": path,
                "spec": factorial.Cell(
                    kind=str(data["kind"]), fee=str(data["fee"]),
                    threshold=int(data["threshold"]), liquidity=str(data["liquidity"]),
                    info=str(data["info"]), rate=float(data["rate"]),
                ),
                "probabilities": data["probabilities"].astype(float),
                "scores": data["scores"].astype(float),
                "choice": data["choice"],
            })
    if not out:
        raise FileNotFoundError(f"no cell_*.npz in {directory}")
    return out


def write_summary(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe(cell: factorial.Cell) -> str:
    return json.dumps(cell.as_record(), ensure_ascii=False)
