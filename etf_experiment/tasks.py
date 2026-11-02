"""Rendering the panel and the conditions into persona_choice task rows.

One task row is one household facing one condition. The engine's loader groups
rows into batches of 1,000, and each batch file is one session, so a row must
carry its own display order and sampling weight: the model never sees ``meta``.

The output schema is the engine's, not ours: ``{"agent_id", "task", "meta"}``
where ``task.model_input`` holds exactly the ``state`` and ``questions`` the
model receives. Keeping to that schema is what lets the same rows be scored by
either the engine's HTTP path or a direct local scorer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import persona as persona_text
from .conditions import build, instruction
from .io import read_jsonl
from .options import KEYS, describe_cash

BATCH_SIZE = 1000


@dataclass(frozen=True)
class Household:
    index: int
    household_id: str
    asset_bracket: str
    risk_preference: str
    weight: float
    display_order: tuple[str, ...]
    idle_cash: float
    deposit_leg: float
    etf_leg: float
    profile: dict

    @property
    def persona(self) -> str:
        return persona_text.persona(self.profile)

    @property
    def flow(self) -> str:
        return persona_text.income_and_spending(self.profile)


def load_panel(path: Path, *, limit: int | None = None) -> list[Household]:
    """Read the frozen panel, keeping the stored order as the household index."""
    households: list[Household] = []
    for position, record in enumerate(read_jsonl(path)):
        if limit is not None and position >= limit:
            break
        order = tuple(record["display_order"])
        if sorted(order) != list(KEYS):
            raise ValueError(f"household {record['household_id']} has a broken display order")
        households.append(
            Household(
                index=int(record["sample_index"]),
                household_id=record["household_id"],
                asset_bracket=record["asset_bracket"],
                risk_preference=record["risk_preference"],
                weight=float(record["normalized_weight"]),
                display_order=order,
                idle_cash=float(record["assigned_idle_X_cny"]),
                deposit_leg=record["combo_deposit_P_cents"] / 100,
                etf_leg=record["combo_etf_I_cents"] / 100,
                profile=persona_text.from_record(record),
            )
        )
    if not households:
        raise ValueError(f"{path} yielded no households")
    return households


def model_input(household: Household, condition: int) -> dict:
    """The ``state`` and ``questions`` the model scores.

    Option order is the household's own, applied to the criteria mapping. The
    engine preserves insertion order, so this is the presentation the model sees;
    results are mapped back to A–F on export.
    """
    spec = build(condition, household.idle_cash)
    criteria = {key: spec.criteria[key] for key in household.display_order}
    return {
        "state": {
            "人物": household.persona,
            "家庭收支（年度分档）": household.flow,
            "当前事件": spec.event,
            "已知条款": spec.clause,
        },
        "questions": {
            "action": {
                "type": "choice",
                "instructions": instruction(household.idle_cash),
                "criteria": criteria,
            }
        },
    }


def input_digest(payload: dict) -> str:
    """Stable hash of the state and questions, used to prove a row never changed."""
    blob = json.dumps(
        {"state": payload["state"], "questions": payload["questions"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def task_row(household: Household, condition: int, model: str) -> dict:
    payload = model_input(household, condition)
    return {
        "agent_id": f"c{condition:02d}-h{household.index:05d}-{model}",
        "task": {"task_id": f"c{condition:02d}", "model_input": payload, "model": model},
        "meta": {
            "scenario_id": f"c{condition:02d}-h{household.index:05d}",
            "task_id": f"c{condition:02d}",
            "household_id": household.household_id,
            "source_row_index": str(household.index),
            "sample_index": str(household.index),
            "asset_bracket": household.asset_bracket,
            "risk_preference": household.risk_preference,
            "normalized_weight": str(household.weight),
            "display_order": ",".join(household.display_order),
            "condition": str(condition),
            "arm": model,
            "input_sha256": input_digest(payload),
        },
    }


def build_tasks(
    households: list[Household],
    *,
    model: str,
    conditions: tuple[int, ...],
    out_dir: Path,
    batch_size: int = BATCH_SIZE,
) -> dict[str, int]:
    """Write one directory of batch files per condition.

    Returns ``{condition_file_stem: rows_written}``. Files are named
    ``cNN_0001.jsonl`` so that the engine's ``--batches cNN_`` filter picks up
    exactly the condition it was asked for.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for condition in conditions:
        rows = [task_row(household, condition, model) for household in households]
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            number = start // batch_size + 1
            stem = f"c{condition:02d}_{number:04d}"
            path = out_dir / f"{stem}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in chunk:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written[stem] = len(chunk)
    return written
