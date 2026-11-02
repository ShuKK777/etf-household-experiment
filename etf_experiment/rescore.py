"""Re-score households with their own option order, keeping pre-softmax scores.

This exists for one reason: the engine's batch scorer once scored a whole window
under the *first* request's option order and discarded the logits. Because every
household has its own display order, that silently permuted probabilities. This pass
groups only identical questions, so each household is scored under the order stored
in its own task, and keeps the pre-softmax scores the position probe needs.

It is not on the main path. The launchers avoid the batching mode that triggered the
problem, so a fresh run does not need this; it is kept so that archived results can be
rebuilt and so the fix is documented in runnable form rather than in a commit message.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .direct import encoder, outcome_arrays
from .io import read_jsonl, write_jsonl


def _group_by_question(rows: list[dict]) -> dict[str, list[dict]]:
    """One bucket per distinct (state-template, criteria, instructions) triple.

    Households in the same bucket can share a forward pass because the question is
    byte-identical; households in different buckets must not, which is exactly what
    the batch scorer got wrong.
    """
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        payload = row["task"]["model_input"]
        criteria = payload["questions"]["action"]["criteria"]
        key = "|".join(criteria.keys()) + "||" + "|".join(criteria.values()) \
            + "||" + payload["questions"]["action"]["instructions"]
        buckets.setdefault(key, []).append(row)
    return buckets


def rescore(
    *,
    inputs: Path,
    out_dir: Path,
    model_dir: Path,
    gpu: int = 0,
    batch_size: int = 64,
    conditions: tuple[int, ...] = tuple(range(1, 16)),
) -> dict[int, int]:
    """Rewrite one condition file per input, with per-household option order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[int, int] = {}
    with encoder(model_dir, gpu) as agent:
        for condition in conditions:
            path = inputs / f"c{condition:02d}.jsonl"
            if not path.is_file():
                continue
            rows = read_jsonl(path)
            results = []
            for bucket in _group_by_question(rows).values():
                head = bucket[0]
                payload = head["task"]["model_input"] if "task" in head else head
                questions = {"action": payload["questions"]["action"]}
                answers = agent.predict_batch(
                    [item.get("task", item)["model_input"]["state"] for item in bucket],
                    questions,
                    batch_size=batch_size,
                )
                for item, answer in zip(bucket, answers):
                    probabilities, scores, choice = outcome_arrays(answer["answers"]["action"])
                    meta = item.get("meta", {})
                    results.append({
                        "agent_id": meta.get("scenario_id", ""),
                        "task_id": f"c{condition:02d}",
                        "status": "ok",
                        "option_id": choice,
                        "scores": dict(zip("ABCDEF", scores)),
                        "probabilities": dict(zip("ABCDEF", probabilities)),
                        "model": str(model_dir.name),
                        "meta_sample_index": str(meta.get("sample_index", "")),
                        "meta_condition": str(condition),
                        "meta_display_order": meta.get("display_order", ""),
                    })
            results.sort(key=lambda row: int(row["meta_sample_index"]))
            write_jsonl(out_dir / f"c{condition:02d}.jsonl", results)
            written[condition] = len(results)
    return written


def scores_matrix(rows: list[dict], households: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """Probability and score matrices in panel order, for downstream tables."""
    probabilities = np.full((households, 6), np.nan)
    scores = np.full((households, 6), np.nan)
    for row in rows:
        index = int(row["meta_sample_index"])
        probabilities[index] = [float(row["probabilities"][key]) for key in "ABCDEF"]
        scores[index] = [float(row["scores"][key]) for key in "ABCDEF"]
    if np.isnan(probabilities).any():
        raise ValueError("re-scored file does not cover every household")
    return probabilities, scores
