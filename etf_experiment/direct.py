"""Scoring directly through the encoder, without the engine's HTTP layer.

Two rounds bypass ``POST /v1/systemone``: the 162-cell factorial and the text
probes. Both score millions of short prompts where a per-request round trip costs
more than the forward pass, and neither needs world state. They call the same
checkpoint and the same option-scoring head, so their probabilities are comparable
to the engine path — but they must be labelled as a different execution path when
reported.

The interesting part is ``capture_scores``. The encoder's public path returns
probabilities but discards the pre-softmax scores, and the position probe needs
those scores to compare "same question, different option order". This wraps
``_decode_answers`` to re-derive them from the same logits, un-permuting with each
question's own option order. Without that, a batched request would score every
household under the first request's order.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def prepare_device(gpu: int) -> None:
    """Pin one process to one GPU before torch is imported.

    These variables must be set before the import, which is why the callers import
    ``laya`` inside the function rather than at module scope.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"


@contextmanager
def encoder(model_dir: Path, gpu: int) -> Iterator[object]:
    """Load the encoder on one GPU and expose pre-softmax scores on every answer."""
    prepare_device(gpu)
    from laya.agent import Agent

    agent = Agent(str(model_dir), device="cuda:0")
    original = agent._decode_answers

    def wrapped(logits, act, items, ids, internal, offset=0, lang=None):
        from laya.agent import _option_logits
        from laya.common import QTYPES, temp_bucket, unpermute_probs

        answers = original(logits, act, items, ids, internal, offset, lang)
        raw_rows = _option_logits(logits, items, offset)
        for position, qid in enumerate(ids):
            question = internal[qid]
            width = len(items[position]["markers"])
            qtype = QTYPES[question["t"]]
            scale = agent.temperature_by_options.get(
                temp_bucket(qtype, width), agent.temperature[qtype]
            )
            scaled = unpermute_probs(raw_rows[position] / scale, question.get("option_order"))
            keys = list(question["crit"].keys())
            answers[qid]["scores"] = {
                key: round(float(value), 6) for key, value in zip(keys, scaled)
            }
        return answers

    agent._decode_answers = wrapped
    yield agent


def ask(agent, state: dict, questions: dict) -> dict:
    """One household, one question. Returns the action answer with probabilities."""
    answer = agent.predict_batch([state], questions, batch_size=1)[0]
    return answer["answers"]["action"]


def outcome_arrays(answer: dict) -> tuple[list[float], list[float], str]:
    """Probability vector, score vector, argmax letter, all in A–F order."""
    return (
        [answer["probabilities"][key] for key in "ABCDEF"],
        [answer.get("scores", {}).get(key, 0.0) for key in "ABCDEF"],
        answer["choice"],
    )
