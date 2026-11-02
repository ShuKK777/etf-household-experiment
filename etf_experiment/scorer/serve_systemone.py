#!/usr/bin/env python3
"""The ``/v1/systemone`` adapter for a frozen Qwen decision-head package.

The engine's Brain posts one household per request. This process scores that
request's own state and questions and drops them; it never assembles a batch or
reuses another household's question. That constraint is what keeps the
per-household display order meaningful.

Paths come from the environment (``HEAD_PACKAGE``, ``UPSTREAM_MODEL``) so the same
files work from any checkout. ``PYTHONPATH`` has to contain this directory for
``qwen_decision`` to import; the launcher sets it.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch.nn.functional as functional

for _name in ("ScalingType", "SwizzleType"):
    if not hasattr(functional, _name):
        setattr(functional, _name, type(_name, (), {}))
if not hasattr(functional, "scaled_grouped_mm"):
    functional.scaled_grouped_mm = lambda *args, **kwargs: None

from qwen_decision import DecisionPredictor

PACKAGE = Path(os.environ["HEAD_PACKAGE"])
UPSTREAM = Path(os.environ["UPSTREAM_MODEL"])
MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen3.5-4b-base-finance-head-200k")
DEVICE = os.environ.get("HEAD_DEVICE", "cuda:0")
PREDICTOR = DecisionPredictor.from_pretrained(PACKAGE, UPSTREAM, device=DEVICE, local_files_only=True)
LOCK = threading.Lock()


def score(state, questions):
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must contain the household's own question")
    answers = {}
    for qid, spec in questions.items():
        if not isinstance(spec, dict) or spec.get("type") != "choice":
            raise ValueError(f"question {qid} is not a choice")
        criteria = spec.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError(f"question {qid} has no criteria")
        row = {
            "state": state,
            "qid": str(qid),
            "question": {
                "instructions": spec.get("instructions", str(qid)),
                "criteria": criteria,
            },
            "candidate_ids": list(criteria),
        }
        # One household, one forward schedule. The lock keeps the backbone
        # on a single request so a later arrival cannot replace this question.
        with LOCK:
            prediction = PREDICTOR.predict(row)
        probabilities = {
            cid: float(value)
            for cid, value in zip(prediction["candidate_ids"], prediction["probabilities"])
        }
        answers[str(qid)] = {
            "type": "choice",
            "choice": prediction["prediction"],
            "probabilities": probabilities,
        }
    return answers


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def _send(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/openapi.json":
            self._send(200, {"openapi": "3.0.0", "info": {"title": MODEL_NAME, "version": "1"}})
            return
        self._send(404, {"detail": "not found"})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/v1/systemone":
            self._send(404, {"detail": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
            answers = score(body.get("state"), body.get("questions"))
        except Exception as exc:
            self._send(422, {"detail": f"{type(exc).__name__}: {exc}"})
            return
        self._send(200, {"model": MODEL_NAME, "answers": answers})


def main():
    port = int(os.environ.get("LAYA_PORT", "18100"))
    host = os.environ.get("SCORER_HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"serving {MODEL_NAME} on {host}:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
