"""Every filesystem location the experiment needs, resolved from the environment.

Nothing in the package hardcodes a machine-specific path. The only defaults are
relative to each other: this checkout's own directory, and an engine checkout
that sits beside it. Point ``ETF_ROOT`` and ``MAS_ENGINE`` at anything else.

The three interpreters are separate on purpose. The engine runs in its own
virtualenv; the encoder checkpoint needs a CUDA build of torch that the engine's
venv does not have; and the Qwen head needs a third environment for peft and
transformers. Each is read from the environment and falls back to the current
interpreter, which is enough for the analysis stages and fails loudly at load
time for the training-free scoring stages if the wrong one is used.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Environment variables, in one place so callers can be grepped for.
ENV_ROOT = "ETF_ROOT"
ENV_ENGINE = "MAS_ENGINE"
ENV_MODELS = "ETF_MODELS"
ENV_DATA = "ETF_DATA"
ENV_INPUTS = "ETF_INPUTS"
ENV_TASKS = "ETF_TASKS"
ENV_RESULTS = "ETF_RESULTS"
ENV_LOGS = "ETF_LOGS"
ENV_FIGURES = "ETF_FIGURES"
ENV_ENGINE_PYTHON = "ETF_ENGINE_PYTHON"
ENV_GPU_PYTHON = "ETF_GPU_PYTHON"
ENV_QWEN_PYTHON = "ETF_QWEN_PYTHON"


def _resolve(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default.resolve()


def _resolve_python(name: str, default: Path | None) -> Path:
    """An interpreter path, or the running interpreter when nothing is configured.

    Falling back to ``sys.executable`` keeps the analysis stages usable on a
    machine that has no engine and no GPU: those stages only need NumPy. A scoring
    stage started with the wrong interpreter fails when it tries to import torch
    or the checkpoint, which is a clearer error than a path that does not exist.
    """
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser().resolve()
    if default is not None and default.is_file():
        return default.resolve()
    return Path(sys.executable).resolve()


@dataclass(frozen=True)
class Paths:
    root: Path
    engine: Path
    models: Path
    data: Path
    inputs: Path
    tasks: Path
    results: Path
    logs: Path
    figures: Path
    engine_python: Path
    gpu_python: Path
    qwen_python: Path

    # ---- derived locations -------------------------------------------------
    @property
    def sample(self) -> Path:
        """The frozen 10,000-household panel with weights and display orders.

        Two layouts exist: the published repository keeps the panel at
        ``data/10000户样本_20261022``, while the original working tree nested it
        under ``data/etf实验``. Resolving both means the same code runs in either
        without an environment override.
        """
        candidates = [
            self.data / "10000户样本_20261022" / "家庭数据.jsonl",
            self.data / "etf实验" / "10000户样本_20261022" / "家庭数据.jsonl",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    @property
    def scorer_dir(self) -> Path:
        """The decision-head adapter we ship, for models the engine cannot load."""
        return Path(__file__).resolve().parent / "scorer"

    def model_dir(self, model: str) -> Path:
        return self.models / model

    def results_for(self, model: str) -> Path:
        """Per-model directory holding the fifteen per-condition jsonl files."""
        return self.results / model

    def tables_for(self, model: str) -> Path:
        return self.results_for(model) / "表"

    @property
    def figures_dir(self) -> Path:
        self.figures.mkdir(parents=True, exist_ok=True)
        return self.figures


def get_paths() -> Paths:
    root = _resolve(ENV_ROOT, Path(__file__).resolve().parents[1])
    engine = _resolve(ENV_ENGINE, root.parent / "MAS_Engine_Microservice")
    return Paths(
        root=root,
        engine=engine,
        models=_resolve(ENV_MODELS, root / "models"),
        data=_resolve(ENV_DATA, root / "data"),
        inputs=_resolve(ENV_INPUTS, root / "inputs"),
        tasks=_resolve(ENV_TASKS, root / "tasks"),
        results=_resolve(ENV_RESULTS, root / "results"),
        logs=_resolve(ENV_LOGS, root / "logs"),
        figures=_resolve(ENV_FIGURES, root / "figures"),
        engine_python=_resolve_python(ENV_ENGINE_PYTHON, engine / ".venv/bin/python"),
        # The encoder scorer needs a CUDA build of torch, which the engine's venv
        # does not have. Without an explicit override this falls back to the
        # running interpreter and the lane fails at import, which is the honest
        # signal that no GPU environment was configured.
        gpu_python=_resolve_python(ENV_GPU_PYTHON, None),
        qwen_python=_resolve_python(ENV_QWEN_PYTHON, None),
    )


# Base ports. Each lane adds its index, so lane 0 gets 18100/50100/18200.
SCORER_PORT = 18100
BRAIN_PORT = 50100
WORLD_PORT = 18200
