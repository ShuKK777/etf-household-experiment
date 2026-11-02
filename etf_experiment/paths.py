"""Every filesystem location the experiment needs, resolved from the environment.

Nothing else in the package hardcodes an absolute path. Set ``ETF_ROOT`` to the
checkout and ``MAS_ENGINE`` to the engine; both have defaults that work when the
engine sits next to this repository, which is how the original runs were laid
out::

    /workspace/ETF/                     <- ETF_ROOT
    /workspace/MAS_Engine_Microservice/ <- MAS_ENGINE

The Python interpreters are separate on purpose: the engine runs in its own
virtualenv, and the encoder checkpoint needs a CUDA build of torch that the
engine's venv does not have.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default.resolve()


def _env_python(name: str, default: Path) -> Path:
    """A virtualenv entry point, or the current interpreter if the venv is absent.

    Falling back keeps `etf-experiment tables` usable on a laptop with only the
    aggregate CSVs and no engine installed.
    """
    raw = os.environ.get(name)
    if raw:
        return Path(raw).expanduser().resolve()
    return default if default.is_file() else Path(os.sys.executable)


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
    root = _env_path("ETF_ROOT", Path(__file__).resolve().parents[1])
    engine = _env_path("MAS_ENGINE", root.parent / "MAS_Engine_Microservice")
    return Paths(
        root=root,
        engine=engine,
        models=_env_path("ETF_MODELS", root / "models"),
        data=_env_path("ETF_DATA", root / "data"),
        inputs=_env_path("ETF_INPUTS", root / "inputs"),
        tasks=_env_path("ETF_TASKS", root / "tasks"),
        results=_env_path("ETF_RESULTS", root / "results"),
        logs=_env_path("ETF_LOGS", root / "logs"),
        figures=_env_path("ETF_FIGURES", root / "figures"),
        engine_python=_env_python("ETF_ENGINE_PYTHON", engine / ".venv/bin/python"),
        gpu_python=_env_python("ETF_GPU_PYTHON", root.parent / "MAS-Fin/.venv-gpu/bin/python"),
        qwen_python=_env_python("ETF_QWEN_PYTHON", root.parent / "RLCD/.venv/bin/python"),
    )


# Base ports. Each lane adds its index, so lane 0 gets 18100/50100/18200.
SCORER_PORT = 18100
BRAIN_PORT = 50100
WORLD_PORT = 18200
