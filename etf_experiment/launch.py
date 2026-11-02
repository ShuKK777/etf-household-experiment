"""Starting the MAS engine lanes for one decision model.

The engine exposes one decision endpoint, ``POST /v1/systemone``. Everything in
this module exists to stand up enough of the engine to call it at volume:

    scorer  (our code, or the engine's laya.serve)  :18100+lane  scores A-F
    brain   mas_engine.service.server               :50100+lane  policy + gRPC
    world   examples.persona_choice.app             :18200+lane  state + tasks
    runner  examples.persona_choice.run_batches     drives the world

Only the scorer is ours for the Qwen models; the other three processes are the
engine's own persona_choice example, unmodified. ``MAS_BRAIN_MEMORY_ENABLED=0``
is what makes each condition independent.

The three port ranges must stay disjoint. They are offset by one per lane, so a
lane's scorer, brain and world are all on different hundreds; the numbers match
the runs already on disk, which is what lets an old ``logs/`` directory line up
with a new launch.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .paths import BRAIN_PORT, SCORER_PORT, WORLD_PORT, Paths, get_paths

LANES = 8
_HEALTH_TIMEOUT = 900


@dataclass
class Lane:
    index: int
    scorer: int
    brain: int
    world: int
    directory: Path

    @property
    def commit_store(self) -> Path:
        return self.directory / "data" / "world-core" / "world-commit.sqlite3"


@dataclass
class LaunchConfig:
    paths: Paths
    model: str
    model_dir: Path
    tasks: Path
    lanes_dir: Path
    results_dir: Path
    log_prefix: str = ""
    kind: str = "laya"
    package: Path | None = None
    upstream: Path | None = None

    # ---- engine settings that differ by scorer -----------------------------
    @property
    def served_name(self) -> str:
        return self.model

    @property
    def parallel_decisions(self) -> int:
        return 24 if self.kind == "laya" else 8

    @property
    def decision_timeout(self) -> int:
        return 120 if self.kind == "laya" else 180

    @property
    def tick_deadline(self) -> int:
        return "900" if self.kind == "laya" else "6000"

    @property
    def extra_env(self) -> dict[str, str]:
        if self.kind == "laya":
            return {
                "DECISION_MODEL_LANG": "zh",
                "DECISION_MODEL_MAX_LEN": "4096",
                "DECISION_MODEL_HEAD_MAX_LEN": "2048",
            }
        return {}


class Launcher:
    def __init__(self, config: LaunchConfig, lanes: int = LANES) -> None:
        self.config = config
        self.lanes = [
            Lane(
                index=i,
                scorer=SCORER_PORT + i,
                brain=BRAIN_PORT + i,
                world=WORLD_PORT + i,
                directory=config.lanes_dir / f"lane{i}",
            )
            for i in range(lanes)
        ]
        self.processes: list[subprocess.Popen] = []

    # ---- environment -------------------------------------------------------
    def base_env(self) -> dict[str, str]:
        engine = self.config.paths.engine
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{engine / 'src'}{os.pathsep}{engine}"
        env["PYTHONUNBUFFERED"] = "1"
        env["MAS_SERVICE_AUTH_REQUIRED"] = "false"
        env["HF_HUB_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["TORCHDYNAMO_DISABLE"] = "1"
        return env

    def spawn(self, cmd: list[str], env: dict[str, str], log_name: str, cwd: Path) -> None:
        log_path = self.config.paths.logs / log_name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8")
        self.processes.append(
            subprocess.Popen(cmd, env=env, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
        )

    # ---- process definitions ----------------------------------------------
    def scorer_command(self, lane: Lane) -> tuple[list[str], dict[str, str], Path]:
        env = self.base_env()
        if self.config.kind == "laya":
            env.update(
                {
                    "LAYA_EXTRA_MODELS": json.dumps({self.config.served_name: str(self.config.model_dir)}),
                    "LAYA_MODELS": self.config.served_name,
                    "LAYA_DEFAULT_MODEL": self.config.served_name,
                    "LAYA_PRELOAD": "1",
                    "LAYA_PORT": str(lane.scorer),
                    "LAYA_MAX_CONCURRENT": "24",
                    "LAYA_DEVICE": "cuda:0",
                    "CUDA_VISIBLE_DEVICES": str(lane.index),
                }
            )
            cmd = [str(self.config.paths.gpu_python), "-m", "laya.serve"]
            cwd = self.config.paths.engine
        else:
            if self.config.package is None or self.config.upstream is None:
                raise ValueError("a qwen lane needs both a head package and an upstream model")
            env.update(
                {
                    "HEAD_PACKAGE": str(self.config.package),
                    "UPSTREAM_MODEL": str(self.config.upstream),
                    "SERVED_MODEL_NAME": self.config.served_name,
                    "HEAD_DEVICE": "cuda:0",
                    "HEAD_MAX_LENGTH": "3072",
                    "LAYA_PORT": str(lane.scorer),
                    "CUDA_VISIBLE_DEVICES": str(lane.index),
                    "PYTHONPATH": str(self.config.paths.root / "head200k"),
                }
            )
            cmd = [str(self.config.paths.qwen_python), str(self.config.paths.root / "head200k/serve_systemone.py")]
            cwd = self.config.paths.root / "head200k"
        return cmd, env, cwd

    def brain_env(self, lane: Lane) -> dict[str, str]:
        env = self.base_env()
        env.update(
            {
                "MAS_BRAIN_PORT": str(lane.brain),
                "MAS_BRAIN_POLICY": "decision",
                "MAS_BRAIN_MEMORY_ENABLED": "0",
                "MAS_BRAIN_ALLOW_IN_MEMORY_RESULT_STORE": "1",
                "MAS_BRAIN_MAX_PARALLEL_DECISIONS": str(self.config.parallel_decisions),
                "MAS_BRAIN_DECISION_TIMEOUT_SECONDS": str(self.config.tick_deadline),
                "MAS_POLICY_ZYGOTE": "1",
                "DECISION_MODEL_PROVIDER": "laya",
                "DECISION_MODEL_BASE_URL": f"http://127.0.0.1:{lane.scorer}",
                "DECISION_MODEL_NAME": self.config.served_name,
                "DECISION_MODEL_TIMEOUT_SECONDS": str(self.config.decision_timeout),
            }
        )
        env.update(self.config.extra_env)
        return env

    def world_env(self, lane: Lane) -> dict[str, str]:
        env = self.base_env()
        env.update(
            {
                "MAS_PERSONA_TASKS_DIR": str(self.config.tasks),
                "MAS_WORLD_CORE_STORE_BACKEND": "sqlite",
                "MAS_BRAIN_GRPC_ADDRESS": f"127.0.0.1:{lane.brain}",
                "MAS_WORLD_CORE_BRAIN_DEADLINE_SECONDS": str(self.config.tick_deadline),
                "MAS_MAX_INITIAL_AGENTS": "1000",
                "MAS_BRAIN_MEMORY_ENABLED": "0",
            }
        )
        return env

    def runner_command(self, lane: Lane) -> list[str]:
        engine_python = str(self.config.paths.engine_python)
        return [
            engine_python, "-m", "examples.persona_choice.run_batches",
            "--world", f"http://127.0.0.1:{lane.world}",
            "--tasks-dir", str(self.config.tasks),
            "--batches", f"lane{lane.index}_",
            "--out", str(self.config.results_dir / f"lane{lane.index}"),
            "--source", str(lane.commit_store),
            "--deadline", str(self.config.tick_deadline),
            "--retries", "1",
        ]

    # ---- orchestration ----------------------------------------------------
    def link_lanes(self, expected_files: int) -> None:
        """Symlink the condition batches onto lane prefixes.

        ``run_batches --batches laneN_`` picks a session's files by prefix, so
        the same condition files are exposed under each lane's prefix instead of
        being copied eight times.
        """
        files = sorted(self.config.tasks.glob("c??_*.jsonl"))
        if len(files) != expected_files:
            raise SystemExit(f"expected {expected_files} task files, found {len(files)}")
        for stale in self.config.tasks.glob("lane*_*.jsonl"):
            stale.unlink()
        for position, path in enumerate(files):
            link = self.config.tasks / f"lane{position % len(self.lanes)}_{path.stem}.jsonl"
            link.symlink_to(path.name)

    def start(self) -> None:
        for lane in self.lanes:
            lane.directory.mkdir(parents=True, exist_ok=True)
            cmd, env, cwd = self.scorer_command(lane)
            self.spawn(cmd, env, f"{self.config.log_prefix}scorer{lane.index}.log", cwd)
            self.spawn(
                [str(self.config.paths.engine_python), "-m", "mas_engine.service.server"],
                self.brain_env(lane),
                f"{self.config.log_prefix}brain{lane.index}.log",
                self.config.paths.engine,
            )
            self.spawn(
                [str(self.config.paths.engine_python), "-m", "uvicorn",
                 "examples.persona_choice.app:create_app", "--factory",
                 "--host", "127.0.0.1", "--port", str(lane.world)],
                self.world_env(lane),
                f"{self.config.log_prefix}world{lane.index}.log",
                lane.directory,
            )
        for lane in self.lanes:
            wait_http(lane.scorer, f"scorer {lane.index}")
        for lane in self.lanes:
            wait_http(lane.world, f"world {lane.index}")
            wait_for_file(lane.commit_store, f"commit store of world {lane.index}")

    def run(self, expected_files: int) -> list[int]:
        self.link_lanes(expected_files)
        self.start()
        runners = []
        for lane in self.lanes:
            runners.append(
                subprocess.Popen(
                    self.runner_command(lane),
                    env=self.base_env(),
                    cwd=self.config.paths.engine,
                    stdout=(self.config.paths.logs / f"{self.config.log_prefix}runner{lane.index}.log").open(
                        "w", encoding="utf-8"
                    ),
                    stderr=subprocess.STDOUT,
                )
            )
        return [process.wait() for process in runners]

    def terminate(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)


def wait_http(port: int, label: str, timeout: int = _HEALTH_TIMEOUT) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                urllib.request.urlopen(f"http://127.0.0.1:{port}/openapi.json", timeout=5)
                return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"{label} never served OpenAPI on port {port}")


def wait_for_file(path: Path, label: str, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if Path(path).is_file():
            return
        time.sleep(0.5)
    raise TimeoutError(f"{label} never appeared at {path}")


def main(config: LaunchConfig, expected_files: int) -> int:
    launcher = Launcher(config)

    def handle(*_: object) -> None:
        launcher.terminate()
        sys.exit(1)

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
    try:
        codes = launcher.run(expected_files)
    finally:
        launcher.terminate()
    return 0 if all(code == 0 for code in codes) else 1
