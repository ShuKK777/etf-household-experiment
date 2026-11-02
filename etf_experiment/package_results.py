#!/usr/bin/env python3
"""Collect the aggregate CSVs into a publishable results/ directory.

Only aggregates are copied. The per-household jsonl is tens of gigabytes and is
reproducible from the code, so it stays out of the repository; what goes in is the
set of tables that every number in the report is read from, plus a manifest that
records where each one came from and how many rows it has.

Usage::

    python -m etf_experiment.package_results --out results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .io import read_csv


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


COLLECTIONS = {
    # destination -> (source pattern, description)
    "baseline": ("etf_output_results/new/{model}/表/基准选择.csv",
                 "六产品在本条件下的一万户口径选择概率，全体与七种分组"),
    "clause_impact": ("etf_output_results/new/{model}/表/条款影响.csv",
                      "每个条件相对基准的逐户变化，含购买未达成的单列"),
    "heterogeneity": ("etf_output_results/new/{model}/表/人群差异.csv",
                      "按资产、债务、收支、风险偏好、金融知识、信任切分"),
}

EXTRA = {
    "placebo/sentence_contrast.csv": "results/placebo_c06/{model}/表/句子对照.csv",
    "two_tick/change.csv": "results/tick2_main/{model}/表/两拍变化.csv",
    "factorial/marginal_effects.csv": "factorial/outputs/边际效应.csv",
    "factorial/placebo_and_label.csv": "factorial/outputs/安慰剂与标签.csv",
    "probes/gate_and_infeasible.csv": "probes/outputs/门槛与不可行购买.csv",
    "probes/teacher_contrast.csv": "probes/outputs/教师对照.csv",
    "probes/wording_and_rewrite.csv": "probes/outputs/词汇与改写.csv",
    "probes/position.csv": "probes/outputs/位置效应.csv",
}

MODELS = ("sharp-322m", "sharp-421m-mlx", "sharp-4b-head-200k", "sharp-4b")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root: Path = args.root
    out: Path = args.out
    manifest: list[dict] = []

    for name, (pattern, description) in COLLECTIONS.items():
        for model in MODELS:
            source = root / pattern.format(model=model)
            if not source.is_file():
                continue
            destination = out / name / f"{model}.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, destination)
            rows = read_csv(destination)
            manifest.append({
                "file": str(destination.relative_to(out)),
                "rows": len(rows),
                "columns": len(rows[0]),
                "sha256_16": digest(destination),
                "description": description,
                "model": model,
            })

    for destination_name, source_pattern in EXTRA.items():
        for model in MODELS:
            source = root / source_pattern.format(model=model)
            if not source.is_file():
                continue
            stem = Path(destination_name)
            destination = out / stem.parent / f"{stem.stem}_{model}.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, destination)
            rows = read_csv(destination)
            manifest.append({
                "file": str(destination.relative_to(out)),
                "rows": len(rows),
                "columns": len(rows[0]),
                "sha256_16": digest(destination),
                "description": stem.stem,
                "model": model,
            })

    (out / "manifest.json").write_text(
        json.dumps({"tables": manifest, "models": list(MODELS)}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    total = sum(item["rows"] for item in manifest)
    print(json.dumps({"files": len(manifest), "rows": total, "out": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
