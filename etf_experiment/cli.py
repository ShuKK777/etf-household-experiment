#!/usr/bin/env python3
"""Command line entry point.

Each subcommand is one stage of the study and reads only files the previous stage
wrote, so stages can be rerun independently and an archived run can be
re-analysed without a GPU.

    etf-experiment tasks    render the panel into engine task rows
    etf-experiment run      start the eight lanes for one model
    etf-experiment tables   rebuild the three published tables
    etf-experiment figures  redraw the report figures
    etf-experiment check    verify the panel and a results directory
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .conditions import CONDITIONS
from .groups import from_records
from .io import read_jsonl
from .options import IDLE_X, KEYS
from .paths import get_paths
from .tasks import build_tasks, load_panel


def cmd_tasks(args: argparse.Namespace) -> int:
    paths = get_paths()
    panel = Path(args.panel) if args.panel else paths.sample
    households = load_panel(panel, limit=args.limit)
    conditions = tuple(args.conditions) if args.conditions else CONDITIONS
    written = build_tasks(
        households,
        model=args.model,
        conditions=conditions,
        out_dir=Path(args.out) if args.out else paths.tasks,
        batch_size=args.batch_size,
    )
    print(json.dumps({"households": len(households), "files": len(written),
                      "rows": sum(written.values())}, ensure_ascii=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .launch import LaunchConfig, main as launch_main

    paths = get_paths()
    config = LaunchConfig(
        paths=paths,
        model=args.model,
        model_dir=Path(args.model_dir) if args.model_dir else paths.model_dir(args.model),
        tasks=Path(args.tasks) if args.tasks else paths.tasks,
        lanes_dir=Path(args.lanes) if args.lanes else paths.root / "lanes" / args.model,
        results_dir=Path(args.results) if args.results else paths.results / args.model,
        log_prefix=args.log_prefix or "",
        kind=args.kind,
        package=Path(args.package) if args.package else None,
        upstream=Path(args.upstream) if args.upstream else None,
    )
    return launch_main(config, expected_files=args.expected_files)


def cmd_tables(args: argparse.Namespace) -> int:
    from .tables import build_all

    paths = get_paths()
    summary = build_all(
        Path(args.results),
        Path(args.panel) if args.panel else paths.sample,
        Path(args.out),
        households=args.households,
        schedule=args.schedule,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from .figures import draw_all

    paths = get_paths()
    written = draw_all(Path(args.tables), paths.figures_dir)
    print(json.dumps({"figures": written}, ensure_ascii=False))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Cheap invariants that catch a broken panel or a half-finished run."""
    paths = get_paths()
    panel = Path(args.panel) if args.panel else paths.sample
    records = read_jsonl(panel)
    problems: list[str] = []

    if len(records) != args.households:
        problems.append(f"panel has {len(records)} households, expected {args.households}")

    seen_order = {tuple(record["display_order"]) for record in records}
    if any(sorted(order) != list(KEYS) for order in seen_order):
        problems.append("a display order is not a permutation of A–F")

    groups = from_records(records)
    for bracket in IDLE_X["new"]:
        if not (groups["asset"] == bracket).any():
            problems.append(f"asset bracket {bracket} is empty")

    weights = [float(record["normalized_weight"]) for record in records]
    if abs(sum(weights) - len(weights)) > 1e-6:
        problems.append("sampling weights do not sum to the household count")

    if args.results:
        for condition in CONDITIONS:
            path = Path(args.results) / f"c{condition:02d}.jsonl"
            if not path.is_file():
                problems.append(f"missing {path.name}")
                continue
            rows = read_jsonl(path)
            bad = [row for row in rows if row.get("status") != "ok"]
            if bad:
                problems.append(f"{path.name}: {len(bad)} unresolved rows")
            if len(rows) != args.households:
                problems.append(f"{path.name}: {len(rows)} rows, expected {args.households}")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        return 1
    print(json.dumps({"panel": str(panel), "households": len(records),
                      "results": args.results or None, "status": "ok"}, ensure_ascii=False))
    return 0


def cmd_factorial(args: argparse.Namespace) -> int:
    from .factorial_score import score_cells

    paths = get_paths()
    written = score_cells(
        panel=paths.sample,
        model_dir=Path(args.model_dir) if args.model_dir else paths.model_dir("sharp-322m"),
        out_dir=Path(args.out) if args.out else paths.results / "factorial",
        gpu=args.gpu,
        only=args.only,
    )
    print(json.dumps({"scored": len(written), "cells": written[:8]}, ensure_ascii=False))
    return 0


def cmd_factorial_tables(args: argparse.Namespace) -> int:
    import numpy as np

    from . import factorial_estimate as estimate
    from .factorial_score import load_cells
    from .io import write_csv

    paths = get_paths()
    cells = load_cells(Path(args.cells))
    factorial_cells = [cell for cell in cells if cell["spec"].kind == "factorial"]
    extra = {cell["spec"].kind: cell for cell in cells if cell["spec"].kind != "factorial"}
    if len(factorial_cells) != 162:
        raise SystemExit(f"expected 162 factorial cells, found {len(factorial_cells)}")

    records = read_jsonl(paths.sample)
    asset = np.array([record["asset_bracket"] for record in records])
    debt = np.array([record["profile"]["extended.debt_amount_bracket"] for record in records])
    risk = np.array([record["profile"]["extended.risk_preference"] for record in records])
    people = {
        "asset": asset,
        "low_asset": asset == "b1",
        "debt_any": ~np.isin(debt, ["none", "unknown"]),
        "no_risk": risk == "none",
        "weight": np.array([float(record["normalized_weight"]) for record in records]),
    }

    out = Path(args.out)
    marginal = estimate.marginal_rows(factorial_cells, people)
    placebo = estimate.extra_rows(extra, factorial_cells, people)
    logit = estimate.logit_rows(factorial_cells, people)
    # The factorial tables were published with CRLF; keeping that terminator makes a
    # rebuild produce no diff at all.
    for name, rows in (("边际效应.csv", marginal), ("安慰剂与标签.csv", placebo),
                       ("条件logit.csv", logit)):
        write_csv(out / name, rows, newline="\r\n")
    print(json.dumps({"marginal": len(marginal), "placebo": len(placebo),
                      "logit": len(logit)}, ensure_ascii=False))
    return 0


def cmd_probes(args: argparse.Namespace) -> int:
    from .probes_score import score_all

    paths = get_paths()
    written = score_all(
        panel=paths.sample,
        model_dir=Path(args.model_dir) if args.model_dir else paths.model_dir("sharp-322m"),
        out_dir=Path(args.out) if args.out else paths.results / "probes",
        inputs=Path(args.inputs) if args.inputs else paths.inputs,
        by_order=Path(args.by_order) if args.by_order else paths.results / "by_order",
        gpu=args.gpu,
        names=args.only,
    )
    print(json.dumps({"scored": len(written), "probes": written}, ensure_ascii=False))
    return 0


def cmd_probe_tables(args: argparse.Namespace) -> int:
    import numpy as np

    from .io import read_jsonl as read
    from .probes_score import write_tables

    paths = get_paths()
    records = read(paths.sample)
    weights = np.array([float(record["normalized_weight"]) for record in records])
    summary = write_tables(Path(args.probes), weights, Path(args.out))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def cmd_teacher(args: argparse.Namespace) -> int:
    from . import teacher
    from .io import write_csv

    paths = get_paths()
    records = read_jsonl(paths.sample)
    order, cube, names = teacher.load_teacher(Path(args.teacher))
    rows = teacher.compare(order, cube, names, Path(args.cells), records)
    write_csv(Path(args.out) / "教师对照.csv", rows)
    print(json.dumps({"rows": len(rows), "sign_hits": teacher.summary(rows)}, ensure_ascii=False))
    return 0


def cmd_pdf(args: argparse.Namespace) -> int:
    from .report_pdf import render

    paths = get_paths()
    source = Path(args.source) if args.source else paths.root / "报告" / "实验汇总.md"
    output = Path(args.out) if args.out else paths.root / "报告" / "实验汇总.pdf"
    print(json.dumps({"pages": render(source, output), "out": str(output)}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="etf-experiment", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tasks", help="render panel + conditions into task rows")
    p.add_argument("--panel")
    p.add_argument("--out")
    p.add_argument("--model", default="sharp-322m")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--limit", type=int)
    p.add_argument("--conditions", type=int, nargs="*")
    p.set_defaults(func=cmd_tasks)

    p = sub.add_parser("run", help="start the engine lanes for one model")
    p.add_argument("--model", required=True)
    p.add_argument("--model-dir")
    p.add_argument("--tasks")
    p.add_argument("--lanes")
    p.add_argument("--results")
    p.add_argument("--log-prefix")
    p.add_argument("--kind", choices=("laya", "qwen"), default="laya")
    p.add_argument("--package", help="decision-head package for a qwen lane")
    p.add_argument("--upstream", help="upstream base model for a qwen lane")
    p.add_argument("--expected-files", type=int, default=150)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("tables", help="rebuild the three published tables")
    p.add_argument("--results", required=True)
    p.add_argument("--panel")
    p.add_argument("--out", required=True)
    p.add_argument("--households", type=int, default=10_000)
    p.add_argument("--schedule", choices=("new", "old"), default="new",
                   help="idle-cash calibration the results were run under")
    p.set_defaults(func=cmd_tables)

    p = sub.add_parser("figures", help="redraw the report figures")
    p.add_argument("--tables", required=True)
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("check", help="verify the panel and optionally a results dir")
    p.add_argument("--panel")
    p.add_argument("--results")
    p.add_argument("--households", type=int, default=10_000)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("factorial", help="score the 162-cell factorial")
    p.add_argument("--model-dir")
    p.add_argument("--out")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--only", type=int, nargs="*", help="cell indices; default is all 164")
    p.set_defaults(func=cmd_factorial)

    p = sub.add_parser("factorial-tables", help="marginal effects, label arm, conditional logit")
    p.add_argument("--cells", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_factorial_tables)

    p = sub.add_parser("probes", help="score the wording and position probes")
    p.add_argument("--model-dir")
    p.add_argument("--out")
    p.add_argument("--inputs")
    p.add_argument("--by-order")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--only", nargs="*", help="probe names; default is all twenty")
    p.set_defaults(func=cmd_probes)

    p = sub.add_parser("probe-tables", help="wording and position tables")
    p.add_argument("--probes", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_probe_tables)

    p = sub.add_parser("teacher-table", help="27B against the small model on 1,000 households")
    p.add_argument("--teacher", required=True)
    p.add_argument("--cells", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_teacher)

    p = sub.add_parser("pdf", help="render the report Markdown to PDF")
    p.add_argument("--source")
    p.add_argument("--out")
    p.set_defaults(func=cmd_pdf)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
