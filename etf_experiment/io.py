"""Reading and writing the experiment's plain-text formats.

Every artefact is either newline-delimited JSON (per-household rows) or CSV
(aggregate tables). Both are read with the same tolerant helpers so a
half-written file from an interrupted run fails loudly instead of silently
producing a short table.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number} is not valid JSON: {error}") from error
    return rows


def iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_csv(path: Path, rows: list[dict], *, newline: str = "\n") -> None:
    """Write a CSV with an explicit line terminator.

    ``csv.writer`` defaults to ``\\r\\n``. The two table families are not consistent in
    the published outputs -- the factorial tables are CRLF, the probe tables are LF --
    so the terminator is a parameter rather than a constant. Reproducing each family
    in its own convention keeps a rebuild a byte-for-byte no-op instead of a diff that
    rewrites every line.
    """
    if not rows:
        raise ValueError(f"refusing to write an empty table to {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
