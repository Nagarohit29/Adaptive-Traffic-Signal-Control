import csv
from collections.abc import Mapping, Sequence
from pathlib import Path


def read_column(path: Path, key: str) -> list[float]:
    with path.open(newline="") as handle:
        return [float(row[key]) for row in csv.DictReader(handle)]


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
