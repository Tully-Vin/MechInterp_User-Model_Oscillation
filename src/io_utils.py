import json
from pathlib import Path
from typing import Iterable, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def latest_file(path: Path, suffix: str = ".jsonl") -> Path:
    files = sorted(path.glob(f"*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files with suffix {suffix} in {path}")
    return files[0]
