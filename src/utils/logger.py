import csv
from pathlib import Path
from typing import Dict, Any


class CSVLogger:
    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.filepath.open("w", newline="", encoding="utf-8")
        self._writer = None

    def log(self, metrics: Dict[str, Any]) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(metrics.keys()))
            self._writer.writeheader()
        self._writer.writerow(metrics)
        self._file.flush()

    def close(self) -> None:
        self._file.close()
