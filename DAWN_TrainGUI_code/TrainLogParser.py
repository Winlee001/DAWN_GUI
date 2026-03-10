import os
import re
from typing import Optional, Tuple


DATASET_RE = re.compile(r"number of training examples\s+(\d+)")
EPOCH_RE = re.compile(
    r"Epoch:\s*\[(\d+)\/(\d+)\/(\d+)\].*?Current_epoch_avg_loss:([0-9.]+)"
)
LR_RE = re.compile(r"lr:\s*([0-9.eE+-]+)")


class TrainLogParser:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self._fp = None
        self._position = 0

    def open(self):
        if not self.log_file:
            return
        if not os.path.exists(self.log_file):
            return
        if self._fp is None:
            self._fp = open(self.log_file, "r", encoding="utf-8", errors="ignore")
            self._fp.seek(self._position)

    def close(self):
        if self._fp:
            try:
                self._fp.close()
            finally:
                self._fp = None

    def read_new_lines(self):
        if not self.log_file:
            return []
        if not os.path.exists(self.log_file):
            return []
        if self._fp is None:
            self.open()
            if self._fp is None:
                return []
        self._fp.seek(self._position)
        lines = self._fp.readlines()
        self._position = self._fp.tell()
        return [line.strip() for line in lines if line.strip()]

    @staticmethod
    def parse_dataset_size(line: str) -> Optional[int]:
        match = DATASET_RE.search(line)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def parse_epoch_loss(line: str) -> Optional[Tuple[int, int, int, float]]:
        match = EPOCH_RE.search(line)
        if match:
            epoch = int(match.group(1))
            total = int(match.group(2))
            best = int(match.group(3))
            loss = float(match.group(4))
            return epoch, total, best, loss
        return None

    @staticmethod
    def parse_lr(line: str) -> Optional[float]:
        match = LR_RE.search(line)
        if match:
            return float(match.group(1))
        return None
