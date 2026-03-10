import os
import re
import subprocess
import sys
from typing import List, Optional

from PyQt6.QtCore import QThread, pyqtSignal


class TrainRunner(QThread):
    outputLine = pyqtSignal(str)
    errorLine = pyqtSignal(str)
    lrParsed = pyqtSignal(float)
    startedProcess = pyqtSignal(int)
    finishedProcess = pyqtSignal(int)
    LR_PATTERN = re.compile(r"\blr\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    def __init__(self, command: List[str], cwd: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.command = command
        self.cwd = cwd
        self.process = None
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
            except Exception:
                pass

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            if self.process.pid:
                self.startedProcess.emit(self.process.pid)

            if self.process.stdout:
                for line in self.process.stdout:
                    if self._stop_requested:
                        break
                    line = line.rstrip("\r\n")
                    if line:
                        self.outputLine.emit(line)
                        match = self.LR_PATTERN.search(line)
                        if match:
                            try:
                                self.lrParsed.emit(float(match.group(1)))
                            except Exception:
                                pass

            if self.process and self.process.poll() is None and not self._stop_requested:
                self.process.wait()

            exit_code = self.process.returncode if self.process else -1
            self.finishedProcess.emit(exit_code)
        except Exception as e:
            self.errorLine.emit(str(e))
            self.finishedProcess.emit(-1)
