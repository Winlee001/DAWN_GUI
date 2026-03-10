from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainMetrics:
    epochs: List[int] = field(default_factory=list)
    avg_loss: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    last_epoch: Optional[int] = None
    last_step: Optional[int] = None
    last_loss: Optional[float] = None
    last_lr: Optional[float] = None
    dataset_size: Optional[int] = None

    def append_epoch(self, epoch: int, loss: float, step: Optional[int] = None):
        self.epochs.append(epoch)
        self.avg_loss.append(loss)
        self.last_epoch = epoch
        self.last_loss = loss
        if step is not None:
            self.steps.append(step)
            self.last_step = step

    def append_lr(self, lr: float):
        self.lr.append(lr)
        self.last_lr = lr

    def set_dataset_size(self, dataset_size: int):
        self.dataset_size = dataset_size
