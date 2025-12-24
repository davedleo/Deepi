from math import cos, pi

from deepi.optimization.optimizers import Optimizer
from deepi.optimization.schedulers import Scheduler


class CosineAnnealingWarmRestarts(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        lr_min: float = 0.0
    ):
        super().__init__(optimizer, "cosine_annealing_warm_restarts")
        self.T_0 = T_0
        self.T_mult = T_mult
        self.lr_min = lr_min
        self.T_i = T_0
        self.T_cur = 0

    def update(self):
        lr = self.optimizer.get_lr()
        self.T_cur += 1
        if self.T_cur > self.T_i:
            self.T_cur = 1
            self.T_i *= self.T_mult
        lr_new = self.lr_min + 0.5 * (lr - self.lr_min) * (1.0 + cos(pi * (self.T_cur - 1) / self.T_i))
        return lr_new
