from typing import Optional

from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.schedulers import Scheduler 


class Cyclic(Scheduler): 

    def __init__(
            self, 
            optimizer: Optimizer, 
            lr_min: float,
            lr_max: float,
            step_size: float,
            scale_amplitude: bool = False,
            scaling_factor: Optional[float] = None
    ): 
        super().__init__(optimizer, "cyclic")
        self.lr_min = lr_min 
        self.lr_max = lr_max
        self.step_size = step_size
        self.scale_amplitude = scale_amplitude
        self.scaling_factor = scaling_factor 

    def update(self) -> float: 
        t = self.t - 1
        cycle = int((1 + t / (2 * self.step_size)) // 1)
        x = abs(t / self.step_size - 2 * cycle + 1)
        scale = max(0, 1 - x)
        lr = self.lr_min + (self.lr_max - self.lr_min) * scale

        if self.scale_amplitude:
            if self.scaling_factor is None:
                lr = self.lr_min + (lr - self.lr_min) / (2 ** (cycle - 1))
            else:
                lr = self.lr_min + (lr - self.lr_min) * (self.scaling_factor ** t)

        return lr