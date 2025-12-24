from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.schedulers import Scheduler 


class OneCycle(Scheduler): 

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        final_div_factor: float = 1e4,
    ):
        super().__init__(optimizer, "one_cycle")
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.final_div_factor = final_div_factor
        self.warmup_steps = int(self.total_steps * self.pct_start)

    def update(self) -> float:
        step = min(self.t, self.total_steps)
        if step <= self.warmup_steps:
            lr = self.lr + (self.max_lr - self.lr) * step / self.warmup_steps
        else:
            decay_steps = self.total_steps - self.warmup_steps
            final_lr = self.lr / self.final_div_factor
            lr = self.max_lr - (self.max_lr - final_lr) * (step - self.warmup_steps) / decay_steps
        return lr