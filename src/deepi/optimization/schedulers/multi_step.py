from typing import List, Dict 

from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.schedulers import Scheduler 


class MultiStep(Scheduler): 

    def __init__(
            self, 
            optimizer: Optimizer, 
            factors: float | List[float] = 0.33, 
            milestones: int | List[int] = 5,
    ): 
        super().__init__(optimizer, "multi_step")
class MultiStep(Scheduler): 

    def __init__(
        self, 
        optimizer: Optimizer, 
        factors: float | list[float] = 0.33, 
        milestones: int | list[int] = 5,
    ): 
        super().__init__(optimizer, "multi_step")

        # Convert single values to lists
        if isinstance(factors, (float, int)):
            self.factors = [float(factors)]
        else:
            self.factors = [float(f) for f in factors]

        if isinstance(milestones, int):
            self.milestones = [milestones]
        else:
            self.milestones = [int(m) for m in milestones]

        # Ensure the lengths match
        if len(self.factors) != len(self.milestones):
            # If single factor but multiple milestones, repeat factor
            if len(self.factors) == 1:
                self.factors = self.factors * len(self.milestones)
            else:
                raise ValueError(
                    f"Number of factors ({len(self.factors)}) must match number of milestones ({len(self.milestones)})"
                )

        # Build milestone → factor dict
        self.factors_d: dict[int, float] = {
            m: f for m, f in zip(self.milestones, self.factors)
        }

    def update(self) -> float: 
        if self.t not in self.milestones:
            return self.optimizer.get_lr() 
        
        lr = self.optimizer.get_lr()
        factor = self.factors_d[self.t]

        return factor * lr