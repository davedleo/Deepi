from deepi.optimization.optimizers import Optimizer 
from deepi.optimization.schedulers import Scheduler 


class Plateau(Scheduler): 

    def __init__(
            self, 
            optimizer: Optimizer, 
            factor: float = 0.1, 
            tol: float = 1e-4,
            patience: int = 10
    ): 
        super().__init__(optimizer, "plateau")
        self.factor = factor 
        self.tol = tol
        self.patience = patience 
        self.counter = 0
        self.val_prev = 1e5

    def update(self, val: float) -> float: 
        if abs(val - self.val_prev) < self.tol: 
            self.counter += 1 
        else: 
            self.counter = 0 
        self.val_prev = val
        
        if self.counter < self.patience: 
            return self.optimizer.get_lr()
        else: 
            self.counter = 0
            lr = self.optimizer.get_lr()
            return lr * self.factor
    
    def step(self, val: float): 
        self.t += 1
        lr = self.update(val)
        self.optimizer.load_lr(lr)