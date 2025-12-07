import numpy as np
from deepi.modules.linear import Linear 


class LowRank(Linear): 
    
    def __init__(
            self, 
            out_size: int,
            rank: int = 1,
            bias: bool = True
    ): 
        super().__init__("low_rank", bias, out_size)
        self.params["w1"] = (rank,)
        self.params["w2"] = (rank, out_size)

    def transform(self, x: np.ndarray) -> np.ndarray: 
        xw1 = x @ self.params["w1"]
        y = xw1 @ self.params["w2"]

        if self._has_bias: 
            y += self.params["b"]

        return y

    
    def gradients(self, dy: np.ndarray) -> np.ndarray:
        dyw2T = dy @ self.params["w2"].T 
        dy_low_rank = dyw2T @ self.params["w1"].T

        self.grads["w1"] = self.x.T @ dy @ self.params["w2"].T
        self.grads["w2"] = (self.x @ self.params["w1"]).T @ dy

        if self._has_bias: 
            self.grads["b"] = dy.sum(axis=0, keepdims=True)

        return dy_low_rank