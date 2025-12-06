from abc import abstractmethod 

import numpy as np  

from deepi.modules import Module 


class Loss(Module): 
    
    def __init__(self, _type: str): 
        super().__init__(f"loss.{_type}", False)
        
    @abstractmethod
    def transform(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray: 
        pass 

    @abstractmethod 
    def gradients(self) -> np.ndarray: 
        pass