from abc import abstractmethod 

import numpy as np  

from deepi.modules import Module 


class Flow(Module): 
    
    def __init__(self, _type: str): 
        super().__init__(f"flow.{_type}", False)
        
    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray: 
        pass 

    @abstractmethod 
    def gradients(self, dy: np.ndarray) -> np.ndarray: 
        pass