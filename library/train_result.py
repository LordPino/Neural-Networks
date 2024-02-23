
from typing import List

import numpy as np


class TrainResult:
    def __init__(self, w: List[np.ndarray], b: List[np.ndarray]):
        self._w = w
        self._b = b
        pass

    def get_weights(self):
        return self._w
    
    def get_biases(self):
        return self._b