from typing import Callable

import numpy as np


type ActivationFunction = Callable[[np.ndarray], np.ndarray]
type ActivationDerivate = Callable[[np.ndarray], np.ndarray]

type OutputFunction = Callable[[np.ndarray], np.ndarray]
type OutputDerivate = Callable[[np.ndarray], np.ndarray]