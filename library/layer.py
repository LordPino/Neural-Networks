
class Layer:
    def __init__(self, neurons: int, activation_function: list):
        self._neurons = neurons
        self.activation_function = activation_function
        self._weight = None
        self._bias = None

    def get_neurons(self) -> int:
        return self._neurons

    def get_weight(self) -> int:
        return self._weight
    
    def set_weight(self, weight: float):
        self._weight = weight

    def get_bias(self) -> int:
        return self._bias
    
    def set_bias(self, bias: float):
        self._bias = bias