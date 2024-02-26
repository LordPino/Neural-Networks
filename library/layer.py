
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate: float, use_rprop: bool):
        # TODO: update parameters and return input gradient
        pass