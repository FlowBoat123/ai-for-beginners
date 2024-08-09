# base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    # forward progress
    def forward_propagation(self, input):
        raise NotImplementedError

    # backward progress
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
    