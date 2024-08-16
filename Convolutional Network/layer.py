# base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    # forward progress
    def forward(self, input):
        pass

    # backward progress
    def backward(self, output_error, learning_rate):
        pass
    