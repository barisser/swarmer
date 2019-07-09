import numpy as np

class RectModel(object):
    def __init__(self, width, depth, weights=None, activation_function='relu'):
        shape = (depth, width)
        self.shape = shape
        weight_shape = (depth - 1, width, width)
        if weights is None:
            self.weights = np.random.rand(*weight_shape) * 2 - 1.0
        else:
            self.weights = weights
        self.values = np.zeros(shape)
        self.activation_function = activation_function

    def reset_values(self):
        self.values = np.zeros(self.shape)

    def run(self, x):
        self.reset_values()
        self.values[0] = x

        if self.activation_function == 'relu':
            for n, layer in enumerate(self.weights):
                self.values[n+1] = np.dot(self.values[n], layer).clip(min=0)
        else:
            raise Exception("Unknown activation function: {0}".format(self.activation_function))

        return self.values[-1]