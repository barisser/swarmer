import numpy as np

class RectModel(object):
    def __init__(self, width, depth, weights=None, activation_function='relu'):
        shape = (depth, width)
        self.shape = shape
        self.weight_shape = (depth - 1, width, width)
        if weights is None:
            self.weights = np.random.rand(*self.weight_shape) * 2 - 1.0
        else:
            self.weights = weights
        self.values = np.zeros(shape)
        self.activation_function = activation_function

        self.node_health = np.ones(shape)
        self.edge_health = np.ones(self.weight_shape)


    def reset_values(self):
        self.values = np.zeros(self.shape)


    def run_once(self, x):
        self.reset_values()
        self.values[0] = x

        if self.activation_function == 'relu':
            for n, layer in enumerate(self.weights):
                self.values[n+1] = np.dot(self.values[n].clip(min=0) * self.node_health[n], layer)
        else:
            raise Exception("Unknown activation function: {0}".format(self.activation_function))

        return self.values[-1]


    def mutate(self, rate=0.001, edge_health_threshold=0.02):
        edge_diff = (self.edge_health >= edge_health_threshold).astype(int) * (np.random.rand(*self.weight_shape) * 2.0 - 1.0) * rate
        self.weights += edge_diff
        return edge_diff

