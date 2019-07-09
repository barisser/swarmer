import networkx as nx
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

    def run(self, x, decay=False):
        self.reset_values()
        self.values[0] = x

        self.edge_passthrough = np.zeros(self.weight_shape)

        if self.activation_function == 'relu':
            for n, layer in enumerate(self.weights):
                self.values[n+1] = np.dot(self.values[n].clip(min=0) * self.node_health[n], layer)
                # weight value i, j is edge weight between 
                # i --> j node ns in layers
                self.edge_passthrough[n] = self.values[n] * layer
        else:
            raise Exception("Unknown activation function: {0}".format(self.activation_function))

        self.node_passthrough = self.values.copy()

        if decay:
            enp = np.exp(np.abs(self.node_passthrough))
            node_health_delta = (enp / enp.mean() - 1.0) * 0.01
            self.node_health += node_health_delta

        return self.values[-1]
