import networkx as nx
import numpy as np

class RectModel(object):
    def __init__(self, shape):
        self.shape = shape
        width, depth = shape
        weight_shape = (width, width, depth-1)
        self.weights = np.random.rand(*weight_shape)