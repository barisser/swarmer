import numpy as np

import swarmer

def test_model_basic():
    w = np.ones((2, 5, 5)) * 0.1
    model = swarmer.RectModel(5, 3, weights=w)
    response = model.run([0.1]*5)
    assert np.allclose(response, [0.025] * 5)

def test_model_decay():
    model = swarmer.RectModel(4, 3)
    response = model.run(np.random.rand(4), decay=True)