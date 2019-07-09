import time

import numpy as np

import swarmer

def test_model_basic():
    w = np.ones((2, 5, 5)) * 0.1
    model = swarmer.RectModel(5, 3, weights=w)
    response = model.run_once([0.1]*5)
    assert np.allclose(response, [0.025] * 5)

def test_model():
    model = swarmer.RectModel(4, 3)
    x = np.random.rand(4)
    response = model.run_once(x)
    model.mutate()
    assert (response != model.run_once(x)).all()

    response3 = model.run(np.random.rand(1000, 4))
    assert response3.shape == (1000, 4)

def test_perf_simple():
    x = np.random.rand(10, 784)
    start=time.time()
    model = swarmer.RectModel(784, 100)
    print("Initialization took {0}s".format(time.time()-start))
    start = time.time()
    model.run(x)
    print("Run took {0}".format(time.time()-start))
