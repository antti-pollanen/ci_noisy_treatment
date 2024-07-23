import numpy as np

import data.gp_data_generator as gp_data_generator


def test_create_se_kernel():
    X1 = np.array([0, 0, 0])
    X2 = np.array([1, 0, 0])

    kernel = gp_data_generator.create_se_kernel(X1, X2, alpha=1, scale=1)

    assert np.array_equal(
        kernel,
        np.exp(-np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]) / 2)
        + np.identity(X1.shape[0]) * 0.0000001,
    )
