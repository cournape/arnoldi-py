import numpy as np


def rand_normalized_vector(n, dtype=np.float64):
    """ Create a random normalized vector
    """
    v = np.random.randn(n).astype(dtype)
    v /= np.linalg.norm(v)

    return v
