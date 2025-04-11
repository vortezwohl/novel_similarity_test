import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.float32:
    if not isinstance(p, np.ndarray):
        p = np.asarray(p)
    if not isinstance(q, np.ndarray):
        q = np.asarray(q)
    q = np.where(q == 0, 1e-10, q)
    return np.sum(p * np.log(p / q))
