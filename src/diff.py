import numpy as np
from src.lda_model import predict_topics


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.float32:
    if not isinstance(p, np.ndarray):
        p = np.asarray(p)
    if not isinstance(q, np.ndarray):
        q = np.asarray(q)
    q = np.where(q == 0, 1e-10, q)
    return np.sum(p * np.log(p / q))


def diff_between_texts(text_a: str, text_b: str) -> np.float32:
    topic_dist_a, topic_dist_b = predict_topics(text_a), predict_topics(text_b)
    return kl_divergence(topic_dist_a, topic_dist_b)
