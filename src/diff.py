import numpy as np
from src.lda_model import predict_topics

topic_dict_cache = dict()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.float32:
    if not isinstance(p, np.ndarray):
        p = np.asarray(p)
    if not isinstance(q, np.ndarray):
        q = np.asarray(q)
    q = np.where(q == 0, 1e-10, q)
    return np.sum(p * np.log(p / q))


def kl_divergence_between_texts(text_a: str, text_b: str) -> np.float32:
    global topic_dict_cache
    topic_dist_a, topic_dist_b = topic_dict_cache.get(text_a, predict_topics(text_a)), topic_dict_cache.get(text_b, predict_topics(text_b))
    if text_a not in topic_dict_cache.keys():
        topic_dict_cache[text_a] = topic_dist_a
    if text_b not in topic_dict_cache.keys():
        topic_dict_cache[text_b] = topic_dist_b
    return kl_divergence(topic_dist_a, topic_dist_b)
