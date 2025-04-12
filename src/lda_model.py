import os.path

from gensim.corpora import Dictionary
from gensim.models import LdaModel
import numpy as np

from src.preprocess import split_words

num_topics = 12
min_p = 1e-12
cwd = os.path.dirname(os.path.abspath(__file__))

with open(f'{cwd}/../data/stopwords', encoding='utf-8') as _f:
    stopwords = _f.readlines()
    stopwords = [x.replace('\n', '') for x in stopwords]

lda_model = LdaModel.load(f'{cwd}/../model/lda/lda_model')
dictionary = Dictionary.load(f'{cwd}/../model/lda/lda_model.id2word')


def predict_topics(_document: str, _model: LdaModel = lda_model, _dictionary: Dictionary = dictionary,
                   _stopwords: list = None, _num_topics: int = num_topics, _min_p: float = min_p):
    if _stopwords is None:
        _stopwords = stopwords
    topic_dist = _model.get_document_topics(_dictionary.doc2bow(split_words(_document, _stopwords)), minimum_probability=_min_p)
    topic_dist = [x[1] for x in topic_dist]
    np.pad(topic_dist, (0, _num_topics - len(topic_dist)), mode='constant', constant_values=_min_p)
    return topic_dist
