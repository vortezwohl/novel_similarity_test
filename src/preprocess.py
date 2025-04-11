import jieba


def split_words(text: str, stopwords: list) -> list:
    seg_list = jieba.cut(text, cut_all=False)
    return [w for w in seg_list if w not in stopwords and len(w) > 1]
