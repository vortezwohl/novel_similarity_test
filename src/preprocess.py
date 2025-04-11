import jieba
import spacy

ner_model = spacy.load("zh_core_web_trf")


def split_words(text: str, stopwords: list) -> list:
    stopwords.extend([x.text for x in ner_model(text).ents if x.label_ in ['PERSON']])
    for w in stopwords:
        text = text.replace(w, '_')
    seg_list = jieba.cut(text, cut_all=False)
    return [w for w in seg_list if w not in stopwords and len(w) > 1]
