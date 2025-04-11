import jieba
import spacy

ner_model = spacy.load("zh_core_web_trf")


def split_words(text: str, stopwords: list) -> list:
    stopwords.extend(extract_names(text))
    for w in stopwords:
        text = text.replace(w, '_')
    seg_list = jieba.cut(text, cut_all=False)
    seg_list = [w for w in seg_list if w not in stopwords and len(w) > 1]
    seg_list = [w for w in seg_list if '_' not in w]
    seg_list = [w for w in seg_list if not w.isnumeric()]
    return seg_list


def extract_names(text: str) -> list:
    names = [x.text for x in ner_model(text).ents if x.label_ in ['PERSON']]
    return names
