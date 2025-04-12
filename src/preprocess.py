import json
from concurrent.futures import ThreadPoolExecutor

import jieba
import spacy
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI(base_url='https://chatnio.cdreader.vip/v1', max_retries=1024, timeout=1600)
llm = 'qwen-plus'
ner_model = spacy.load("zh_core_web_sm")


def split_words(text: str, stopwords: list) -> list:
    known_names = ner(text)
    stopwords.extend(known_names)
    for w in stopwords:
        text = text.replace(w, '_')
    seg_list = jieba.cut(text, cut_all=False)
    return [w for w in seg_list if len(w) > 1 and '_' not in w and not w.isnumeric()]


def spacy_based_ner(text: str) -> list:
    names = [x.text for x in ner_model(text).ents if x.label_ in ['PERSON']]
    print('SpaCy NER:', names)
    return names


def llm_based_ner(text: str) -> list:
    text = text[:20000]
    role = '命名实体识别算法'
    output_restraint = '你的输出只能在同一行, JSONL格式, 请确保 JSONL 的格式合法.'
    system_prompt = {
        '你是': role,
        '输出数据格式': 'JSONL',
        '输出示例': ['小王', '小美', '杰克', '丽丽'],
        '输出格式限制': output_restraint
    }
    prompt = {
        '文段': text,
        '系统指令设定': system_prompt,
        '任务目标': '请你仔细阅读[文段], 找出文段中所提及的所有命名实体(角色), 包括主角, 配角.'
    }
    response = openai_client.chat.completions.create(
        model=llm,
        messages=[
            {'role': 'user', 'content': json.dumps(prompt, ensure_ascii=False)},
            {'role': 'system', 'content': json.dumps(system_prompt, ensure_ascii=False)},
        ],
        temperature=0.1,
        top_p=0.2
    ).choices[0].message.content
    names = json.loads(response[response.find('['): response.find(']') + 1])
    print('GPT NER:', names)
    return names


def ner(text: str) -> list:
    names = []
    _split = int(len(text) / 10000)
    _inputs = []
    for i in range(_split):
        _inputs.append(text[i*10000:(i+1)*10000])
    with ThreadPoolExecutor(max_workers=_split) as executor:
        tmp_names = list(executor.map(llm_based_ner, _inputs))
    for _names in tmp_names:
        names.extend(_names)
    return list(set(names))
