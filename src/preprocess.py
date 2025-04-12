import json
from concurrent.futures import ThreadPoolExecutor

import jieba
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI(base_url='https://chatnio.cdreader.vip/v1', max_retries=1024, timeout=1600)
llm = 'qwen-plus'


def llm_invoke(user_prompt: dict, system_prompt: dict) -> list:
    while True:
        try:
            response = openai_client.chat.completions.create(
                model=llm,
                messages=[
                    {'role': 'user', 'content': json.dumps(user_prompt, ensure_ascii=False)},
                    {'role': 'system', 'content': json.dumps(system_prompt, ensure_ascii=False)},
                ],
                temperature=0.1,
                top_p=0.2
            ).choices[0].message.content
            items = json.loads(response[response.find('['): response.find(']') + 1])
            return items
        except json.decoder.JSONDecodeError:
            ...
        except openai.InternalServerError:
            ...
        except openai.AuthenticationError:
            print('Authentication error')


def check(names: list):
    names = [x for x in names if len(x) > 1]
    text = str(names)
    role = '通用社会概念识别算法'
    output_restraint = '你的输出只能在同一行, JSONL格式, 请确保 JSONL 的格式合法.'
    system_prompt = {
        '你是': role,
        '输出数据格式': 'JSONL',
        '输出示例': ['丈夫', '奶奶', '中年妇女', '警察', '总裁', '皇后', '宝宝'],
        '输出格式限制': output_restraint
    }
    prompt = {
        '花名册': text,
        '系统指令设定': system_prompt,
        '任务目标': '请你仔细逐一检查[花名册], 找出文段中所提及的所有**通用社会概念**, '
                    '例如职业(老板, 总裁, 秘书, 助理...等所有职业名称), '
                    '身份(丈夫, 妻子, 老公, 老婆, 哥们...等所有通用身份概念), '
                    '称谓(爸爸, 妈妈, 爷爷, 奶奶, 姐姐, 妹妹, 哥哥, 弟弟...等所有称谓)',
        '限制': '不要提取人名'
    }
    non_names = llm_invoke(prompt, system_prompt)
    # print('These arent names:', non_names)
    for non_name in non_names:
        if non_name in names:
            names.remove(non_name)
    return names


def llm_based_ner(text: str, field: str = '网文') -> list:
    text = text[:20000]
    role = '人名/姓名/家族名识别算法'
    output_restraint = '你的输出只能在同一行, JSONL格式, 请确保 JSONL 的格式合法.'
    system_prompt = {
        '你是': role,
        '领域': field,
        '输出数据格式': 'JSONL',
        '输出示例': ['沈思', '沉婉', '钟元', '景眠', '叶瑾', '楚修', '苏', '陈'],
        '输出格式限制': output_restraint
    }
    prompt = {
        '文段': text,
        '系统指令设定': system_prompt,
        '任务目标': '请你仔细阅读[文段], 找出文段中所提及的所有**人名/姓名**和**家族名**, 仅找出**人名/姓名**和**家族名**, 不涉及**社会概念**和**称谓**. 包括主角, 配角.',
        '提示': '1. 某些人名如"澄映", "舒念", "语歌"等较为拗口甚至和常用词谐音. 请你多加注意.\n'
                '2. 对于一些较为拗口的人名, 请你根据上下文确定其性质, 通常人名会和动词介词相关联. 请不要漏过任何一个的人名!',
        '限制': '对于"陈姨", "铭宝", "吴总"这样的称谓, 你只需要提取 "陈", "铭", "吴" 即可, 不需要涉及通用称谓. 对于 "陈家", "谢家" 则提取其家族姓氏 "陈", "谢".\n'
                '对于"国王", "皇后", "总裁", "妻子", "丈夫", "警察", "中年妇女"等社会概念, 它们并不是人名, 不需要提取.',
    }
    names = llm_invoke(prompt, system_prompt)
    names = check(names)
    # print('Names Found:', names)
    return names


def ner(text: str) -> list:
    names = []
    chunk_size = 1600
    _split = int(len(text) / chunk_size) + 1
    # print('Split:', _split)
    _inputs = []
    for i in range(_split):
        _inputs.append(text[i*chunk_size:(i+1)*chunk_size])
    with ThreadPoolExecutor(max_workers=_split) as executor:
        tmp_names = list(executor.map(llm_based_ner, _inputs))
    for _names in tmp_names:
        names.extend(_names)
    return list(set(names))


def split_words(text: str, stopwords: list) -> list:
    # known_names = ner(text)
    known_names = []
    for w in stopwords:
        text = text.replace(w, '_')
    for n in known_names:
        text = text.replace(n, '_')
    seg_list = jieba.cut(text, cut_all=False)
    return [w for w in seg_list if len(w) > 1 and '_' not in w and not w.isnumeric()]
