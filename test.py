import os.path
import time
from pprint import pp

import pandas as pd

from src import get_files, read_file, kl_divergence_between_texts

positives = []
negatives = []

positives_path = './data/positive'
for filename in get_files(positives_path):
    positives.append((filename, read_file(os.path.join(positives_path, filename))))

negatives_path = './data/negative'
for filename in get_files(negatives_path):
    negatives.append((filename, read_file(os.path.join(negatives_path, filename))))

_all = positives + negatives

print(len(positives))
print(len(negatives))
print(len(_all))

samples = _all

diff_results = list[dict]()
eval_results = list[dict]()

for sample in samples:
    diffs = []
    for doc in _all:
        pass_flag_doc = doc not in negatives
        _diff = kl_divergence_between_texts(sample[1], doc[1])
        print(f'diff("{sample[0]}", "{doc[0]}")={_diff}')
        if sample != doc:
            diffs.append((doc[0], pass_flag_doc, _diff))
        diff_results.append({
            'novel_p': sample[0],
            'novel_p_passed': sample not in negatives,
            'novel_q': doc[0],
            'novel_q_passed': pass_flag_doc,
            'kl_divergence(p, q)': _diff
        })
    diffs.sort(key=lambda x: x[2])
    # _split = min(len(positives), len(negatives))
    _split = 20
    pp(diffs[:_split])
    confidence = len([x for x in diffs[:_split] if x[1]])/float(_split)
    print(f'confidence_of_pass("{sample[0]}")={confidence}')
    eval_results.append({
        'novel': samples[0],
        'novel_passed': sample not in negatives,
        'confidence_of_pass': confidence
    })

agree_count = 0
for eval_res in eval_results:
    if eval_res['novel_passed'] and eval_res['confidence_of_pass'] >= 0.5:
        agree_count += 1

eval_results.append({
        'novel': f'acc={agree_count / len(eval_results)}',
        'novel_passed': '',
        'confidence_of_pass': ''
})

pd.DataFrame(diff_results).to_csv(f'output/diff_results_{time.time()}.csv')
pd.DataFrame(eval_results).to_csv(f'output/eval_results_{time.time()}.csv')
