import os.path
import time
from concurrent.futures import ThreadPoolExecutor
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

min_len = min(len(negatives), len(positives))
positives = positives[:min_len]
negatives = negatives[:min_len]
_all = positives + negatives

print(len(positives))
print(len(negatives))
print(len(_all))

samples = _all

diff_results = list[dict]()
eval_results = list[dict]()


def kl_diff(kwargs):
    p = kwargs['p']
    q = kwargs['q']
    pass_flag_q = q not in negatives
    _diff = kl_divergence_between_texts(p[1], q[1])
    print(f'diff("{p[0]}", "{q[0]}")={_diff}', flush=True)
    diff_results.append({
        'novel_p': p[0],
        'novel_p_passed': p not in negatives,
        'novel_q': q[0],
        'novel_q_passed': pass_flag_q,
        'kl_divergence(p, q)': _diff
    })
    return q[0], pass_flag_q, _diff


if __name__ == '__main__':
    workers = 36
    for sample in samples:
        pairs = []
        for doc in _all:
            pairs.append({'p': sample, 'q': doc})
        with ThreadPoolExecutor(max_workers=workers) as executor:
            diffs = list(executor.map(kl_diff, pairs))
        diffs.sort(key=lambda x: x[2])
        # 去掉自己
        diffs = diffs[1:]
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
