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

SLICE = 2

for sample in samples[:SLICE]:
    diffs = []
    for doc in _all[:SLICE]:
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
    _split = min(len(positives), len(negatives))
    pp(diffs[:_split])
    print(f'pass_rate("{sample[0]}")={len([x for x in diffs[:_split] if x[1]])/float(_split)}')

pd.DataFrame(diff_results).to_csv(f'output/diff_results_{time.time()}.csv')
