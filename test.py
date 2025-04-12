import os.path
from pprint import pp

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

diffs = []
for sample in samples:
    for doc in _all:
        pass_flag_doc = True
        if doc in negatives:
            pass_flag_doc = False
        _diff = kl_divergence_between_texts(sample[1], doc[1])
        print(f'diff("{sample[0]}", "{doc[0]}")={_diff}')
        diffs.append((doc[0], pass_flag_doc, _diff))
    diffs.sort(key=lambda x: x[2])
    _split = min(len(positives), len(negatives))
    pp(diffs[:_split])
    print(f'pass_rate("{sample[0]}")={len([x for x in diffs[:_split] if x[1]])/float(_split)}')
