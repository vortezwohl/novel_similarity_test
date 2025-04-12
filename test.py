import os.path

from src import get_files, read_file

positives = []
negatives = []

positives_path = './data/positive'
for filename in get_files(positives_path):
    positives.append((filename, read_file(os.path.join(positives_path, filename))))

negatives_path = './data/negative'
for filename in get_files(negatives_path):
    negatives.append((filename, read_file(os.path.join(negatives_path, filename))))

print(len(positives))
print(len(negatives))
