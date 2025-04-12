import os


def read_file(path: str, encoding: str = 'utf-8') -> str:
    with open(path, mode='r', encoding=encoding) as f:
        return f.read()


def get_files(path: str) -> list:
    entries = os.listdir(path)
    return [entry for entry in entries if os.path.isfile(os.path.join(path, entry))]
