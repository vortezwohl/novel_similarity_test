def read_file(path: str, encoding: str = 'utf-8') -> str:
    with open(path, mode='r', encoding=encoding) as f:
        return f.read()
