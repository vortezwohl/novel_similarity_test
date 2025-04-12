import hashlib


def long_text_hash(text, _slice: int = 20000) -> str:
    text = text[:_slice]
    hash_object = hashlib.md5()
    hash_object.update(text.encode('utf-8'))
    return hash_object.hexdigest()
