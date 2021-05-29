import os
import hashlib


def _gen_filename(base_dir: str, subsite: str, ext: str, idx: int = None, return_glob=False):
    """
    :param base_dir:
    :param subsite:
    :param ext:
    :param idx:
    :param return_glob:
    :return:
    """
    filename = os.path.join(base_dir, subsite)
    if return_glob:
        filename += "*" + ext
    else:
        filename += str(idx) + ext
    return filename


def hash(input: str):
    hash_obj = hashlib.md5(input.encode())
    return hash_obj.hexdigest()


