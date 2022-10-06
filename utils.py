import pandas as pd
import os

from .client import SemRushClient


with open(os.path.dirname(__file__) + r'/column_ref.txt', "r") as f:
    column_ref_string = f.read()
_rows = [_.split('\t') for _ in column_ref_string.split('\n')]
COLUMN_REF = {_r[0]: _r[1] for _r in _rows}


def load_from_key(key: str,
                  default_database: str = "uk") -> SemRushClient:
    return SemRushClient(api_key=key,
                         default_database=default_database)


def load_from_string(string: str,
                     default_database: str = "uk") -> SemRushClient:
    api_key = string.split(":")[1].strip()
    return load_from_key(key=api_key,
                         default_database=default_database)


def load_from_file(file_path: str,
                   default_database: str = "uk"):
    with open(file_path, "r") as f:
        string = f.read()
    return load_from_string(string=string,
                            default_database=default_database)


def column_reference(ref: str):
    return COLUMN_REF.get(ref)
