import os
import pickle
from os import DirEntry
from typing import Any, List
import numpy as np
import torch

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def get_gv_root() -> str:
    return os.path.join(os.path.dirname(__file__), 'binbin/data/task-graph')


def yaml_load(path: str):
    with open(path) as f:
        return yaml.load(f, Loader)


def pickle_load(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def pickle_dump(obj: "Any", path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def print_env_vars():
    for name, value in os.environ.items():
        print("{0}: {1}".format(name, value))


def gcd(a: int, b: int):
    assert a > 0 and b > 0, '入参必须都是正整数'
    return _gcd(a, b)


def _gcd(a: int, b: int) -> int:
    return b if a % b == 0 else _gcd(b, a % b)


def lcm(a: int, b: int) -> int:
    assert a > 0 and b > 0, '入参必须都是正整数'
    return int(a * b / gcd(a, b))


def get_all_gv_files_in_data_dir(data_dir) -> "List[str]":
    gv_files = []
    candidates: "List[DirEntry]" = list(os.scandir(data_dir))
    while len(candidates) > 0:
        candidate = candidates.pop()
        if candidate.is_file() and candidate.path.endswith('.gv'):
            gv_files.append(candidate.path)
        elif candidate.is_dir():
            candidates.extend(os.scandir(candidate))
    return gv_files


def sparse_matrix_2_adj(sparse_matrix):
    max_num = np.max(sparse_matrix)
    adj = torch.zeros([max_num + 1, max_num + 1], dtype=torch.float32)
    for edge in sparse_matrix:
        adj[edge[0]][edge[1]] = 1
    return adj
