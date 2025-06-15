from collections import defaultdict

_id_bucket = defaultdict(int)


def next_id(prefix: str) -> int:
    _id_bucket[prefix] += 1
    return _id_bucket[prefix]


def reset_id():
    global _id_bucket
    _id_bucket = defaultdict(int)


class IDObject:
    id: str

    def __str__(self):
        return self.id

    @property
    def id_prefix(self) -> str:
        return self.__class__.__name__

    def __init__(self):
        self.id = f"{self.id_prefix}_{next_id(self.id_prefix)}"
