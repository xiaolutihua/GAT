import sys
from typing import TYPE_CHECKING, List

from simpy import Environment
from simpy.core import SimTime
from common.logging import LOGGING_LEVEL_NORMAL, get_logging_level

if TYPE_CHECKING:
    from common.entity import Entity


class SimEnv(Environment):
    entities: "List[Entity]"

    def __init__(self, initial_time: SimTime = 0):
        super().__init__(initial_time)
        self.entities = []

    def start(self):
        for entity in self.entities:
            entity.start()

    def log(self, *args, **kwargs):
        if LOGGING_LEVEL_NORMAL >= get_logging_level():
            # noinspection PyProtectedMember
            func_name = sys._getframe(1).f_code.co_name
            print(f'{self.now:<10.2f} {self.__class__.__name__:16} {func_name:32}', *args, **kwargs)
