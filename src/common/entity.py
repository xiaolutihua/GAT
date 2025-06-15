import sys
from typing import TYPE_CHECKING
from common.id_object import IDObject
from common.logging import LOGGING_LEVEL_TRACE, LOGGING_LEVEL_NORMAL, LOGGING_LEVEL_HIGH, get_logging_level

if TYPE_CHECKING:
    from common.sim_env import SimEnv


class Entity(IDObject):
    env: "SimEnv"
    is_traced: bool

    def __init__(self, env: "SimEnv", transient: bool = False):
        super().__init__()
        self.env = env
        if not transient:
            self.env.entities.append(self)
        self.is_traced = False

    def start(self):
        pass

    def log(self, *args, level: int = LOGGING_LEVEL_NORMAL, **kwargs):
        self._log(level=level, *args, **kwargs)

    def trace(self, *args, **kwargs):
        self._log(level=LOGGING_LEVEL_TRACE, *args, **kwargs)

    def prompt(self, *args, **kwargs):
        self._log(level=LOGGING_LEVEL_HIGH, *args, **kwargs)

    def _log(self, *args, level: int, **kwargs):
        if level >= get_logging_level() or self.is_traced:
            # noinspection PyProtectedMember
            func_name = sys._getframe(2).f_code.co_name
            print(f'{self.env.now:<10.2f} {self.id:16} {func_name:32}', *args, **kwargs)
            if self.is_traced:
                print(self._trace_str())

    def _trace_str(self) -> str:
        return f"正在追踪 {self.id}"
