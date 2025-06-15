LOGGING_LEVEL_TRACE = 0
LOGGING_LEVEL_NORMAL = 1
LOGGING_LEVEL_HIGH = 2
LOGGING_LEVEL_NO_LOG = 100

_logging_level = LOGGING_LEVEL_HIGH


def get_logging_level() -> int:
    return _logging_level


def set_logging_level(level: int):
    global _logging_level
    _logging_level = level


class LoggingContext:
    original_level: int
    context_level: int

    def __init__(self, level: int):
        self.original_level = get_logging_level()
        self.context_level = level

    def __enter__(self):
        set_logging_level(self.context_level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_logging_level(self.original_level)
