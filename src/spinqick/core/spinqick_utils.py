"""helpful functions and enums for commonly used key word arguements within spinqick functions"""

from enum import StrEnum, auto


class AverageLevel(StrEnum):
    INNER = auto()
    OUTER = auto()
    BOTH = auto()
