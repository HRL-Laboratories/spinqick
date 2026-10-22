"""Compatibility helpers for supported Python versions."""

import sys
from enum import Enum

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # Python 3.10 compatibility

    class StrEnum(str, Enum):
        """Backport the standard-library behavior used by spinQICK enums."""

        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

        def __str__(self):
            return self.value
