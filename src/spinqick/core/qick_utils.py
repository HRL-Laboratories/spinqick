"""helpful functions and enums for commonly used key word arguements within qick functions"""

from enum import StrEnum
from qick import QickConfig


def check_nyquist(freq: float, gen: int, soccfg: QickConfig) -> int:
    """returns the correct nqz setting for a given generator and frequency"""
    gen_fs = soccfg["gens"][gen]["fs"]
    if freq > gen_fs / 2:
        nqz = 2
    else:
        nqz = 1
    return nqz


class Outsel(StrEnum):
    """
    Output select, see:

    https://qick-docs.readthedocs.io/latest/_autosummary/qick.asm_v2.html

    For more details
    """

    PRODUCT = "product"
    DDS = "dds"
    INPUT = "input"
    ZERO = "zero"


class Mode(StrEnum):
    """
    Selects the mode, "oneshot" or "periodic", see:

    https://github.com/openquantumhardware/qick/blob/main/qick_lib/qick/asm_v2.py

    For more details
    """

    ONESHOT = "oneshot"
    PERIODIC = "periodic"


class Stdysel(StrEnum):
    """
    Selects steady state output mode to either output last or return to zero, see:

    https://github.com/openquantumhardware/qick/blob/main/qick_lib/qick/asm_v2.py

    For more details
    """

    LAST = "last"
    ZERO = "zero"


class Waveform(StrEnum):
    """
    Selects waveform envelope for standard types and arb, see:

    https://qick-docs.readthedocs.io/latest/_autosummary/qick.asm_v2.html

    For more details
    """

    ARB = "arb"
    FLAT_TOP = "flat_top"
    CONSTANT = "const"


class Time(StrEnum):
    """
    Allows for the selection of a indeterminate time for pulse length by selecting auto. See:

    https://qick-docs.readthedocs.io/latest/_autosummary/qick.asm_v1.html#qick.asm_v1.QickProgram.pulse

    For more details
    """

    AUTO = "auto"


class Defaults:
    """
    Some Spinqick default values for convenience
    """

    GAIN = 1.0
    MAX_GAIN_BITS = 32765
