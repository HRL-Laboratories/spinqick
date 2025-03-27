"""enums for commonly used key word arguements within qick functions"""


class Outsel:
    """
    Output select, see:

    https://qick-docs.readthedocs.io/latest/_autosummary/qick.asm_v2.html

    For more details
    """

    PRODUCT = "product"
    DDS = "dds"
    INPUT = "input"
    ZERO = "zero"


class Mode:
    """
    Selects the mode, "oneshot" or "periodic", see:

    https://github.com/openquantumhardware/qick/blob/main/qick_lib/qick/asm_v2.py

    For more details
    """

    ONESHOT = "oneshot"
    PERIODIC = "periodic"


class Stdysel:
    """
    Selects steady state output mode to either output last or return to zero, see:

    https://github.com/openquantumhardware/qick/blob/main/qick_lib/qick/asm_v2.py

    For more details
    """

    LAST = "last"
    ZERO = "zero"


class Waveform:
    """
    Selects waveform envelope for standard types and arb, see:

    https://qick-docs.readthedocs.io/latest/_autosummary/qick.asm_v2.html

    For more details
    """

    ARB = "arb"
    FLAT_TOP = "flat_top"
    CONSTANT = "const"


class Time:
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

    GAIN = 32000
