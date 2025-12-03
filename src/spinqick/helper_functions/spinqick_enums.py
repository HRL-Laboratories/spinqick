"""Helpful functions and enums for commonly used key word arguements within spinqick functions."""

from enum import StrEnum, auto


class AverageLevel(StrEnum):
    INNER = auto()
    OUTER = auto()
    BOTH = auto()


class ExchangeAxis(StrEnum):
    N = auto()
    M = auto()
    Z = auto()


class GateNames(StrEnum):
    """Modify this to suit the way you label your system."""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"
    P6 = "P6"
    X1 = "X1"
    X2 = "X2"
    X3 = "X3"
    X4 = "X4"
    X5 = "X5"
    T1 = "T1"
    T6 = "T6"
    T3 = "T3"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    Z1 = "Z1"
    Z2 = "Z2"
    Z3 = "Z3"
    Z4 = "Z4"
    M1 = "M1"
    M2 = "M2"
    SG = "SG"
    IFG = "IFG"
    OFG = "OFG"
    HEMT1 = "HEMT1"
    HEMT2 = "HEMT2"
    SD = "SD"
    DVDD = "DVDD"
    AVDD = "AVDD"
    AVSS = "AVSS"
    DVSS = "DVSS"
    SW_LGC = "SW_LGC"
    TEST = "test"


class GateTypes(StrEnum):
    """Strenum for labeling the purpose of each gate."""

    MEASURE = auto()
    TUNNEL = auto()
    EXCHANGE = auto()
    PLUNGER = auto()
    AUX = auto()
