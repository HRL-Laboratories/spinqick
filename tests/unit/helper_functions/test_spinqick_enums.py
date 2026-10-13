"""Tests for string-valued spinQICK enums."""

from spinqick.helper_functions import spinqick_enums


def test_auto_string_enum_values_are_lowercase():
    assert spinqick_enums.AverageLevel.INNER.value == "inner"
    assert spinqick_enums.ExchangeAxis.N.value == "n"
    assert spinqick_enums.GateTypes.PLUNGER.value == "plunger"


def test_string_enum_formats_as_its_value():
    assert str(spinqick_enums.GateNames.P1) == "P1"
    assert f"{spinqick_enums.GateTypes.MEASURE}" == "measure"
