"""Tests for small hardware-independent core utilities."""

from spinqick.core.utils import check_nyquist


def test_check_nyquist_uses_first_zone_at_or_below_half_sampling_rate():
    soccfg = {"gens": {0: {"fs": 1000.0}}}

    assert check_nyquist(500.0, 0, soccfg) == 1


def test_check_nyquist_uses_second_zone_above_half_sampling_rate():
    soccfg = {"gens": {0: {"fs": 1000.0}}}

    assert check_nyquist(500.001, 0, soccfg) == 2
