"""Hardware-free tests for selected exchange-only pulse helpers."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from spinqick.core import eo_pulses
from spinqick.helper_functions import spinqick_enums
from spinqick.models import qubit_models


def test_round_up_pulses_rounds_to_dac_cycle():
    soccfg = Mock()
    soccfg.cycles2us.side_effect = lambda cycles: cycles * 0.1

    result = eo_pulses.round_up_pulses(0.25, soccfg)

    assert result == pytest.approx(0.3)
    assert soccfg.cycles2us.call_args_list[0].args == (1,)


def test_play_pi_plays_pi_and_idle_return_for_coarse_timing():
    program = Mock()
    cfg = SimpleNamespace(
        n=SimpleNamespace(
            gates=SimpleNamespace(x=SimpleNamespace(gen=2, name="exchange"))
        )
    )

    eo_pulses.play_pi(
        program,
        spinqick_enums.ExchangeAxis.N,
        cast(qubit_models.Eo1Qubit, cfg),
        t=4.0,
        t_res="course",
    )

    program.pulse.assert_any_call(2, "exchange_pi_n", t=4.0)
    program.pulse.assert_any_call(2, "exchange_idle_return", t="auto")
    assert program.pulse.call_count == 2


def test_play_pi_omits_idle_return_for_fine_timing():
    program = Mock()
    cfg = SimpleNamespace(
        n=SimpleNamespace(
            gates=SimpleNamespace(x=SimpleNamespace(gen=2, name="exchange"))
        )
    )

    eo_pulses.play_pi(
        program,
        spinqick_enums.ExchangeAxis.N,
        cast(qubit_models.Eo1Qubit, cfg),
        t_res="fine",
    )

    program.pulse.assert_called_once_with(2, "exchange_pi_n", t=0)


def test_clifford_identity_is_zero_rotation():
    assert eo_pulses.CLIFFORDS_1Q["I"] == [0, 0, 0, 0]
