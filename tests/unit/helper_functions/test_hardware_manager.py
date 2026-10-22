"""Hardware-free tests for DC voltage control and safety checks."""

import pytest

from spinqick.helper_functions import hardware_manager, spinqick_enums


@pytest.fixture
def dc_source(monkeypatch, hardware_config, fake_voltage_source):
    monkeypatch.setattr(
        hardware_manager.file_manager,
        "load_hardware_config",
        lambda: hardware_config,
    )
    return hardware_manager.DCSource(fake_voltage_source)


def test_set_dc_voltage_applies_conversion_and_closes_source(
    dc_source, fake_voltage_source
):
    dc_source.set_dc_voltage(0.25, spinqick_enums.GateNames.P1)

    assert ("set_voltage", 1, 0.5) in fake_voltage_source.calls
    assert fake_voltage_source.is_open is False


def test_get_dc_voltage_applies_inverse_conversion_and_closes_source(
    dc_source, fake_voltage_source
):
    fake_voltage_source.voltages[1] = 0.5

    result = dc_source.get_dc_voltage(spinqick_enums.GateNames.P1)

    assert result == pytest.approx(0.25)
    assert fake_voltage_source.is_open is False


@pytest.mark.parametrize("requested", [0.500001, float("inf"), float("nan")])
def test_set_dc_voltage_rejects_unsafe_values(
    dc_source, fake_voltage_source, requested
):
    with pytest.raises(ValueError):
        dc_source.set_dc_voltage(requested, spinqick_enums.GateNames.P1)

    assert not any(call[0] == "set_voltage" for call in fake_voltage_source.calls)
    assert fake_voltage_source.is_open is False


def test_program_ramp_rejects_stop_above_limit(dc_source, fake_voltage_source):
    with pytest.raises(ValueError):
        dc_source.program_ramp(
            vstart=0.0,
            vstop=0.500001,
            tstep=0.001,
            nsteps=10,
            gate=spinqick_enums.GateNames.P1,
        )

    assert fake_voltage_source.sweeps == []


def test_program_ramp_uses_converted_endpoints(dc_source, fake_voltage_source):
    dc_source.program_ramp(
        vstart=0.1,
        vstop=0.2,
        tstep=0.001,
        nsteps=10,
        gate=spinqick_enums.GateNames.P1,
    )

    assert fake_voltage_source.sweeps == [(1, 0.2, 0.4, 0.01, 10)]
    assert fake_voltage_source.is_open is False


def test_calculate_compensated_voltage_identity_matrix(dc_source):
    gates, values, matrix = dc_source.calculate_compensated_voltage(
        delta_v=[0.2],
        gates=[spinqick_enums.GateNames.P1],
        iso_gates=[spinqick_enums.GateNames.P2],
    )

    assert gates == [spinqick_enums.GateNames.P1, spinqick_enums.GateNames.P2]
    assert values.tolist() == pytest.approx([0.2, 0.0])
    assert matrix.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_calculate_compensated_voltage_requires_crosscoupling(
    dc_source, hardware_config
):
    hardware_config.channels[spinqick_enums.GateNames.P2].crosscoupling = None
    dc_source.cfg = hardware_config

    with pytest.raises(RuntimeError, match="cross-coupling"):
        dc_source.calculate_compensated_voltage(
            delta_v=[0.2],
            gates=[spinqick_enums.GateNames.P1],
            iso_gates=[spinqick_enums.GateNames.P2],
        )
