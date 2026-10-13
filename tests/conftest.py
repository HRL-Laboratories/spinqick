"""Shared hardware-manager fixtures for hardware-free tests."""

import pytest

from spinqick.helper_functions import spinqick_enums
from spinqick.models.hardware_config_models import (
    HardwareConfig,
    SlowGate,
    SourceDrainIn,
)


class FakeVoltageSource:
    """Stateful voltage-source fake; it never opens a real connection."""

    def __init__(self):
        self.is_open = False
        self.calls = []
        self.voltages = {}
        self.sweeps = []

    def open(self, address: str):
        self.is_open = True
        self.calls.append(("open", address))

    def close(self):
        self.is_open = False
        self.calls.append(("close",))

    def get_voltage(self, ch: int) -> float:
        self.calls.append(("get_voltage", ch))
        return self.voltages.get(ch, 0.0)

    def set_voltage(self, ch: int, volts: float):
        self.calls.append(("set_voltage", ch, volts))
        self.voltages[ch] = volts

    def set_sweep(
        self, ch: int, start: float, stop: float, length: float, num_steps: int
    ):
        self.calls.append(("set_sweep", ch, start, stop, length, num_steps))
        self.sweeps.append((ch, start, stop, length, num_steps))

    def trigger(self, ch: int):
        self.calls.append(("trigger", ch))

    def arm_sweep(self, ch: int):
        self.calls.append(("arm_sweep", ch))


@pytest.fixture
def fake_voltage_source():
    return FakeVoltageSource()


@pytest.fixture
def hardware_config():
    gate_type = spinqick_enums.GateTypes.PLUNGER
    return HardwareConfig(
        sd_in=SourceDrainIn(qick_gen=0, unit_conversion=1.0, sd_units="V"),
        m1_readout=[],
        m2_readout=[],
        channels={
            spinqick_enums.GateNames.P1: SlowGate(
                dc_conversion_factor=2.0,
                slow_dac_address="test://voltage-source",
                slow_dac_channel=1,
                max_v=0.5,
                gate_type=gate_type,
            ),
            spinqick_enums.GateNames.P2: SlowGate(
                dc_conversion_factor=1.0,
                slow_dac_address="test://voltage-source",
                slow_dac_channel=2,
                max_v=1.0,
                gate_type=gate_type,
                crosscoupling={spinqick_enums.GateNames.P1: 0.0},
            ),
        },
        voltage_source="test",
    )
