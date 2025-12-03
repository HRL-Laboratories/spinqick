"""The DCSource class is designed to add a layer of abstraction between spinqick and whichever low
speed DACs you are using.

We use these DACs to control steady-state operation of the devices.
"""

import logging
from typing import List, Protocol

import numpy as np
import yaml

from spinqick.helper_functions import file_manager, spinqick_enums
from spinqick.models import hardware_config_models as hcm

logger = logging.getLogger(__name__)


class VoltageSource(Protocol):
    def open(self, address: str):
        pass

    def close(self):
        pass

    def get_voltage(self, ch: int) -> float:
        pass

    def set_voltage(self, ch: int, volts: float):
        pass

    def set_sweep(self, ch: int, start: float, stop: float, length: float, num_steps: int):
        pass

    def trigger(self, ch: int):
        pass

    def arm_sweep(self, ch: int):
        pass


class DummyDCSource:
    def open(self, address: str):
        pass

    def close(self):
        pass

    def get_voltage(self, ch: int) -> float:
        return 0.001

    def set_voltage(self, ch: int, volts: float):
        pass

    def set_sweep(self, ch: int, start: float, stop: float, length: float, num_steps: int):
        pass

    def trigger(self, ch: int):
        pass

    def arm_sweep(self, ch: int):
        pass


class DCSource:
    """Wraps low speed DAC control functions.

    User supplies a hardware config file with channel to gate mapping and voltage conversion factors
    """

    def __init__(self, voltage_source: VoltageSource):
        """Initializes the DCSource class based on the voltage source of choice.

        :param voltage_source: specify the type of voltage source
        """

        self.cfg = file_manager.load_hardware_config()
        self.vsource = voltage_source
        self.source_type = self.cfg.voltage_source

    @property
    def all_voltages(self):
        """Get all slow dac voltages."""
        voltage_dict = {}
        for gate in self.cfg.channels:
            voltage_dict[gate] = self.get_dc_voltage(gate)
        return voltage_dict

    def get_dc_voltage(self, gate: spinqick_enums.GateNames) -> float:
        """Get voltage at gate.  Values are converted via the dc_conversion_factor parameter which
        is stored in the hardware config.

        :param gate: Gate to read voltage from
        :returns: Voltage in units of volts at the gate.
        """

        address = self.cfg.channels[gate].slow_dac_address
        self.vsource.open(address=address)
        ch = self.cfg.channels[gate].slow_dac_channel
        channel_cfg = self.cfg.channels[gate]
        assert not isinstance(channel_cfg, hcm.AuxGate)
        conversion = channel_cfg.dc_conversion_factor
        vreturn = self.vsource.get_voltage(ch) / conversion
        assert isinstance(vreturn, float)
        self.vsource.close()
        return vreturn

    def set_dc_voltage(self, volts: float, gate: spinqick_enums.GateNames):
        """Sets gate voltage.

        :param volts: Voltage in units of volts at the gate
        :param gate: Gate to set voltage on
        """
        address = self.cfg.channels[gate].slow_dac_address
        self.vsource.open(address=address)
        ch = self.cfg.channels[gate].slow_dac_channel
        channel_cfg = self.cfg.channels[gate]
        assert not isinstance(channel_cfg, hcm.AuxGate)
        conversion = channel_cfg.dc_conversion_factor
        vset = volts * conversion
        if volts > channel_cfg.max_v:
            raise Exception("requested voltage would exceed max_v on gate %s" % gate)
        else:
            self.vsource.set_voltage(ch, vset)
            self.vsource.close()

    def calculate_compensated_voltage(
        self,
        delta_v: list[float],
        gates: list[spinqick_enums.GateNames],
        iso_gates: List[spinqick_enums.GateNames],
    ):
        """Given crosscoupling parameters in hardware_config and desired voltages at gates,
        calculate voltages to apply.

        :param delta_v: provide a list of desired potential differences at the gates
        :param gates: list of gates
        :param iso_gates: list of gates whose potential at the gate will be kept constant
        """

        gate_list = gates + iso_gates
        g_dim = len(gate_list)
        g_array = np.eye(g_dim)
        voltage_vector = np.zeros((g_dim))
        voltage_vector[: len(gates)] = delta_v
        for i, gate_i in enumerate(gate_list):
            if gate_i in gates:
                continue
            for k, gate_k in enumerate(gate_list):
                if gate_i != gate_k:
                    gate_dict = self.cfg.channels[gate_i]
                    if isinstance(gate_dict, (hcm.FastGate, hcm.SlowGate)):
                        crosscoupling_matrix = gate_dict.crosscoupling
                        if crosscoupling_matrix is None:
                            raise RuntimeError(
                                "Can't set compensated voltage without a defined cross-coupling matrix"
                            )
                        g_i_k = crosscoupling_matrix[gate_k]
                        g_array[i, k] = g_i_k
        v_apply = np.matmul(np.linalg.inv(g_array), voltage_vector)
        return gate_list, v_apply, g_array

    def set_dc_voltage_compensate(
        self,
        volts: float | List[float],
        gates: spinqick_enums.GateNames | List[spinqick_enums.GateNames],
        iso_gates: spinqick_enums.GateNames | List[spinqick_enums.GateNames],
    ):
        """Set gate voltage while compensating on iso_gates.

        :param volts: Voltage in units of volts at the gate
        :param gates: Gate to set voltage on
        :param iso_gates: charge sensor gate to compensate with
        """

        if not isinstance(volts, list):
            volts = [volts]
        if not isinstance(gates, list):
            gates = [gates]
        if not isinstance(iso_gates, list):
            iso_gates = [iso_gates]

        delta_v_list = []
        v0_list = []
        for k, gate in enumerate(gates):
            v0 = self.get_dc_voltage(gate)
            v0_list.append(v0)
            delta_v = volts[k] - v0
            delta_v_list.append(delta_v)
        for iso_gate in iso_gates:
            v0 = self.get_dc_voltage(iso_gate)
            v0_list.append(v0)
        set_gates, set_delta_v, _ = self.calculate_compensated_voltage(
            delta_v_list, gates, iso_gates
        )
        for i, gate in enumerate(set_gates):
            self.set_dc_voltage(set_delta_v[i] + v0_list[i], gate)

    def program_ramp(
        self,
        vstart: float,
        vstop: float,
        tstep: float,
        nsteps: int,
        gate: spinqick_enums.GateNames,
    ):
        """Program a fast sweep on DCSource.  Needs to be followed with arm and trigger.

        :param vstart: Ramp start voltage in units of volts at the gate
        :param vstop: Ramp end voltage in units of volts at the gate
        :param tstep: time per step in seconds
        :param nsteps: number of steps
        :param gate: Gate to set voltage on
        """

        address = self.cfg.channels[gate].slow_dac_address
        self.vsource.open(address=address)
        ch = self.cfg.channels[gate].slow_dac_channel
        channel_cfg = self.cfg.channels[gate]
        assert not isinstance(channel_cfg, hcm.AuxGate)
        conversion = channel_cfg.dc_conversion_factor
        vstart_converted = vstart * conversion
        if vstart > channel_cfg.max_v:
            raise Exception("requested ramp start voltage would exceed max_v on gate %s" % gate)
        vstop_converted = vstop * conversion
        if vstart > channel_cfg.max_v:
            raise Exception("requested ramp end voltage would exceed max_v on gate %s" % gate)
        else:
            self.vsource.set_sweep(
                ch=ch,
                start=vstart_converted,
                stop=vstop_converted,
                length=tstep * nsteps,
                num_steps=nsteps,
            )
            self.vsource.close()

    def program_ramp_compensate(
        self,
        vstart: float | List[float],
        vstop: float | List[float],
        tstep: float,
        nsteps: int,
        gates: spinqick_enums.GateNames | List[spinqick_enums.GateNames],
        iso_gates: spinqick_enums.GateNames | List[spinqick_enums.GateNames],
    ):
        """Program a fast sweep on DCSource with compensation on a charge sensor gate. Needs to be
        followed with arm and trigger.

        :param vstart: Ramp start voltage in units of volts at the gate
        :param vstop: Ramp end voltage in units of volts at the gate
        :param tstep: time per step in seconds
        :param nsteps: number of steps
        :param gate: Gate to set voltage on
        :param m_gate: charge sensor gate to compensate with
        """

        if not isinstance(gates, list):
            gates = [gates]
        if not isinstance(iso_gates, list):
            iso_gates = [iso_gates]
        if not isinstance(vstart, list):
            if len(gates) > 1:
                vstart = [vstart for v in gates]
            else:
                vstart = [vstart]
        if not isinstance(vstop, list):
            if len(gates) > 1:
                vstop = [vstop for v in gates]
            else:
                vstop = [vstop]

        delta_vstart_list = []
        delta_vstop_list = []
        v0_list = []
        for i, gate in enumerate(gates):
            v0 = self.get_dc_voltage(gate)
            v0_list.append(v0)
            delta_v1 = vstart[i] - v0
            delta_vstart_list.append(delta_v1)
            delta_v2 = vstop[i] - v0
            delta_vstop_list.append(delta_v2)
        for iso_gate in iso_gates:
            v0_i = self.get_dc_voltage(iso_gate)
            v0_list.append(v0_i)
        set_gates, set_delta_vstart, _ = self.calculate_compensated_voltage(
            delta_vstart_list, gates, iso_gates
        )
        set_gates, set_delta_vstop, _ = self.calculate_compensated_voltage(
            delta_vstop_list, gates, iso_gates
        )

        # program gate voltages
        for i, gate in enumerate(set_gates):
            self.program_ramp(
                set_delta_vstart[i] + v0_list[i],
                set_delta_vstop[i] + v0_list[i],
                tstep,
                nsteps,
                gate,
            )

    def digital_trigger(self, gate: spinqick_enums.GateNames):
        """Trigger the fast sweep on DCSource digitally.

        :param gate: Gate to trigger
        """

        address = self.cfg.channels[gate].slow_dac_address
        self.vsource.open(address=address)
        ch = self.cfg.channels[gate].slow_dac_channel
        self.vsource.trigger(ch=ch)
        self.vsource.close()

    def arm_sweep(self, gate: spinqick_enums.GateNames):
        """Arm sleeperdac sweep.

        :param gate: Gate to arm sweep on
        """
        address = self.cfg.channels[gate].slow_dac_address
        self.vsource.open(address=address)
        ch = self.cfg.channels[gate].slow_dac_channel
        self.vsource.arm_sweep(ch=ch)
        self.vsource.close()

    def set_voltage_from_dict(self, v_dict: dict):
        """Set voltage from a dict of gate, voltage pairs."""
        for gate in list(v_dict.keys()):
            self.set_dc_voltage(v_dict[gate], gate)

    def save_voltage_state(self, file_path: str | None = None):
        """Save voltage state of all gates.

        :param file_path: Path to save voltage data to. Defaults to the default data directory.
        """
        voltage_dict = self.all_voltages
        if file_path is None:
            f_path, t_stamp = file_manager.get_new_timestamp()
            file_path = file_manager.get_new_filename("_dc_state.yaml", f_path, t_stamp)
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(voltage_dict, file)
