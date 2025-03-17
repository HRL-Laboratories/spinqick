"""The DCSource class is designed to add a layer of abstraction between spinqick and whichever low speed DACs you are using. We use
these DACs to control steady-state operation of the devices.
"""

from typing import Protocol
import yaml
import numpy as np
import logging

from spinqick.helper_functions import file_manager
from spinqick.models import hardware_config_models as hcm
from spinqick.settings import file_settings

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

    def set_sweep(
        self, ch: int, start: float, stop: float, length: float, num_steps: int
    ):
        pass

    def trigger(self, ch: int):
        pass

    def arm_sweep(self, ch: int):
        pass


class DCSource:
    """wraps low speed DAC control functions. User supplies a hardware config file with channel to gate mapping and voltage conversion factors"""

    def __init__(self, voltage_source: VoltageSource):
        """initializes the DCSource class based on the voltage source of choice

        :param voltage_source: specify the type of voltage source
        :param datadir: specify the data directory that the hardware config is stored in
        """

        hardware_path = file_settings.hardware_config
        self.cfg = hcm.HardwareConfig(**file_manager.load_config(hardware_path))

        self.vsource = voltage_source
        self.source_type = self.cfg.voltage_source

        if self.source_type == hcm.VoltageSourceType.test:
            # use a fake voltage source to test code that calls a dc source
            logger.info("instantiating... nothing! voltage_source=test.")

    def get_dc_voltage(self, gate: str) -> float:
        """get voltage at gate.  Values are converted via the dc_conversion_factor parameter which is stored in the hardware config

        :param gate: Gate to read voltage from

        :returns: Voltage in units of volts at the gate.
        """
        if self.cfg.voltage_source == hcm.VoltageSourceType.slow_dac:
            address = self.cfg.channels[str(gate)].slow_dac_address
            self.vsource.open(address=address)
            ch = self.cfg.channels[str(gate)].slow_dac_channel
            conversion = self.cfg.channels[gate].dc_conversion_factor
            vreturn = self.vsource.get_voltage(ch) / conversion
            assert isinstance(vreturn, float)
            self.vsource.close()
            return vreturn
        elif self.cfg.voltage_source == hcm.VoltageSourceType.test:
            return 1.0
        else:
            raise KeyError("Unknown voltage_source key")

    def set_dc_voltage(self, volts: float, gate: str):
        """Sets gate voltage

        :param volts: Voltage in units of volts at the gate
        :param gate: Gate to set voltage on
        """
        if self.cfg.voltage_source == hcm.VoltageSourceType.slow_dac:
            address = self.cfg.channels[str(gate)].slow_dac_address
            self.vsource.open(address=address)
            ch = self.cfg.channels[str(gate)].slow_dac_channel
            conversion = self.cfg.channels[gate].dc_conversion_factor
            vset = volts * conversion
            if volts > self.cfg.channels[str(gate)].max_v:
                raise Exception(
                    "requested voltage would exceed max_v on gate %s" % gate
                )
            else:
                self.vsource.set_voltage(ch, vset)
                self.vsource.close()
        if self.cfg.voltage_source == hcm.VoltageSourceType.test:
            logger.info("test set %s" % volts)

    def calculate_compensated_voltage(
        self, delta_v: list[float], gates: list[str], iso_gates: list[str]
    ):
        # generate crosscoupling matrix
        gate_list = gates + iso_gates
        g_dim = len(gate_list)
        g_array = np.eye(g_dim)
        voltage_vector = np.zeros((g_dim))
        voltage_vector[: len(gates)] = delta_v
        for i, gate_i in enumerate(gate_list):
            if gate_i in gates:
                continue
            for k, gate_k in enumerate(gate_list):
                if gate_i == gate_k:
                    continue
                else:
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
        volts: float | list[float],
        gates: str | list[str],
        iso_gates: str | list[str],
    ):
        """set gate voltage while compensating m_gate

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

        if self.cfg.voltage_source == hcm.VoltageSourceType.slow_dac:
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
        if self.cfg.voltage_source == hcm.VoltageSourceType.test:
            logger.info("test set %s" % volts)

    def program_ramp(
        self, vstart: float, vstop: float, tstep: float, nsteps: int, gate: str
    ):
        """program a fast sweep on DCSource.  Needs to be followed with arm and trigger

        :param vstart: Ramp start voltage in units of volts at the gate
        :param vstop: Ramp end voltage in units of volts at the gate
        :param tstep: time per step in seconds
        :param nsteps: number of steps
        :param gate: Gate to set voltage on

        """
        if self.cfg.voltage_source == hcm.VoltageSourceType.slow_dac:
            address = self.cfg.channels[str(gate)].slow_dac_address
            self.vsource.open(address=address)
            ch = self.cfg.channels[str(gate)].slow_dac_channel
            conversion = self.cfg.channels[gate].dc_conversion_factor
            vstart_converted = vstart * conversion
            if vstart > self.cfg.channels[str(gate)].max_v:
                raise Exception(
                    "requested ramp start voltage would exceed max_v on gate %s" % gate
                )
            vstop_converted = vstop * conversion
            if vstart > self.cfg.channels[str(gate)].max_v:
                raise Exception(
                    "requested ramp end voltage would exceed max_v on gate %s" % gate
                )
            else:
                self.vsource.set_sweep(
                    ch=ch,
                    start=vstart_converted,
                    stop=vstop_converted,
                    length=tstep * nsteps,
                    num_steps=nsteps,
                )
                self.vsource.close()
        if self.cfg.voltage_source == hcm.VoltageSourceType.test:
            logger.info("test ramp %s to %s" % (vstart, vstop))

    def program_ramp_compensate(
        self,
        vstart: float,
        vstop: float,
        tstep: float,
        nsteps: int,
        gates: str | list[str],
        iso_gates: str | list[str],
    ):
        """program a fast sweep on DCSource with compensation on a charge sensor gate. Needs to be followed with arm and trigger

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

        if self.cfg.voltage_source == hcm.VoltageSourceType.slow_dac:
            delta_vstart_list = []
            delta_vstop_list = []
            v0_list = []
            for k, gate in enumerate(gates):
                v0 = self.get_dc_voltage(gate)
                v0_list.append(v0)
                delta_v1 = vstart - v0
                delta_vstart_list.append(delta_v1)
                delta_v2 = vstop - v0
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
        if self.cfg.voltage_source == hcm.VoltageSourceType.test:
            logger.info("test ramp %s to %s" % (vstart, vstop))

    def digital_trigger(self, gate: str):
        """trigger the fast sweep on DCSource digitally.

        :param gate: Gate to trigger
        """
        if self.cfg.voltage_source == hcm.VoltageSourceType.slow_dac:
            address = self.cfg.channels[str(gate)].slow_dac_address
            self.vsource.open(address=address)
            ch = self.cfg.channels[str(gate)].slow_dac_channel
            self.vsource.trigger(ch=ch)
            self.vsource.close()
        if self.cfg.voltage_source == hcm.VoltageSourceType.test:
            logger.info("fake trigger")

    def arm_sweep(self, gate: str):
        """arm sleeperdac sweep

        :param gate: Gate to arm sweep on
        """
        if self.cfg.voltage_source == hcm.VoltageSourceType.slow_dac:
            address = self.cfg.channels[str(gate)].slow_dac_address
            self.vsource.open(address=address)
            ch = self.cfg.channels[str(gate)].slow_dac_channel
            self.vsource.arm_sweep(ch=ch)
            self.vsource.close()
        if self.cfg.voltage_source == hcm.VoltageSourceType.test:
            logger.info("fake arm sweep")

    def save_voltage_state(self, file_path: str | None = None):
        """save voltage state of all gates

        :param file_path: Path to save voltage data to. Defaults to the default data directory.
        """
        if self.cfg.voltage_source == hcm.VoltageSourceType.slow_dac:
            voltage_dict = {}
            for gate in self.cfg.channels:
                voltage_dict[gate] = self.get_dc_voltage(gate)
            if not file_path:
                f_path, t_stamp = file_manager.get_new_timestamp()
                file_path = file_manager.get_new_filename(
                    "_dc_state.yaml", f_path, t_stamp
                )
            with open(file_path, "w") as file:
                yaml.dump(voltage_dict, file)

        elif self.cfg.voltage_source == hcm.VoltageSourceType.test:
            logger.info("fake save")
