"""The DCSource class is designed to add a layer of abstraction between spinqick and whichever low speed DACs you are using. We use
these DACs to control steady-state operation of the devices.
"""

from typing import Protocol

from spinqick.helper_functions import file_manager
from spinqick.models import hardware_config_models as hcm
from spinqick.settings import file_settings


class VoltageSource(Protocol):
    source_type: hcm.VoltageSourceType = hcm.VoltageSourceType.test

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

        if voltage_source.source_type == hcm.VoltageSourceType.test:
            # use a fake voltage source to test code that calls a dc source
            print("instantiating... nothing! voltage_source=test.")

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
            self.vsource.set_voltage(ch, volts * conversion)
            self.vsource.close()
        if self.cfg.voltage_source == hcm.VoltageSourceType.test:
            print("test set %s" % volts)

    def set_dc_voltage_compensate(self, volts: float, gate: str, m_gate: str):
        """set gate voltage while compensating m_gate

        :param volts: Voltage in units of volts at the gate
        :param gate: Gate to set voltage on
        :param m_gate: charge sensor gate to compensate with

        """

        cross_coupling_matrix = self.cfg.channels[gate].crosscoupling
        if cross_coupling_matrix is None:
            raise RuntimeError(
                "Can't set compensated voltage without a defined cross-coupling matrix"
            )
        if self.cfg.voltage_source == hcm.VoltageSourceType.slow_dac:
            ### set dc voltage
            v0 = self.get_dc_voltage(gate)
            self.set_dc_voltage(volts, gate)
            ### compensate the m gate
            crosscoupling = cross_coupling_matrix[m_gate]
            mbias = self.get_dc_voltage(m_gate)
            vm = (v0 - volts) * crosscoupling + mbias
            self.set_dc_voltage(vm, m_gate)
        if self.cfg.voltage_source == hcm.VoltageSourceType.test:
            print("test set %s" % volts)

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
            self.vsource.set_sweep(
                ch=ch,
                start=vstart * conversion,
                stop=vstop * conversion,
                length=tstep * nsteps,
                num_steps=nsteps,
            )
            self.vsource.close()
        if self.cfg.voltage_source == hcm.VoltageSourceType.test:
            print("test ramp %s to %s" % (vstart, vstop))

    def program_ramp_compensate(
        self,
        vstart: float,
        vstop: float,
        tstep: float,
        nsteps: int,
        gate: str,
        m_gate: str,
    ):
        """program a fast sweep on DCSource with compensation on a charge sensor gate. Needs to be followed with arm and trigger

        :param vstart: Ramp start voltage in units of volts at the gate
        :param vstop: Ramp end voltage in units of volts at the gate
        :param tstep: time per step in seconds
        :param nsteps: number of steps
        :param gate: Gate to set voltage on
        :param m_gate: charge sensor gate to compensate with
        """
        crosscoupling_matrix = self.cfg.channels[gate].crosscoupling

        if crosscoupling_matrix is None:
            raise RuntimeError(
                "Can not program ramp with compensation without configuring a crosscoupling-matrix"
            )
        if self.cfg.voltage_source == hcm.VoltageSourceType.slow_dac:
            # program gate voltages
            self.program_ramp(vstart, vstop, tstep, nsteps, gate)
            crosscoupling = crosscoupling_matrix[m_gate]
            mbias = self.get_dc_voltage(m_gate)
            gatebias = self.get_dc_voltage(gate)
            vm_start = (gatebias - vstart) * crosscoupling + mbias
            vm_stop = (gatebias - vstop) * crosscoupling + mbias
            # program m-gate voltages
            self.program_ramp(vm_start, vm_stop, tstep, nsteps, m_gate)
            self.vsource.close()
        if self.cfg.voltage_source == hcm.VoltageSourceType.test:
            print("test ramp %s to %s" % (vstart, vstop))

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
            print("fake trigger")

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
            print("fake arm sweep")
            print("fake arm sweep")
