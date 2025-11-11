"""Module to hold the DotExperiment class and config updater function, which are used in all
experiment classes.

This module also holds a set of functions for performing unit conversions on experiment config
models.
"""

import logging
from typing import Literal, TypeVar, Union

import numpy as np

from spinqick import settings
from spinqick.helper_functions import file_manager
from spinqick.models import (
    dcs_model,
    full_experiment_model,
    hardware_config_models,
    ld_qubit_models,
    qubit_models,
    spam_models,
)
from spinqick.settings import file_settings

logger = logging.getLogger(__name__)
G = TypeVar("G", float, np.ndarray)


def convert_dcs_to_rfsoc(
    dcs_config: dcs_model.DcsConfigParams,
    readout_label: Literal["m1_readout", "m2_readout"],
    hardware_config: hardware_config_models.HardwareConfig,
):
    """Convert DcsConfigParams to DcsConfig, which contains gain in rfsoc units and rfsoc channel
    numbers."""
    sd_gen = hardware_config.sd_in.qick_gen
    readout = getattr(hardware_config, readout_label)
    readout_channels = [r.qick_adc for r in readout]
    ac_gen = hardware_config.ac_gate.qick_gen if hardware_config.ac_gate is not None else None
    if dcs_config.ac_gate_gain is not None and hardware_config.ac_gate is not None:
        ac_gain = dcs_config.ac_gate_gain * hardware_config.ac_gate.unit_conversion
    else:
        ac_gain = None
    dcs_dict = dcs_config.model_dump()
    dcs_dict["sd_gen"] = sd_gen
    dcs_dict["ro_chs"] = readout_channels
    dcs_dict["pulse_gain_readout"] = (
        dcs_config.pulse_gain_readout * hardware_config.sd_in.unit_conversion
    )
    dcs_dict["ac_gate_gen"] = ac_gen
    dcs_dict["ac_gate_gain"] = ac_gain
    return dcs_model.DcsConfig(**dcs_dict)


def convert_spam_to_rfsoc(
    spam_config: spam_models.DefaultSpam,
    hardware_config: hardware_config_models.HardwareConfig,
):
    """Convert DefaultSpam to DefaultSpamDac, which contains gain in rfsoc units and rfsoc channel
    numbers."""
    new_dict = {}
    for spamstep in spam_config.model_fields_set:
        duration = getattr(spam_config, spamstep).duration
        gate_list = getattr(spam_config, spamstep).gate_list
        temp_gate_dict: dict[
            settings.GateNames, Union[spam_models.SpamRampDac, spam_models.SpamPulseDac]
        ] = {}
        for gate in gate_list:
            assert isinstance(gate, settings.GateNames)
            gate_cfg = hardware_config.channels[gate]
            if isinstance(gate_cfg, hardware_config_models.FastGate):
                gen = gate_cfg.qick_gen
            else:
                raise Exception("gate %s is not a FastGate" % gate)
            pulse_gain_volts = gate_list[gate].voltage
            dac_units = volts2dacunits(pulse_gain_volts, gate, hardware_config)
            if isinstance(gate_list[gate], spam_models.SpamRamp):
                pulse_gain_volts_2 = gate_list[gate].voltage_2
                dac_units2 = volts2dacunits(pulse_gain_volts_2, gate, hardware_config)
                new_ramp = spam_models.SpamRampDac(
                    gen=gen, coordinate=dac_units, coordinate_2=dac_units2
                )
                temp_gate_dict[gate] = new_ramp
            else:
                new_pulse = spam_models.SpamPulseDac(gen=gen, coordinate=dac_units)
                temp_gate_dict[gate] = new_pulse
        new_spamstep = spam_models.SpamStepDac(duration=duration, gate_list=temp_gate_dict)
        # new_dict[spamstep] = new_spamstep.model_dump()
        new_dict[spamstep] = new_spamstep
    # return spam_models.DefaultSpamDac(**new_dict)
    return spam_models.DefaultSpamDac(
        flush=new_dict["flush"],
        entry_20=new_dict["entry_20"],
        exit_11=new_dict["exit_11"],
        idle=new_dict["idle"],
        entry_11=new_dict["entry_11"],
        meas=new_dict["meas"],
    )


def convert_spam_to_voltage(
    spam_config: spam_models.DefaultSpamDac,
    hardware_config: hardware_config_models.HardwareConfig,
):
    """Convert DefaultSpamDac to DefaultSpam, which contains dac outputs in voltage units and no
    generator numbers."""
    new_dict = {}
    for spamstep in spam_config.model_fields_set:
        duration = getattr(spam_config, spamstep).duration
        gate_list = getattr(spam_config, spamstep).gate_list
        temp_gate_dict: dict[
            settings.GateNames, Union[spam_models.SpamRamp, spam_models.SpamPulse]
        ] = {}
        for gate in gate_list:
            pulse_gain = gate_list[gate].coordinate
            volts = dacunits2volts(pulse_gain, gate, hardware_config)
            if isinstance(gate_list[gate], spam_models.SpamRampDac):
                pulse_gain_2 = gate_list[gate].coordinate_2
                volts2 = dacunits2volts(pulse_gain_2, gate, hardware_config)
                new_ramp = spam_models.SpamRamp(voltage=volts, voltage_2=volts2)
                temp_gate_dict[gate] = new_ramp
            else:
                new_point = spam_models.SpamPulse(voltage=volts)
                temp_gate_dict[gate] = new_point
        new_spamstep = spam_models.SpamStep(duration=duration, gate_list=temp_gate_dict)
        #     new_dict[spamstep] = new_spamstep.model_dump()
        # return spam_models.DefaultSpam(**new_dict)
        new_dict[spamstep] = new_spamstep
    return spam_models.DefaultSpam(
        flush=new_dict["flush"],
        entry_20=new_dict["entry_20"],
        exit_11=new_dict["exit_11"],
        idle=new_dict["idle"],
        entry_11=new_dict["entry_11"],
        meas=new_dict["meas"],
    )


def convert_exchange_gate_to_rfsoc(
    exchange_gate: qubit_models.ExchangeGateParams,
    hardware_config: hardware_config_models.HardwareConfig,
):
    """Convert ExchangeGateParams to ExchangeGate, which contains gain in rfsoc units and rfsoc
    channel numbers."""
    idle_gain = volts2dacunits(
        exchange_gate.gate_voltages.idle_voltage, exchange_gate.name, hardware_config
    )
    exchange_gain = volts2dacunits(
        exchange_gate.gate_voltages.exchange_voltage,
        exchange_gate.name,
        hardware_config,
    )
    eo_gains = qubit_models.ExchangeGains(idle_gain=idle_gain, exchange_gain=exchange_gain)
    gate_config = hardware_config.channels[exchange_gate.name]
    assert isinstance(gate_config, hardware_config_models.FastGate)
    gen = gate_config.qick_gen
    return qubit_models.ExchangeGate(name=exchange_gate.name, gains=eo_gains, gen=gen)


def convert_exchange_axis_to_rfsoc(
    exchange_axis: qubit_models.ExchangeAxisConfig,
    hardware_config: hardware_config_models.HardwareConfig,
):
    """Convert ExchangeAxisConfig voltages to gain in rfsoc units and add rfsoc channel numbers."""
    gates = ["px", "py", "x"]
    gatemap = {}
    for gate in gates:
        ex_gate: qubit_models.ExchangeGateParams = getattr(exchange_axis.gates, gate)
        # assert isinstance(ex_gate, qubit_models.ExchangeGateParams)
        gatemap[gate] = convert_exchange_gate_to_rfsoc(ex_gate, hardware_config)

    new_gatemap = qubit_models.ExchangeGateMap(**gatemap)
    return qubit_models.ExchangeAxisConfig(
        gates=new_gatemap,
        detuning_vector=exchange_axis.detuning_vector,
        exchange_vector=exchange_axis.exchange_vector,
        symmetric_vector=exchange_axis.symmetric_vector,
        times=exchange_axis.times,
        exchange_cal=exchange_axis.exchange_cal,
    )


def convert_experiment_to_rfsoc(
    experiment_config: full_experiment_model.ExperimentConfig,
    hardware_config: hardware_config_models.HardwareConfig,
):
    """Convert experiment config to rfsoc units and add rfsoc channel numbers."""
    m1 = experiment_config.m1_readout
    m2 = experiment_config.m2_readout
    m1_rfsoc = convert_dcs_to_rfsoc(m1, "m1_readout", hardware_config)
    m2_rfsoc = convert_dcs_to_rfsoc(m2, "m2_readout", hardware_config)
    if experiment_config.qubit_configs is not None:
        q_cfg: dict[
            str,
            Union[
                ld_qubit_models.Ld1Qubit,
                qubit_models.Eo1Qubit,
                full_experiment_model.Ro1Qubit,
            ],
        ] = {}
        for qubit_name, qubit in experiment_config.qubit_configs.items():
            spam = qubit.ro_cfg.psb_cfg
            spam_rfsoc = convert_spam_to_rfsoc(spam, hardware_config)
            dcs_cfg = m1_rfsoc if qubit.ro_cfg.measure_dot == "M1" else m2_rfsoc
            readout = spam_models.ReadoutConfig(
                psb_cfg=spam_rfsoc,
                dcs_cfg=dcs_cfg,
                reference=qubit.ro_cfg.reference,
                thresh=qubit.ro_cfg.thresh,
                threshold=qubit.ro_cfg.threshold,
            )
            if qubit.qubit_params is not None:
                if isinstance(qubit.qubit_params, ld_qubit_models.Ld1QubitParams):
                    qubit_rfsoc = ld_qubit_models.Ld1Qubit(
                        gate=qubit.qubit_params.gate,
                        f0=qubit.qubit_params.f0,
                        pulses=qubit.qubit_params.pulses,
                        rf_gen=qubit.qubit_params.rf_gen,
                        ro_cfg=readout,
                    )
                    q_cfg[qubit_name] = qubit_rfsoc
                elif isinstance(qubit.qubit_params, qubit_models.Eo1QubitAxes):
                    eoaxes = qubit.qubit_params.model_dump()
                    eorfsoc = {}
                    for axis, exchange_cfg in eoaxes.items():
                        if exchange_cfg is None:
                            continue
                        else:
                            rfsoc_axis = convert_exchange_axis_to_rfsoc(
                                qubit_models.ExchangeAxisConfig(**exchange_cfg),
                                hardware_config,
                            )
                            eorfsoc[axis] = rfsoc_axis.model_dump()
                    eorfsoc["ro_cfg"] = readout.model_dump()
                    q_cfg[qubit_name] = qubit_models.Eo1Qubit(**eorfsoc)
            else:
                q_cfg[qubit_name] = full_experiment_model.Ro1Qubit(ro_cfg=readout)
    else:
        q_cfg = {}

    rfsoc_model = full_experiment_model.ExperimentConfigRfsoc(
        m1_readout=m1_rfsoc, m2_readout=m2_rfsoc, qubit_configs=q_cfg
    )
    return rfsoc_model


def volts2dacunits(
    volts: G,
    gate: settings.GateNames,
    hardware_config: hardware_config_models.HardwareConfig,
) -> G:
    """Convert voltage out of qick frontend to dac units.

    :param volts: voltage to convert
    :param gate: gate name
    :return: RFSoC output in DAC units
    """
    gate_config = hardware_config.channels[gate]
    try:
        assert isinstance(gate_config, hardware_config_models.FastGate)
    except AssertionError as exc:
        raise Exception("gate %s is not a FastGate" % gate) from exc
    conversion = gate_config.dac_conversion_factor
    return volts * conversion


def dacunits2volts(
    dacunits: G,
    gate: settings.GateNames,
    hardware_config: hardware_config_models.HardwareConfig,
) -> G:
    """Convert dac units to voltage out of RFSoC front end.

    :param dacunits: DAC units
    :param gate: gate name
    :return: voltage
    """
    gate_config = hardware_config.channels[gate]
    try:
        assert isinstance(gate_config, hardware_config_models.FastGate)
    except AssertionError as exc:
        raise Exception("gate %s is not a FastGate" % gate) from exc
    conversion = gate_config.dac_conversion_factor
    return dacunits / conversion


class DotExperiment:
    """Base class for spinqick experiments. Manages the readout and hardware configs, and updates
    them appropriately.

    :param datadir: This overrides the data directory setting in file_settings, if desired.
    :param save_data: Whether to automatically save data after each experiment is run.
    :param plot: Whether to automatically plot experiment data.
    """

    def __init__(
        self,
        datadir: str = file_settings.data_directory,
        save_data: bool = True,
        plot: bool = True,
    ):
        self.datadir = datadir
        self.save_data = save_data
        self.plot = plot
        self.data_path = file_manager.get_data_dir(datadir=datadir)
        file_manager.check_configs_exist()
        self.config_path = file_settings.dot_experiment_config
        self.hardware_path = file_settings.hardware_config
        self.experiment_config_params: full_experiment_model.ExperimentConfig = (
            file_manager.load_config_json(
                file_settings.dot_experiment_config,
                full_experiment_model.ExperimentConfig,
            )
        )
        self.hardware_config: hardware_config_models.HardwareConfig = file_manager.load_config_json(
            self.hardware_path, hardware_config_models.HardwareConfig
        )
        self.dcs: Literal["M1", "M2"] = "M1"  # selects which M gate is being used
        self.qubit: str = ""  # select which qubit the user is currently measuring

    @property
    def experiment_config(self):
        """Experiment config in rfsoc units and including generator numbers."""
        return convert_experiment_to_rfsoc(self.experiment_config_params, self.hardware_config)

    # TODO implement setter

    @property
    def dcs_config(self):
        """Returns dcs config in rfsoc units and including generator numbers."""
        if self.dcs == "M1":
            return convert_dcs_to_rfsoc(
                self.experiment_config_params.m1_readout,
                "m1_readout",
                self.hardware_config,
            )
        else:
            return convert_dcs_to_rfsoc(
                self.experiment_config_params.m2_readout,
                "m2_readout",
                self.hardware_config,
            )

    @property
    def spam_config(self):
        """Retrieves spam config parameters from experiment config."""
        qubit_cfg = self.experiment_config.qubit_configs
        if qubit_cfg is not None:
            s2c = qubit_cfg[self.qubit].ro_cfg.psb_cfg
        else:
            raise Exception("No qubit is defined")
        return s2c

    @property
    def adc_units(self):
        """Returns a list of units."""
        sd_list = (
            self.hardware_config.m1_readout if self.dcs == "M1" else self.hardware_config.m2_readout
        )
        units = [s.adc_units for s in sd_list]
        return units

    @property
    def adc_unit_conversions(self):
        """Returns a list of unit conversions."""
        sd_list = (
            self.hardware_config.m1_readout if self.dcs == "M1" else self.hardware_config.m2_readout
        )
        units = [s.unit_conversion for s in sd_list]
        return units

    def update_local(self):
        """Update local config parameters from hardware and experiment config files."""

        self.hardware_config = file_manager.load_config_json(
            self.hardware_path, hardware_config_models.HardwareConfig
        )
        self.experiment_config_params = file_manager.load_config_json(
            self.config_path, full_experiment_model.ExperimentConfig
        )
        logger.info("updated local params")

    def volts2dac(self, volts: G, gate: settings.GateNames) -> G:
        """Convert voltage out of qick frontend to dac units.

        :param volts: voltage to convert
        :param gate: gate name
        :return: RFSoC output in DAC units
        """
        return volts2dacunits(volts, gate, self.hardware_config)

    def dac2volts(self, dacunits: G, gate: settings.GateNames) -> G:
        """Convert dac units to voltage out of RFSoC front end.

        :param dacunits: DAC units
        :param gate: gate name
        :return: voltage
        """
        return dacunits2volts(dacunits, gate, self.hardware_config)


def updater(func):
    """Call this as a decorator to update config before and after a dot experiment."""

    def wrapper(dot_expt: DotExperiment, *args, **kwargs):
        # pull the parameters from the json file and assign to the local config object
        dot_expt.update_local()
        result = func(dot_expt, *args, **kwargs)
        return result

    return wrapper
