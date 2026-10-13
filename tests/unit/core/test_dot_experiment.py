"""Hardware-free tests for DotExperiment lifecycle and conversion helpers."""

from unittest.mock import Mock, call

import pytest

from spinqick.core.dot_experiment import (
    DotExperiment,
    dacunits2volts,
    updater,
    volts2dacunits,
)
from spinqick.helper_functions import spinqick_enums


def make_uninitialized_experiment(backend, *, save_data, plot):
    """Create only the lifecycle state needed by ``finalize``."""
    experiment = object.__new__(DotExperiment)
    experiment.backend = backend
    experiment.save_data = save_data
    experiment.plot = plot
    return experiment


def test_finalize_skips_backend_when_saving_is_disabled():
    backend = Mock()
    experiment = make_uninitialized_experiment(backend, save_data=False, plot=True)

    experiment.finalize(object())

    backend.save.assert_not_called()
    backend.save_plot.assert_not_called()
    backend.close.assert_not_called()


def test_finalize_saves_and_closes_without_plotting():
    backend = Mock()
    backend.save.return_value = "synthetic-handle"
    data = object()
    experiment = make_uninitialized_experiment(backend, save_data=True, plot=False)

    experiment.finalize(data)

    backend.save.assert_called_once_with(data)
    backend.save_plot.assert_not_called()
    backend.close.assert_called_once_with("synthetic-handle")


def test_finalize_saves_requested_figures_and_closes_handle():
    backend = Mock()
    backend.save.return_value = "synthetic-handle"
    experiment = make_uninitialized_experiment(backend, save_data=True, plot=True)

    experiment.finalize(object(), fignums=[1, "figure-2"])

    assert backend.save_plot.call_args_list == [
        call("synthetic-handle", 1),
        call("synthetic-handle", "figure-2"),
    ]
    backend.close.assert_called_once_with("synthetic-handle")


def test_finalize_uses_current_figure_when_no_figure_list_is_given():
    backend = Mock()
    backend.save.return_value = "synthetic-handle"
    experiment = make_uninitialized_experiment(backend, save_data=True, plot=True)

    experiment.finalize(object())

    backend.save_plot.assert_called_once_with("synthetic-handle", None)
    backend.close.assert_called_once_with("synthetic-handle")


def test_updater_refreshes_configuration_before_operation():
    experiment = Mock()
    observed = []

    @updater
    def operation(dot_experiment):
        observed.append(dot_experiment)
        return "complete"

    result = operation(experiment)

    assert result == "complete"
    experiment.update_local.assert_called_once_with()
    assert observed == [experiment]


def test_volts2dacunits_rejects_slow_gate(hardware_config):
    with pytest.raises(Exception, match="not a FastGate"):
        volts2dacunits(0.1, spinqick_enums.GateNames.P1, hardware_config)


def test_dacunits2volts_rejects_slow_gate(hardware_config):
    with pytest.raises(Exception, match="not a FastGate"):
        dacunits2volts(10.0, spinqick_enums.GateNames.P1, hardware_config)
