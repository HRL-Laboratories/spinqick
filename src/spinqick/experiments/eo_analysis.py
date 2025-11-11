"""Exchange only analysis and helper functions."""

import logging

import memspectrum
import numpy as np
import pylab as plt
from scipy import optimize, signal

from spinqick.core import spinqick_data
from spinqick.helper_functions import plot_tools

logger = logging.getLogger(__name__)


def define_fingerprint_vectors(
    px_points: np.ndarray,
    py_points: np.ndarray,
    idle_point: np.ndarray,
    x_point: float,
):
    """Define the detuning and exchange axes based on nonequilibrium cell parameters .

    :param px_points: [Px1, Px2] format, defines the start and endpoints of a line being used to
        define detuning vector
    :param py_points: [Py1, Py2]
    :param x_point: x value used during nonequilibrium cell sweep.
    """
    delta_px = px_points[1] - px_points[0]
    delta_py = py_points[1] - py_points[0]
    detuning_raw = np.array([delta_px, delta_py, 0])
    detuning = detuning_raw / np.linalg.norm(detuning_raw)
    midpoint = np.array([px_points[0] + delta_px / 2, py_points[0] + delta_py / 2])
    symmetric_raw = np.array([midpoint[0], midpoint[1], x_point]) - np.array(
        [idle_point[0], idle_point[1], idle_point[2]]
    )
    symmetric = symmetric_raw / (x_point - idle_point[2])
    return detuning, symmetric


def calculate_fingerprint_gate_vals(detuning, x, detuning_vector, symmetric_vector, idle_point):
    """Calculate individual gate voltages given detuning, x-gate gain, and detuning and symmetric
    vectors."""
    vector_out = np.zeros(3)
    vector_out = detuning_vector * detuning + symmetric_vector * x + idle_point
    return vector_out


def course_cal_function(theta, afit, bfit, theta_max):
    """fit function used for course cal, from https://doi.org/10.1038/s41565-019-0500-4"""
    return -2 * afit * np.log((theta_max - theta) / theta_max / np.sqrt(theta)) + bfit


def course_cal(threshed_data, volts_data, n_pulses):
    """Perform the course calibration procedure."""
    d_filt = signal.savgol_filter(threshed_data, 10, 2)
    mval = np.max(d_filt) - (np.max(d_filt) - np.min(d_filt)) / 2
    find_pk_data = np.abs(d_filt - mval)
    maxpts, _ = signal.find_peaks(find_pk_data, width=None, prominence=0.1)
    angle_array = np.pi * (1 + np.arange(len(maxpts))) / n_pulses
    v_array = volts_data[maxpts]
    p0_array = threshed_data[maxpts]
    # pylint: disable-next=unbalanced-tuple-unpacking
    popt, _ = optimize.curve_fit(
        course_cal_function, angle_array, v_array, p0=[0.1, 1, angle_array[-1] * 2]
    )
    best_fit = course_cal_function(angle_array, *popt)
    fit_dict = {"A": popt[0], "B": popt[1], "theta_max": popt[2]}
    return angle_array, d_filt, best_fit, popt, v_array, p0_array, fit_dict, maxpts


def course_cal_fit(sqd: spinqick_data.PsbData, n_pulses: int, x_gate: str):
    """Course calibration fit, appends data to the spinqick data object."""
    threshed_data = sqd.threshed_data
    assert threshed_data is not None
    data = threshed_data[0]
    xvolts = sqd.axes["x"]["sweeps"][x_gate]["data"]
    angle_array, d_filt, best_fit, popt, v_array, p0_array, fit_dict, _ = course_cal(
        data, xvolts, n_pulses
    )
    sqd.add_fit_params(fit_dict, best_fit=d_filt, fit_axis="x")
    return angle_array, d_filt, best_fit, popt, v_array, p0_array


def process_fine_cal(
    theta_array: np.ndarray,
    voltage_array: np.ndarray,
    data_array: np.ndarray,
    n_pulses: int,
    timestamp: int,
    plot: bool = True,
):
    """fitting procedure for fine calibration from https://doi.org/10.1038/s41565-019-0500-4"""
    mesa = memspectrum.MESA()
    mesa.solve(data_array, method="standard", optimisation_method="FPE")
    extension_length = 5000
    n_sim = 200
    m1 = mesa.forecast(data_array[::-1], length=extension_length, number_of_simulations=n_sim)
    m2 = mesa.forecast(data_array, length=extension_length, number_of_simulations=n_sim)
    extended_data = np.concatenate(
        [sum([m[::-1] for m in m1]) / n_sim, data_array, sum(m2) / n_sim]
    )
    signal_freq = theta_array[-1] / len(data_array) / np.pi / 2 * n_pulses
    sos_filt = signal.butter(1, [signal_freq / 2, signal_freq * 4], "bandpass", output="sos")
    filtered_extended = signal.sosfiltfilt(sos_filt, extended_data)
    fignums = []
    if plot:
        fig1 = plot_tools.plot1_simple(
            theta_array,
            filtered_extended[extension_length : extension_length + len(data_array)]
            + np.mean(data_array),
            timestamp,
            dset_label="fit",
        )
        plot_tools.plot1_simple(
            theta_array, data_array, timestamp, dset_label="data", new_figure=False
        )
        plt.xlabel("estimated theta (rad)")
        plt.ylabel("singlet probability")
        plt.legend()
        first_fignum = fig1.number
        fignums.append(first_fignum)
    filtered_transformed = signal.hilbert(filtered_extended)
    theta_fit = (
        np.unwrap(
            np.angle(filtered_transformed)[extension_length : extension_length + len(data_array)]
        )
        / n_pulses
    )
    if plot:
        fig2 = plot_tools.plot1_simple(
            voltage_array, theta_fit, timestamp, dset_label="new theta vals"
        )
        plt.plot(voltage_array, theta_array, "o", label="initial theta vals")
        plt.yscale("log")
        plt.xlabel("x gate voltage (V)")
        plt.ylabel("theta (rad)")
        plt.legend()
        fignum = fig2.number
        fignums.append(fignum)
    return fignums, theta_fit


def fine_cal_voltage(
    theta: float,
    theta_list: list,
    voltage_list: list,
    afit: float,
    bfit: float,
    theta_max: float,
):
    """finecal interpolation function from https://doi.org/10.1038/s41565-019-0500-4"""
    theta_list_append = [theta_list[0]] + theta_list + [theta_list[-1]]
    mask = [
        np.abs(theta - theta_list_append[i])
        < (theta_list_append[i + 1] - theta_list_append[i - 1]) / 2
        for i in np.arange(1, len(theta_list) + 1)
    ]
    theta_array = np.array(theta_list)
    voltage_array = np.array(voltage_list)
    t_range = theta_array[mask]
    v_range = voltage_array[mask]
    alpha = (theta - t_range[0]) / (t_range[1] - t_range[0])
    exp_i = np.exp((bfit - v_range[0]) / (2 * afit))
    exp_j = np.exp((bfit - v_range[1]) / (2 * afit))
    f_inv_i = (theta_max * (-1 * exp_i + np.sqrt(exp_i**2 + 4 / theta_max)) / 2) ** 2
    f_inv_j = (theta_max * (-1 * exp_j + np.sqrt(exp_j**2 + 4 / theta_max)) / 2) ** 2
    theta_adj = (1 - alpha) * f_inv_i + alpha * f_inv_j
    vfinal = course_cal_function(theta_adj, afit, bfit, theta_max)
    print(t_range[0])
    print(v_range[0])
    return vfinal
