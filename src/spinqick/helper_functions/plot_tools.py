"""
functions to help with plotting data output from averager functions
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_decimated(iq_list, config, IQ=False):
    """Plot output of acquire_decimated()."""
    fig = plt.figure()
    readout_length = config["readout_length"]
    time = np.linspace(
        0, readout_length / config["adc_sr"], int(readout_length)
    )  # [us]

    for ii, iq in enumerate(iq_list):
        iq = np.transpose(iq)  # RAveragerProgram
        plt.plot(
            time, np.abs(iq[0] + 1j * iq[1]), label="mag, ADC %d" % (config["ro_ch"])
        )
        if IQ:
            plt.plot(time, iq[0], label="I value, ADC %d" % (config["ro_ch"]))
            plt.plot(time, iq[1], label="Q value, ADC %d" % (config["ro_ch"]))

    plt.legend()
    plt.xlabel(r"time ($\mu s$)")
    plt.ylabel("ADC units")
    return fig


def interpret_data_DCS(avgi, avgq, data_dim="2D"):
    """Helper for evaluating NDAverager data."""

    # magnitude using i and q signals, use difference between reference (singlet) and second measurement
    mag = np.sqrt(avgi[0] ** 2 + avgq[0] ** 2)

    # average over the number of shots at each point, unless there is only 1 shot per point
    if len(mag.shape) > 2 and data_dim == "2D":
        avged_mag = np.mean(mag, axis=0)
    elif len(mag.shape) > 1 and data_dim == "1D":
        avged_mag = np.mean(mag, axis=0)
    else:
        avged_mag = mag  # each point is a single shot

    return avged_mag


def interpret_data_PSB(avgi, avgq, data_dim="2D", thresh=None):
    """Helper for evaluating NDAverager data from experiments that use a reference measurement."""

    # magnitude using i and q signals, use difference between reference (singlet) and second measurement
    mag = np.sqrt(avgi[0][0] ** 2 + avgq[0][0] ** 2) - np.sqrt(
        avgi[0][1] ** 2 + avgq[0][1] ** 2
    )

    if thresh is not None:
        mag = np.where(np.abs(mag) > thresh, 0, 1)

    # average over the number of shots at each point, unless there is only 1 shot per point
    if len(mag.shape) > 2 and data_dim == "2D":
        avged_mag = np.mean(mag, axis=0)
    elif len(mag.shape) > 1 and data_dim == "1D":
        avged_mag = np.mean(mag, axis=0)
    else:
        avged_mag = mag  # each point is a single shot

    return avged_mag


def plot2_PSBdata(pnts, avgi, avgq, PSB=True, thresh=None, transpose=True):
    """Helper for plotting NDAverager data."""

    if PSB:
        mag = interpret_data_PSB(avgi, avgq, thresh=thresh)
    else:
        mag = interpret_data_DCS(avgi, avgq)

    if transpose:
        mag = np.transpose(mag)

    x_pts = pnts[0]
    y_pts = pnts[1]
    data_grid = mag
    fig = plt.figure()
    # plt.pcolormesh(dac2volts(x_pts)*1000, dac2volts(y_pts)*1000, data_grid, shading='nearest', cmap = 'binary_r')
    plt.pcolormesh(x_pts, y_pts, data_grid, shading="nearest", cmap="binary_r")
    if PSB:
        plt.colorbar(label="DCS conductance - reference measurement, arbs")
    else:
        plt.colorbar(label="DCS conductance, arbs")
    return fig


def plot1_PSBdata(pnts, avgi, avgq, thresh=None):
    """Helper for plotting NDAverager data."""

    mag = interpret_data_PSB(avgi, avgq, data_dim="1D", thresh=thresh)

    fig = plt.figure()
    plt.plot(pnts[0], mag, ".-")
    return fig


def plot2_simple(
    xarray, yarray, data, mode="sdchop", cbar_label="DCS conductance, arbs"
):
    fig = plt.figure()
    if mode == "sdchop":
        data = np.abs(data)
        plt.pcolormesh(xarray, yarray, data, shading="nearest", cmap="binary_r")
        plt.colorbar(label=cbar_label)
    return fig
