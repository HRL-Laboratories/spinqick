"""
Functions to help with plotting data output from averager functions
"""

import numpy as np
import matplotlib.pyplot as plt
from spinqick.core import spinqick_data


def plot_decimated(iq_list, config, plot_iq=False):
    """Plot output of acquire_decimated()."""
    fig = plt.figure()
    readout_length = config["readout_length"]
    time = np.linspace(
        0, readout_length / config["adc_sr"], int(readout_length)
    )  # [us]

    for iq in iq_list:
        iq = np.transpose(iq)  # RAveragerProgram
        plt.plot(
            time, np.abs(iq[0] + 1j * iq[1]), label="mag, ADC %d" % (config["ro_ch"])
        )
        if plot_iq:
            plt.plot(time, iq[0], label="I value, ADC %d" % (config["ro_ch"]))
            plt.plot(time, iq[1], label="Q value, ADC %d" % (config["ro_ch"]))

    plt.legend()
    plt.xlabel(r"time ($\mu s$)")
    plt.ylabel("ADC units")
    return fig


def interpret_data_dcs(avgi, avgq, data_dim="2D"):
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


def interpret_data_psb(avgi, avgq, data_dim="2D", thresh=None):
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


def plot2_psb_data(pnts, avgi, avgq, psb=True, thresh=None, transpose=True):
    """Helper for plotting NDAverager data."""

    if psb:
        mag = interpret_data_psb(avgi, avgq, thresh=thresh)
    else:
        mag = interpret_data_dcs(avgi, avgq)

    if transpose:
        mag = np.transpose(mag)

    x_pts = pnts[0]
    y_pts = pnts[1]
    data_grid = mag
    fig = plt.figure()
    plt.pcolormesh(x_pts, y_pts, data_grid, shading="nearest", cmap="binary_r")
    if psb:
        plt.colorbar(label="DCS conductance - reference measurement, arbs")
    else:
        plt.colorbar(label="DCS conductance, arbs")
    return fig


def plot1_psb_data(pnts, avgi, avgq, thresh=None):
    """Helper for plotting NDAverager data."""

    mag = interpret_data_psb(avgi, avgq, data_dim="1D", thresh=thresh)

    fig = plt.figure()
    plt.plot(pnts[0], mag, ".-")
    return fig


def plot2_simple(xarray, yarray, data, timestamp, cbar_label="DCS conductance, arbs"):
    """Basic 2D plot"""
    fig = plt.figure()
    plt.pcolormesh(xarray, yarray, data, shading="nearest", cmap="binary_r")
    plt.colorbar(label=cbar_label)
    plt.title("t: %d" % timestamp, loc="right", fontdict={"fontsize": 6})
    return fig


def plot1_simple(
    xarray, data, timestamp, dset_label: str | None = None, new_figure=True
):
    """Basic 1D plot"""
    if new_figure:
        fig = plt.figure()
    else:
        fig = plt.gcf()
    if dset_label is None:
        plt.plot(xarray, data)
    else:
        plt.plot(xarray, data, label=dset_label)
    plt.title("t: %d" % timestamp, loc="right", fontdict={"fontsize": 6})
    return fig


def plot2_psb(sqd: spinqick_data.PsbData, x_gate: str, y_gate: str):
    """plot psb data"""
    xarray = sqd.axes["x"]["sweeps"][x_gate]["data"]
    xloop = sqd.axes["x"]["loop_no"]
    yarray = sqd.axes["y"]["sweeps"][y_gate]["data"]
    yloop = sqd.axes["y"]["loop_no"]
    if sqd.threshed_data is not None:
        plot_data = sqd.threshed_data
        plt_type = "thresholded"
    elif sqd.difference_data is not None:
        plot_data = sqd.difference_data
        plt_type = "conductance data minus reference measurement"
    else:
        assert sqd.analyzed_data is not None
        plot_data = [data[0] for data in sqd.analyzed_data]
        plt_type = "conductance"  # TODO add units
    for adc_data in plot_data:
        if xloop < yloop:
            plot_data_adj = np.transpose(adc_data)
        else:
            plot_data_adj = adc_data
        fig = plot2_simple(
            xarray, yarray, plot_data_adj, sqd.timestamp, cbar_label=plt_type
        )

    return fig


def plot1_psb(sqd: spinqick_data.PsbData, x_gate: str):
    """plot psb data 1d plot"""
    xarray = sqd.axes["x"]["sweeps"][x_gate]["data"]
    if sqd.threshed_data is not None:
        plot_data = sqd.threshed_data
    elif sqd.difference_data is not None:
        plot_data = sqd.difference_data
    else:
        assert sqd.analyzed_data is not None
        plot_data = [data[0] for data in sqd.analyzed_data]
    for adc_data in plot_data:
        fig = plot1_simple(xarray, adc_data, sqd.timestamp)

    return fig
