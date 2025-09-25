"""functions for analyzing data"""

from typing import List
import logging
import numpy as np
from lmfit import models
from sklearn.mixture import GaussianMixture
from spinqick.core import spinqick_data, spinqick_utils

logger = logging.getLogger(__name__)


def analyze_conductance(data: List[np.ndarray]):
    """calculate data magnitude from acquire function output"""
    full_data = []
    for adc_data in data:
        mag = np.sqrt(
            adc_data[..., 0] ** 2 + adc_data[..., 1] ** 2
        )  # ideally we change this to read out each adc in series
        full_data.append(mag)
    return full_data


def analyze_transconductance(data: List[np.ndarray]):
    """calculate transconductance from acquire function output"""
    full_data = []
    for adc_data in data:
        max_ind = np.argmax(adc_data[..., 0])  # get max amplitude in I
        real_flat = adc_data[..., 0].flatten()
        imaginary_flat = adc_data[..., 1].flatten()
        rot_ang = np.angle(
            real_flat[max_ind] + 1j * imaginary_flat[max_ind]
        )  # get angle between I and Q for this point
        complex_data = adc_data[..., 0] + 1j * adc_data[..., 1]
        rotated = np.exp(-1j * rot_ang) * complex_data
        rotated_real = np.real(rotated)
        full_data.append(rotated_real)  # data should be fully rotated into I
    return full_data


def thresh_psb(data: np.ndarray, threshold: float):
    """threshold data"""
    if threshold < 0:
        threshed = np.heaviside(data - threshold, 0)
    else:
        threshed = np.heaviside(threshold - data, 0)
    return threshed


def interpret_data_psb(data, diff=True, thresh=None):
    """take magnitude of data, take difference between singlet and triplet data and threshold if needed"""

    mag = analyze_conductance(data)[0]  # for now this is just for the first adc

    if diff:
        differential = mag[1] - mag[0]
    else:
        differential = mag
    if thresh is not None:
        threshed = thresh_psb(differential, thresh)
    else:
        threshed = differential

    return threshed


def calculate_conductance(
    data: spinqick_data.SpinqickData,
    adc_conversion: list[float],
    average_level: spinqick_utils.AverageLevel | None = None,
):
    """calculates conductance from raw data and saves on the analyzed_data attribute"""
    conductance_data = analyze_conductance(data.raw_data)
    adcs = range(len(conductance_data))
    # average inner and/or outer loops
    if average_level is not None:
        if average_level in [
            spinqick_utils.AverageLevel.BOTH,
            spinqick_utils.AverageLevel.INNER,
        ]:
            conductance_data = [np.mean(conductance_data[i], axis=-1) for i in adcs]
        if average_level in [
            spinqick_utils.AverageLevel.BOTH,
            spinqick_utils.AverageLevel.OUTER,
        ]:
            conductance_data = [np.mean(conductance_data[i], axis=1) for i in adcs]
    data.analyzed_data = [conductance_data[i] * adc_conversion[i] for i in adcs]
    data.analysis_type = "conductance"
    data.analysis_averaged = average_level


def calculate_transconductance(
    data: spinqick_data.SpinqickData, adc_conversion: list[float]
):
    """calculates transconductance from raw data and saves on the analyzed_data attribute"""
    trans_data = analyze_transconductance(data.raw_data)
    data.analyzed_data = [
        trans_data[i] * adc_conversion[i] for i in range(len(trans_data))
    ]
    data.analysis_type = "transconductance"


def calculate_electron_temperature(
    data: spinqick_data.SpinqickData,
    plunger_gate: str,
):
    """Calculates the electron temperature from a 1D sweep over a loading line."""

    x_data = data.axes["x"]["sweeps"][plunger_gate]["data"]
    y_data = data.analyzed_data
    assert y_data is not None
    sigmoid, out = fit_sigmoid(x_data, y_data[0][0])

    kta = out.params["kt"].value
    return kta


def fit_blobs(
    data: np.ndarray, n_components: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fits gaussian mixture models to integrated measurements."""
    # GaussianMixture needs shape (n,1) for n independent measurements.
    data = np.squeeze(data)
    if data.ndim == 1:
        data_to_bin = data.reshape(-1, 1)
        data_to_bin = np.expand_dims(data_to_bin, 1)
    elif data.ndim == 2:
        data_to_bin = np.moveaxis(data, -1, 0)  # shot axis to first axis
        data_to_bin = data_to_bin.reshape(
            data_to_bin.shape[0], 1, *data_to_bin.shape[1:]
        )
    else:
        raise NotImplementedError(
            "Gaussian mixture models not implement "
            + "for datasets with more than two dimensions"
        )

    mean, covs, weights = np.empty((3, data_to_bin.shape[-1], n_components))
    for ii in range(data_to_bin.shape[-1]):
        gmm_fit = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            tol=1e-12,
            reg_covar=1e-12,
        ).fit(data_to_bin[..., ii])

        weight = np.squeeze(gmm_fit.weights_)
        covars = np.squeeze(gmm_fit.covariances_)
        mus = np.squeeze(gmm_fit.means_)
        idxs = np.argsort(weight / np.sqrt(covars))[::-1]  # find most prominent peaks

        mean[ii, :] = mus[idxs]
        covs[ii, :] = covars[idxs]
        weights[ii, :] = weight[idxs]

    return mean, covs, weights


def fit_gaussian(x_data, y_data):
    """fit data to a gaussian"""
    gaussian = models.GaussianModel()
    line = models.ConstantModel()
    model = gaussian + line
    mag = y_data
    # pars = gaussian.guess(mag, x=x_data)
    pars = model.make_params(
        center=dict(
            value=x_data[np.argmax(y_data)], min=np.min(x_data), max=np.max(x_data)
        ),
        amplitude=dict(value=np.sum(y_data), min=0),
        # amplitude=dict(value=np.sum(y_data)),
        sigma=dict(value=(np.max(x_data) - np.min(x_data)) / 10),
        min=0,
        c=dict(value=np.min(y_data)),
    )
    out = model.fit(mag, pars, x=x_data)
    return model, out


def fit_sigmoid(x_data: np.ndarray, y_data: np.ndarray):
    """fit data to a sigmoid"""
    sigmoid = models.ThermalDistributionModel(form="fermi")
    amp = models.ConstantModel()
    offset = models.ConstantModel(prefix="o")
    model = sigmoid * amp + offset
    pars = model.make_params(
        c=dict(
            value=np.max(y_data) - np.min(y_data),
            min=0,
            max=np.max(y_data),
        ),
        oc=dict(
            value=np.min(y_data),
            min=np.min(y_data),
            max=np.max(y_data),
        ),
        center=dict(
            min=np.min(x_data),
            max=np.max(x_data),
            value=np.mean(x_data),
        ),
        kt=dict(
            value=np.mean(x_data[1:] - x_data[:-1]),
        ),
        amplitude=dict(value=1, vary=False),
    )
    out = model.fit(y_data, pars, x=x_data)
    return model, out


def calculate_difference(
    data: spinqick_data.PsbData,
    average_level: spinqick_utils.AverageLevel | None = None,
):
    """subtract reference measurement from actual data"""
    diff = []
    assert data.analyzed_data is not None
    for conductance_data in data.analyzed_data:
        diff_vals = conductance_data[1] - conductance_data[0]
        diff.append(diff_vals)

    adcs = range(len(diff))
    # average inner and/or outer loops
    if average_level is not None:
        if average_level in [
            spinqick_utils.AverageLevel.BOTH,
            spinqick_utils.AverageLevel.INNER,
        ]:
            diff = [np.mean(diff[i], axis=-1) for i in adcs]
        if average_level in [
            spinqick_utils.AverageLevel.BOTH,
            spinqick_utils.AverageLevel.OUTER,
        ]:
            diff = [np.mean(diff[i], axis=0) for i in adcs]

    data.difference_data = diff
    data.difference_avged = average_level


def calculate_thresholded(
    data: spinqick_data.PsbData,
    threshold: List[float],
    average_level: spinqick_utils.AverageLevel | None = None,
):
    """calculate thresholded 0 or 1 data from analyzed data and average after thresholding"""
    threshed = []
    if data.difference_data is None:
        pre_thresh = data.analyzed_data
    else:
        pre_thresh = data.difference_data
    if pre_thresh is not None:
        for i, pre_thresh_array in enumerate(pre_thresh):
            thresh_vals = thresh_psb(pre_thresh_array, threshold[i])
            threshed.append(thresh_vals)
        adcs = range(len(threshed))
        # average inner and/or outer loops
        if average_level is not None:
            if average_level in [
                spinqick_utils.AverageLevel.BOTH,
                spinqick_utils.AverageLevel.INNER,
            ]:
                threshed = [np.mean(threshed[i], axis=-1) for i in adcs]
            if average_level in [
                spinqick_utils.AverageLevel.BOTH,
                spinqick_utils.AverageLevel.OUTER,
            ]:
                threshed = [np.mean(threshed[i], axis=0) for i in adcs]
        data.threshed_data = threshed
        data.threshold = threshold
        data.thresh_avged = average_level


def analyze_psb_standard(
    sq_data: spinqick_data.PsbData,
    adc_unit_conversions: List[float],
    reference: bool,
    thresh: bool,
    threshold: float | None,
    final_avg_lvl: spinqick_utils.AverageLevel
    | None = spinqick_utils.AverageLevel.BOTH,
):
    """calculates conductance and if desired the thresholded data from a spinqick data object"""
    if not reference and not thresh:
        calculate_conductance(
            sq_data,
            adc_unit_conversions,
            average_level=final_avg_lvl,
        )
    else:
        calculate_conductance(
            sq_data,
            adc_unit_conversions,
            average_level=None,
        )
        if reference:
            if thresh:
                calculate_difference(sq_data, average_level=None)
                assert threshold
                calculate_thresholded(sq_data, [threshold], average_level=final_avg_lvl)
            else:
                calculate_difference(sq_data, average_level=final_avg_lvl)
