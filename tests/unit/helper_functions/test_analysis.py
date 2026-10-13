"""Unit tests for analysis helpers using synthetic arrays only."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from spinqick.core.spinqick_data import PsbData, SpinqickData
from spinqick.helper_functions import analysis


def test_analyze_conductance_calculates_iq_magnitude():
    data = [np.array([[3.0, 4.0], [5.0, 12.0]])]

    result = analysis.analyze_conductance(data)

    np.testing.assert_allclose(result[0], np.array([5.0, 13.0]))


def test_analyze_transconductance_rotates_maximum_i_point_to_real_axis():
    data = [
        np.array(
            [
                [[1.0, 1.0], [2.0, 0.0]],
            ]
        )
    ]

    result = analysis.analyze_transconductance(data)

    np.testing.assert_allclose(result[0], np.array([[1.0, 2.0]]), atol=1e-12)


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (0.0, np.array([1.0, 0.0, 0.0])),
        (-1.0, np.array([0.0, 1.0, 1.0])),
    ],
)
def test_thresh_psb_supports_positive_and_negative_thresholds(threshold, expected):
    data = np.array([-2.0, 0.0, 2.0])

    result = analysis.thresh_psb(data, threshold)

    np.testing.assert_array_equal(result, expected)


def test_interpret_data_psb_calculates_difference_and_threshold():
    data = [
        np.array(
            [
                [[1.0, 0.0], [2.0, 0.0]],
                [[4.0, 0.0], [1.0, 0.0]],
            ]
        )
    ]

    result = analysis.interpret_data_psb(data, diff=True, thresh=0.0)

    # Magnitudes are [[1, 2], [4, 1]], so the difference is [3, -1].
    np.testing.assert_array_equal(result, np.array([0.0, 1.0]))


def test_calculate_conductance_sets_analysis_fields():
    data = SimpleNamespace(
        raw_data=[np.array([[[3.0, 4.0], [0.0, 5.0]]])],
        analyzed_data=None,
        analysis_type=None,
        analysis_averaged=None,
    )

    analysis.calculate_conductance(
        cast(SpinqickData, data),
        adc_conversion=[2.0],
        average_level=None,
    )

    np.testing.assert_allclose(data.analyzed_data[0], np.array([[10.0, 10.0]]))
    assert data.analysis_type == "conductance"
    assert data.analysis_averaged is None


def test_calculate_transconductance_sets_analysis_fields():
    data = SimpleNamespace(
        raw_data=[np.array([[[1.0, 1.0], [2.0, 0.0]]])],
        analyzed_data=None,
        analysis_type=None,
    )

    analysis.calculate_transconductance(cast(SpinqickData, data), adc_conversion=[2.0])

    np.testing.assert_allclose(data.analyzed_data[0], np.array([[2.0, 4.0]]))
    assert data.analysis_type == "transconductance"


def test_calculate_difference_sets_difference_metadata():
    data = SimpleNamespace(
        analyzed_data=[
            np.array(
                [
                    [1.0, 2.0],
                    [4.0, 7.0],
                ]
            )
        ],
        difference_data=None,
        difference_avged=None,
    )

    analysis.calculate_difference(
        cast(PsbData, data),
        average_level=None,
    )

    np.testing.assert_allclose(data.difference_data[0], np.array([3.0, 5.0]))
    assert data.difference_avged is None


def test_calculate_thresholded_sets_threshold_metadata():
    data = SimpleNamespace(
        analyzed_data=[np.array([[1.0, 3.0]])],
        difference_data=None,
        threshed_data=None,
        threshold=None,
        thresh_avged=None,
    )

    analysis.calculate_thresholded(
        cast(PsbData, data),
        threshold=[2.0],
        average_level=None,
    )

    np.testing.assert_array_equal(data.threshed_data[0], np.array([[1.0, 0.0]]))
    assert data.threshold == [2.0]
    assert data.thresh_avged is None
