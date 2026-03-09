"""
File for managing filter types/settings and applying them to waveforms.

Unit convention
~~~~~~~~~~~~~~~
``soccfg`` returns sampling frequencies in **MHz**.  ``FilterPath.apply()``
converts ``fs`` from MHz to **Hz** at the boundary so that all filter
internals and config values (e.g. Butterworth ``cutoff``) use base SI
units.  Delays are specified in **nanoseconds** in the config and
converted internally.

Amplitude convention
~~~~~~~~~~~~~~~~~~~~
``FilterPath.apply()`` normalises the input waveform by dividing by
``MAX_DAC_GAIN`` (32766) before running the filter chain, so every
filter operates in the range [-1, 1] where 1.0 = full DAC scale.
After filtering the result is scaled back to DAC counts.  This means
filter parameters like ``CLIP(limit=1.0)`` and
``RENORMALIZE(target=1.0)`` always refer to fractions of the DAC range.

To add a filter:
    1. Define new filter class in bottom section of the file (filters are alphabetized).
    2. In spinqick_enums.py add the new enum to FilterTypes
    3. Use that enum in filter_config to call on your filter
    Note: kwargs get passed through, so you can add what you need for any filter type and
    retroactively support more kwargs for existing filters if needed.
"""

import json
import logging
from abc import ABC
from pathlib import Path

import numpy as np
from scipy import signal

from spinqick.helper_functions.dac_pulses import MAX_DAC_GAIN
from spinqick.settings import file_settings

logger = logging.getLogger(__name__)

# Module-level cache — populated by load_filter_config() / build_filter_map()
_filter_config: dict = {"filters": {}, "paths": {}, "gate_scheme": {}}
_filter_map: dict[int, "FilterPath"] = {}


class Filter(ABC):
    """Base class for all filters.

    Named parameters are consumed by each subclass ``__init__``.  Any
    *extra* config keys are stored in ``extra_kwargs`` and forwarded to
    the underlying signal-processing call in ``apply()`` (e.g. ``padlen``
    for ``filtfilt``, ``method`` for ``convolve``).
    """

    def __init__(self, **kwargs):
        self.extra_kwargs = kwargs

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items() if k != "extra_kwargs" or v
        )
        return f"{type(self).__name__}({params})"

    def apply(self, wf, **kwargs):
        """Apply the filter to the input waveform."""
        pass


class FilterPath:
    """Class that manages the filtering path for waveforms."""

    def __init__(self, filter_conf: list[dict] = None):
        """
        Instantiates a Filter_Path object that applies the sequence of
        filters specified in the filter_config.
        :param filter_conf: A list of dictionaries specifying the filters to apply and their settings.
        """
        self.filter_path = []
        if filter_conf is not None:
            for filter_dict in filter_conf:
                filter_obj = self._create_filter(filter_dict)
                self.filter_path.append(filter_obj)

    def __repr__(self) -> str:
        n = len(self.filter_path)
        header = f"Filter_Path({n} filter{'s' if n != 1 else ''}):"
        if n == 0:
            return f"{header} (empty)"
        steps = "\n".join(f"  {i + 1}. {f!r}" for i, f in enumerate(self.filter_path))
        return f"{header}\n{steps}"

    def _create_filter(self, filter_dict: dict) -> Filter:
        """
        Takes the filter dictionary passed from the filter_config and instantiates a Filter class
        representing that filter.
        """
        from spinqick.helper_functions.spinqick_enums import FilterTypes

        params = filter_dict.copy()
        filter_type = params.pop("type")
        if filter_type in FilterTypes.__members__:
            filter_class = FilterTypes[filter_type].value
            return filter_class(**params)
        else:
            raise ValueError(f"Filter type {filter_type} not recognized.")

    def apply(self, wf, pre=None, post=None, **kwargs):
        """
        Applies the filter path to the waveform. It will optionally prepend and postpend
        repeats of the wf to mitigate edge effects. The number of repeats is determined by the length of the longest filter in the path.

        The input waveform is expected in DAC counts (peak ≈ MAX_DAC_GAIN).
        It is normalised to [-1, 1] before the filter chain and scaled
        back to DAC counts afterwards.  This ensures parameters like
        ``CLIP(limit=1.0)`` mean "full DAC range".

        :param wf: The input waveform to filter (DAC counts).
        :param pre: The number of times to repeat the waveform before the original waveform. If None, no pre-padding will be applied before filtering.
        :param post: The number of times to repeat the waveform after the original waveform. If None, no post-padding will be applied before filtering.
        :param kwargs: Additional keyword arguments passed to each filter's apply method.
            ``fs`` is expected in MHz (as returned by soccfg) and is
            converted to Hz here so all filter internals and config
            values use base SI units.
        """
        # Convert fs from MHz (soccfg convention) to Hz (SI) at the boundary
        if "fs" in kwargs:
            kwargs = {**kwargs, "fs": kwargs["fs"] * 1e6}

        # Normalise to [-1, 1] where 1.0 = full DAC range
        wf_norm = wf / MAX_DAC_GAIN

        wf_extended = wf_norm
        if pre is not None:
            for i in range(pre):
                wf_extended = np.concatenate((wf_norm, wf_extended))
        if post is not None:
            for i in range(post):
                wf_extended = np.concatenate((wf_extended, wf_norm))

        wf_filt = wf_extended
        for filter in self.filter_path:
            wf_filt = filter.apply(wf_filt, **kwargs)

        if pre is not None:
            pre_length = len(wf_norm) * pre
        else:
            pre_length = 0
        if post is not None:
            post_length = -1 * len(wf_norm) * post
            wf_final = wf_filt[pre_length:post_length]
        else:
            wf_final = wf_filt[pre_length:]

        # Scale back to DAC counts
        return wf_final * MAX_DAC_GAIN


def load_filter_config() -> dict:
    """Read the filter config JSON from disk into the module-level cache.

    Call this in DotExperiment.__init__ and update_local() so changes to
    filter_config.json are picked up at each experiment run.
    """
    global _filter_config
    if file_settings.filter_config is not None:
        _filter_config = json.loads(Path(file_settings.filter_config).read_text())
    else:
        _filter_config = {"filters": {}, "paths": {}, "gate_scheme": {}}
    logger.info(
        "Loaded filter config with %d filters, %d paths, %d gate_scheme entries.",
        len(_filter_config.get("filters", {})),
        len(_filter_config.get("paths", {})),
        len(_filter_config.get("gate_scheme", {})),
    )
    return _filter_config


def build_filter_map(hardware_config) -> dict[int, "FilterPath"]:
    """Build a mapping of qick_gen number → FilterPath from the cached filter config.

    Uses three levels of indirection from filter_config.json:
      - gate_scheme: maps gate names or gate types to a path name
      - paths: maps path names to ordered lists of filter names
      - filters: maps filter names to filter type + params

    Resolution priority for each channel: gate name in gate_scheme →
    gate_type in gate_scheme → skip. A gate_scheme value of "none" means
    no filtering for that gate.

    Call this after load_filter_config() and after hardware_config is loaded.
    The resulting map is cached at module level so add_predistorted_envelope
    can access it via get_filter_path(ch) without parameter threading.

    :param hardware_config: HardwareConfig instance with channel definitions.
    :return: dict mapping qick_gen int to FilterPath.
    """
    global _filter_map
    _filter_map = {}
    gate_scheme = _filter_config.get("gate_scheme", {})
    paths = _filter_config.get("paths", {})
    filters = _filter_config.get("filters", {})

    for gate_name, gate_cfg in hardware_config.channels.items():
        if not hasattr(gate_cfg, "qick_gen"):
            continue

        gen = gate_cfg.qick_gen
        gate_type = str(gate_cfg.gate_type) if hasattr(gate_cfg, "gate_type") else None

        # Resolve gate_scheme: gate name → gate_type → skip
        path_name = None
        if str(gate_name) in gate_scheme:
            path_name = gate_scheme[str(gate_name)]
        elif gate_type is not None and gate_type in gate_scheme:
            path_name = gate_scheme[gate_type]

        if path_name is None or path_name == "none":
            logger.debug("No filter path for gen %d (%s).", gen, gate_name)
            continue

        if path_name not in paths:
            logger.warning(
                "gate_scheme references path '%s' for %s, but it is not defined in paths.",
                path_name,
                gate_name,
            )
            continue

        filter_list = [filters[name].copy() for name in paths[path_name]]
        _filter_map[gen] = FilterPath(filter_conf=filter_list)
        logger.info(
            "Filter path '%s' assigned to gen %d (%s).", path_name, gen, gate_name
        )

    return _filter_map


def get_filter_path(ch: int) -> "FilterPath | None":
    """Look up the cached FilterPath for a given qick generator channel.

    :param ch: qick generator number.
    :return: FilterPath if one is configured, None otherwise.
    """
    return _filter_map.get(ch)


### ------------- Filters defined below ------------- ###


class FilterBessel(Filter):
    """Bessel (Thomson) lowpass — maximally flat group delay, near-zero overshoot.

    Config example::

        {"type": "BESSEL", "order": 4, "cutoff": 500e6, "zero_phase": true}

    Extra keys are forwarded to ``filtfilt``/``lfilter``::

        {"type": "BESSEL", "order": 4, "cutoff": 500e6, "padlen": 100}

    :param order: Filter order (default 4).
    :param cutoff: –3 dB cutoff frequency in Hz.
    :param zero_phase: Use ``filtfilt`` (default True) or causal ``lfilter``.
    """

    def __init__(
        self, order: int = 4, cutoff: float = 0.5, zero_phase: bool = True, **kwargs
    ):
        self.order = order
        self.cutoff = cutoff
        self.zero_phase = zero_phase
        super().__init__(**kwargs)

    def apply(self, wf, **kwargs):
        """Apply the Bessel filter.  Requires ``fs`` in Hz via kwargs."""
        fs = kwargs.get("fs")
        if fs is None:
            raise ValueError(
                "FilterBessel.apply requires fs (sampling frequency in Hz)."
            )
        b, a = signal.bessel(self.order, self.cutoff, fs=fs)
        if self.zero_phase:
            return signal.filtfilt(b, a, wf, **self.extra_kwargs)
        else:
            return signal.lfilter(b, a, wf, **self.extra_kwargs)


class FilterButterworth(Filter):
    """Butterworth lowpass — maximally flat magnitude response.

    Config example::

        {"type": "BUTTERWORTH", "order": 2, "cutoff": 500e6, "zero_phase": True}

    :param order: Filter order (default 5).
    :param cutoff: –3 dB cutoff frequency in Hz.
    :param zero_phase: Use ``filtfilt`` (default True) or causal ``lfilter``.
    """

    def __init__(self, order=5, cutoff=0.5, zero_phase=True, **kwargs):
        self.order = order
        self.cutoff = cutoff
        self.zero_phase = zero_phase
        super().__init__(**kwargs)

    def apply(self, wf, **kwargs):
        """Apply the Butterworth filter.  Requires ``fs`` in Hz via kwargs."""
        fs = kwargs.get("fs")
        if fs is None:
            raise ValueError(
                "FilterButterworth.apply requires fs (sampling frequency in Hz)."
            )
        b, a = signal.butter(self.order, self.cutoff, fs=fs)
        if self.zero_phase:
            return signal.filtfilt(b, a, wf, **self.extra_kwargs)
        else:
            return signal.lfilter(b, a, wf, **self.extra_kwargs)


class FilterClip(Filter):
    """Hard-clip a waveform to ±limit.

    The limit is expressed as a fraction of ``MAX_DAC_GAIN``.
    A limit of 1.0 means the waveform is clipped at the full DAC range.

    Config example::

        {"type": "CLIP", "limit": 1.0}

    :param limit: Maximum absolute amplitude as a fraction of full DAC range (default 1.0).
    """

    def __init__(self, limit: float = 1.0, **kwargs):
        self.limit = limit
        super().__init__(**kwargs)

    def apply(self, wf, **kwargs):
        """Clip the waveform to [-limit, +limit]."""
        return np.clip(wf, -self.limit, self.limit)


class FilterFIR(Filter):
    """Apply pre-computed FIR taps via convolution.

    Config example::

        {"type": "FIR", "taps": [0.25, 0.5, 0.25]}

    :param taps: FIR coefficient array.
    """

    def __init__(self, taps, **kwargs):
        self.taps = taps
        super().__init__(**kwargs)

    def apply(self, wf, **kwargs):
        """Apply the FIR filter to the input waveform."""
        return signal.convolve(wf, self.taps, mode="same", **self.extra_kwargs)


class FilterFractionalDelay(Filter):
    """Sub-sample time shift via windowed-sinc FIR interpolation.

    Config example::

        {"type": "FRACTIONAL_DELAY", "delay_ns": 0.05, "num_taps": 21, "window": "hamming"}

    :param delay_ns: Time shift in ns (positive = delay, negative = advance).
    :param num_taps: FIR length (forced odd, default 21).
    :param window: Window for ``scipy.signal.get_window`` (default ``"hamming"``).
    """

    def __init__(
        self,
        delay_ns: float = 0.0,
        num_taps: int = 21,
        window: str = "hamming",
        **kwargs,
    ):
        self.delay_ns = delay_ns
        # Ensure odd tap count so the kernel is symmetric about its centre
        self.num_taps = num_taps if num_taps % 2 == 1 else num_taps + 1
        self.window = window
        super().__init__(**kwargs)

    def apply(self, wf, **kwargs):
        """Apply the fractional delay.  Requires ``fs`` in Hz via kwargs."""
        fs = kwargs.get("fs")
        if fs is None:
            raise ValueError(
                "FilterFractionalDelay.apply requires fs (sampling frequency in Hz)."
            )

        # Convert nanosecond delay to fractional samples.
        # delay_samples = delay_ns × 1e-9 s/ns × fs Hz = delay_ns × fs × 1e-9
        delay_samples = self.delay_ns * 1e-9 * fs

        # Build windowed-sinc kernel centred on the fractional delay
        M = self.num_taps // 2
        n = np.arange(self.num_taps) - M  # -M … 0 … +M
        kernel = np.sinc(n - delay_samples) * signal.get_window(
            self.window, self.num_taps
        )
        kernel /= kernel.sum()  # normalise to preserve DC gain

        return signal.convolve(wf, kernel, mode="same", **self.extra_kwargs)


class FilterGaussian(Filter):
    """Gaussian smoothing via convolution with a normalised Gaussian kernel.

    Convolves the waveform with a Gaussian kernel whose width is
    proportional to ``alpha``.  Constant regions (including DC offsets
    and leading/trailing zeros) are preserved because the kernel sums
    to unity.

    * ``alpha = 0`` → identity (no smoothing).
    * Larger ``alpha`` → wider kernel → smoother transitions.

    Config example::

        {"type": "GAUSSIAN", "alpha": 0.5, "sigma": 3.0}

    :param alpha: Kernel width as a fraction of waveform length (0–1).
    :param sigma: Std-devs per kernel half (default 3.0); higher = sharper edges.
    """

    def __init__(self, alpha: float = 0.5, sigma: float = 3.0, **kwargs):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.alpha = alpha
        self.sigma = sigma
        super().__init__(**kwargs)

    def apply(self, wf, **kwargs):
        """Smooth transitions by convolving with a Gaussian kernel."""
        N = len(wf)
        if self.alpha == 0 or N == 0:
            return wf.copy()

        kernel_len = int(self.alpha * N / 2)
        kernel_len = max(3, kernel_len | 1)  # odd, at least 3
        if N < kernel_len:
            return wf.copy()

        kernel = signal.windows.gaussian(kernel_len, std=kernel_len / (2 * self.sigma))
        kernel /= kernel.sum()

        return signal.convolve(wf, kernel, mode="same")


class FilterIIR(Filter):
    """Apply pre-computed IIR coefficients via causal ``lfilter``.

    Config example::

        {"type": "IIR", "b": [0.1, 0.2, 0.1], "a": [1.0, -0.5, 0.1]}

    :param b: Numerator (feedforward) coefficients.
    :param a: Denominator (feedback) coefficients.
    """

    def __init__(self, b=None, a=None, **kwargs):
        self.b = b
        self.a = a
        super().__init__(**kwargs)

    def apply(self, wf, **kwargs):
        """Apply the IIR filter to the input waveform."""
        return signal.lfilter(self.b, self.a, wf, **self.extra_kwargs)


class FilterLogCorrection(Filter):
    """Log correction for exponential V-to-J coupling (Xue et al., arXiv:2107.00628).

    Maps ``output = peak · log(1 + A·|w/peak|) / log(1 + A) · sign(w)``.
    A = 0 is identity; larger A = stronger compression.

    Config example::

        {"type": "LOG_CORRECTION", "A": 50}

    :param A: Dynamic-range parameter.  Measure by sweeping the barrier
        gate voltage, fitting J(V) to an exponential, and computing
        ``A = exp(V_peak / V_0) - 1``.
    """

    def __init__(self, A: float = 100.0, **kwargs):
        if A < 0:
            raise ValueError("A must be non-negative.")
        self.A = A
        super().__init__(**kwargs)

    def apply(self, wf, **kwargs):
        """Apply logarithmic correction to the input waveform."""
        if self.A == 0:
            return wf.copy()
        peak = np.max(np.abs(wf))
        if peak == 0:
            return wf.copy()
        wf_norm = np.abs(wf) / peak
        corrected = np.log1p(self.A * wf_norm) / np.log1p(self.A)
        return corrected * peak * np.sign(wf)


class FilterRenormalize(Filter):
    """Rescale so peak amplitude equals target (shape-preserving, no clipping).

    The target is expressed as a fraction of ``MAX_DAC_GAIN``.
    A target of 1.0 means the peak is rescaled to the full DAC range.

    Config example::

        {"type": "RENORMALIZE", "target": 1.0}

    :param target: Desired peak absolute amplitude as a fraction of full DAC range (default 1.0).
    """

    def __init__(self, target: float = 1.0, **kwargs):
        self.target = target
        super().__init__(**kwargs)

    def apply(self, wf, **kwargs):
        """Scale the waveform so max(|wf|) == target, only if it exceeds target."""
        peak = np.max(np.abs(wf))
        if peak > self.target:
            return wf * (self.target / peak)
        return wf


class FilterSavitzkyGolay(Filter):
    """Savitzky–Golay polynomial smoothing — preserves plateau better than lowpass.

    Config example::

        {"type": "SAVITZKY_GOLAY", "window_length": 11, "polyorder": 3}

    :param window_length: Smoothing window (positive odd int).
    :param polyorder: Local polynomial order (must be < ``window_length``).
    """

    def __init__(self, window_length: int = 11, polyorder: int = 3, **kwargs):
        if window_length % 2 == 0 or window_length < 1:
            raise ValueError("window_length must be a positive odd integer.")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length.")
        self.window_length = window_length
        self.polyorder = polyorder
        super().__init__(**kwargs)

    def apply(self, wf, **kwargs):
        """Apply Savitzky–Golay smoothing to the input waveform."""
        if len(wf) < self.window_length:
            return wf.copy()
        return signal.savgol_filter(
            wf, self.window_length, self.polyorder, **self.extra_kwargs
        )


class FilterTukey(Filter):
    """Cosine-tapered smoothing via convolution with a Hann kernel.

    Convolves the waveform with a normalised raised-cosine (Hann) kernel
    whose width is proportional to ``alpha``.  Constant regions
    (including DC offsets and leading/trailing zeros) are preserved
    because the kernel sums to unity.

    * ``alpha = 0`` → identity (no smoothing).
    * Larger ``alpha`` → wider kernel → smoother transitions.

    Config example::

        {"type": "TUKEY", "alpha": 0.5}

    :param alpha: Kernel width as a fraction of waveform length (0–1).
    """

    def __init__(self, alpha=0.5, **kwargs):
        self.alpha = alpha
        super().__init__(**kwargs)

    def apply(self, wf, **kwargs):
        """Smooth transitions by convolving with a Hann kernel."""
        N = len(wf)
        if self.alpha == 0 or N == 0:
            return wf.copy()

        kernel_len = int(self.alpha * N / 2)
        kernel_len = max(3, kernel_len | 1)  # odd, at least 3
        if N < kernel_len:
            return wf.copy()

        kernel = signal.windows.hann(kernel_len)
        kernel /= kernel.sum()

        return signal.convolve(wf, kernel, mode="same")
