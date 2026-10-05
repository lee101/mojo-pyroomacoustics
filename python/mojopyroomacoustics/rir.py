"""Room impulse response construction from `pyroomacoustics.simulation.ism`.

`rir_builder`, `delay_sum` and `fractional_delay` are upstream's own C++ kernels
in `libroom_src/rir_builder.cpp`, ported here and used to assemble the impulse
response between a source and a microphone exactly as `compute_ism_rir` does.
"""

from __future__ import annotations

import math

import numpy as np

from . import _lib
from .parameters import constants

__all__ = [
    "rir_builder",
    "delay_sum",
    "fractional_delay",
    "compute_ism_rir",
]


def _tables(fs, fdl=None, lut_gran=None):
    """Build (and return) the sinc LUT and Hann window the C++ kernels share.

    Upstream builds these inside every `rir_builder` call. They depend only on
    the filter geometry, so they are built once and reused.
    """
    if fdl is None:
        fdl = constants.get("frac_delay_length")
    if lut_gran is None:
        lut_gran = constants.get("sinc_lut_granularity")
    lut = np.zeros((fdl + 2) * lut_gran, dtype=np.float64)
    hann = np.zeros(fdl, dtype=np.float64)
    lib = _lib.require()
    lib.mpa_build_sinc_lut(_lib.addr(lut), fdl, lut_gran)
    lib.mpa_build_hann(_lib.addr(hann), fdl)
    return lut, hann, fdl, lut_gran


def rir_builder(rir, time, alpha, fs, fdl=None, lut_gran=None, parallel=None):
    """Build a RIR from image-source times and attenuations.

    Parameters
    ----------
    rir: ndarray
        Output buffer, accumulated into (`+=`), as upstream does.
    time: ndarray
        Arrival times in seconds, one per image source.
    alpha: ndarray
        Attenuation of each arrival.
    fs: int
        Sampling frequency.
    fdl: int, optional
        Length of the fractional delay filters. Defaults to `frac_delay_length`.
    lut_gran: int, optional
        Points per integer interval in the sinc interpolation LUT.
    parallel: bool, optional
        Force the threaded path on or off. Defaults to threading above the
        kernel's own size threshold.

    Notes
    -----
    `rir` must be zeroed by the caller, exactly as upstream's caller does, and
    long enough to hold the last arrival's filter tail.
    """
    time = _lib.f64(time)
    alpha = _lib.f64(alpha)
    rir = _lib.f64(rir)
    lut, hann, fdl, lut_gran = _tables(fs, fdl, lut_gran)
    lib = _lib.require()

    n_times = time.size
    taps = n_times * fdl
    if parallel is None:
        parallel = taps >= (1 << 16)

    if parallel:
        lib.mpa_rir_builder_parallel(
            _lib.addr(rir), _lib.addr(time), _lib.addr(alpha), _lib.addr(lut),
            _lib.addr(hann), n_times, rir.size, int(fs), fdl, lut_gran,
            _lib.workers(),
        )
    else:
        lib.mpa_rir_builder(
            _lib.addr(rir), _lib.addr(time), _lib.addr(alpha), _lib.addr(lut),
            _lib.addr(hann), n_times, rir.size, int(fs), fdl, lut_gran,
        )
    return rir


def delay_sum(irs, delays, output=None, parallel=None):
    """Overlap-add short impulse responses at integer delays.

    Parameters
    ----------
    irs: ndarray (n_irs, n_taps)
        The short impulse responses.
    delays: ndarray (n_irs,)
        Integer sample delays.
    output: ndarray, optional
        Output buffer. Allocated when not given.
    """
    irs = _lib.f64(irs)
    delays = _lib.f64(delays).astype(np.float64)
    n_irs, n_taps = irs.shape
    if output is None:
        output = np.zeros(int(delays.max()) + n_taps, dtype=np.float64)
    output = _lib.f64(output)
    lib = _lib.require()

    if parallel is None:
        parallel = n_irs * n_taps >= (1 << 16)
    if parallel:
        lib.mpa_delay_sum_parallel(
            _lib.addr(output), _lib.addr(irs), _lib.addr(delays), n_irs, n_taps,
            output.size, _lib.workers(),
        )
    else:
        lib.mpa_delay_sum(
            _lib.addr(output), _lib.addr(irs), _lib.addr(delays), n_irs, n_taps,
            output.size,
        )
    return output


def fractional_delay(time, fdl=None, lut_gran=None):
    """Build fractional-delay filters for each fractional arrival time.

    Returns
    -------
    ndarray (n_times, fdl)
        One windowed sinc filter per arrival.
    """
    time = _lib.f64(time)
    lut, hann, fdl, lut_gran = _tables(1, fdl, lut_gran)
    out = np.zeros((time.size, fdl), dtype=np.float64)
    _lib.require().mpa_fractional_delay(
        _lib.addr(out), _lib.addr(time), _lib.addr(lut), _lib.addr(hann),
        time.size, fdl, lut_gran,
    )
    return out


def compute_ism_rir(src, mic, mic_dir=None, src_directions=None, is_visible=None,
                    fdl=None, c=None, fs=None, octave_bands=None, min_phase=True,
                    air_abs_coeffs=None):
    """Compute the room impulse response between a source and a microphone.

    This follows upstream's `simulation/ism.py:compute_ism_rir` for the flat and
    octave-band cases: per-image distance, arrival time, amplitude and optional
    air absorption, then one `rir_builder` pass per octave band accumulated into
    the RIR.

    Parameters
    ----------
    src: pyroomacoustics.SoundSource
        The source, carrying `images` (dim, n) and `damping`.
    mic: ndarray (dim,)
        The microphone position.
    fdl: int, optional
        Fractional delay filter length. Defaults to `frac_delay_length`.
    c: float, optional
        Speed of sound. Defaults to `constants.get("c")`.
    fs: int, optional
        Sampling frequency. Taken from `src.fs` when not given.
    is_visible: ndarray, optional
        Visibility mask over the image sources. All visible when omitted.

    Returns
    -------
    ndarray
        The room impulse response.
    """
    if fdl is None:
        fdl = constants.get("frac_delay_length")
    if c is None:
        c = float(constants.get("c"))
    if fs is None:
        fs = int(getattr(src, "fs", 16000))

    images = np.asarray(src.images, dtype=np.float64)
    att = np.asarray(src.damping, dtype=np.float64).reshape(-1, images.shape[1])
    if is_visible is not None:
        images = images[:, is_visible]
        att = att[:, is_visible]

    mic = np.asarray(mic, dtype=np.float64).ravel()
    dim = images.shape[0]
    n_img = images.shape[1]

    # damping is per octave band; a flat response is the single-band case
    n_bands = att.shape[0]

    dst = np.zeros(n_bands * n_img + n_img, dtype=np.float64)
    if air_abs_coeffs is not None:
        air_abs = np.asarray(air_abs_coeffs, dtype=np.float64).ravel()
    else:
        air_abs = np.zeros(max(n_bands, 1), dtype=np.float64)
    _lib.require().mpa_ism_image_amplitudes(
        _lib.addr(dst), _lib.addr(_lib.f64(images)), _lib.addr(_lib.f64(mic)),
        _lib.addr(_lib.f64(att.ravel())), dim, n_img, float(c),
        _lib.addr(_lib.f64(air_abs)), n_bands,
    )

    amplitudes = dst[: n_bands * n_img].reshape(n_bands, n_img)
    time = dst[n_bands * n_img :]

    # we add the delay due to the fractional delay filter to the arrival times
    # to avoid problems when propagation is shorter than the filter delay
    fdl2 = fdl // 2
    time = time + (fdl2 / fs)

    rir_len = _lib.require().mpa_ism_rir_length(
        _lib.addr(_lib.f64(time)), n_img, int(fs), fdl
    )
    rir = np.zeros(rir_len, dtype=np.float64)
    lut, hann, _, lut_gran = _tables(fs, fdl)
    _lib.require().mpa_ism_rir_assemble(
        _lib.addr(rir), _lib.addr(_lib.f64(time)), _lib.addr(_lib.f64(amplitudes)),
        n_img, n_bands, int(fs), fdl, _lib.addr(lut), _lib.addr(hann), lut_gran,
        rir_len,
    )
    return rir