"""mojo-pyroomacoustics: pyroomacoustics' room impulse response and beamforming
kernels, implemented in Mojo and callable with upstream's names.

    import mojopyroomacoustics as mpa

    rir = mpa.rir_builder_from_images(images, damping, mic, fs=16000)
    bf = mpa.Beamformer(mpa.circular_2D_array([0, 0], 8, 0, 0.04), 16000)
    bf.far_field_weights(np.pi / 2)
    freq, response = bf.response(np.linspace(0, np.pi, 64), 1000.0)

The port covers the compute-bound kernels. It does not cover upstream's room
engines (`ShoeBox`, `IndoorSound`, the hybrid ray-tracing simulator), its
directivity models, or its file and audio I/O; see the README for the full list.
"""

import numpy as np

from .parameters import Constants, constants, get_num_threads
from .utilities import (
    autocorr,
    convmtx,
    fractional_delay,
    fractional_delay_filter_bank,
    goertzel,
    levinson,
    low_pass_dirac,
    lpc,
)
from .rir import compute_ism_rir, delay_sum, fractional_delay as frac_delay_filters
from .rir import rir_builder
from . import beamforming
from .beamforming import (
    Beamformer,
    H,
    MicrophoneArray,
    build_rir_matrix,
    circular_2D_array,
    distance,
    linear_2D_array,
    mdot,
    mdot3,
    poisson_2D_array,
    rake_delay_and_sum,
    rake_mvdr_filters,
    spiral_2D_array,
    square_2D_array,
    unit_vec2D,
)

__version__ = "0.1.0"


def rir_builder_from_images(images, damping, mic, fs=16000, c=None, fdl=None,
                            air_abs_coeffs=None):
    """Build a room impulse response from explicit image-source positions.

    This is the standalone form of upstream's `compute_ism_rir`: it takes the
    image positions and their per-band damping directly instead of a
    `SoundSource`, so a caller who has already run the image-source model does
    not have to construct one.

    Parameters
    ----------
    images: ndarray (dim, n_images)
        Image source positions.
    damping: ndarray
        Per-image attenuation, either (n_images,) or (n_bands, n_images).
    mic: ndarray (dim,)
        Microphone position.
    fs: int
        Sampling frequency.
    c: float, optional
        Speed of sound. Defaults to `constants.get("c")`.
    fdl: int, optional
        Fractional delay filter length.
    air_abs_coeffs: ndarray, optional
        Air absorption per octave band.

    Returns
    -------
    ndarray
        The room impulse response.
    """

    class _Src:
        pass

    src = _Src()
    src.images = np.asarray(images, dtype=np.float64)
    damping = np.asarray(damping, dtype=np.float64)
    src.damping = damping.reshape(1, -1) if damping.ndim == 1 else damping
    src.fs = fs
    return compute_ism_rir(
        src, mic, fdl=fdl, c=c, fs=fs, air_abs_coeffs=air_abs_coeffs
    )


__all__ = [
    "Beamformer",
    "Constants",
    "H",
    "MicrophoneArray",
    "autocorr",
    "beamforming",
    "build_rir_matrix",
    "circular_2D_array",
    "compute_ism_rir",
    "constants",
    "convmtx",
    "delay_sum",
    "distance",
    "fractional_delay",
    "fractional_delay_filter_bank",
    "goertzel",
    "levinson",
    "linear_2D_array",
    "low_pass_dirac",
    "lpc",
    "mdot",
    "mdot3",
    "poisson_2D_array",
    "rake_delay_and_sum",
    "rake_mvdr_filters",
    "rir_builder",
    "rir_builder_from_images",
    "spiral_2D_array",
    "square_2D_array",
    "unit_vec2D",
    "get_num_threads",
]