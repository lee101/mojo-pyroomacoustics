"""Package-wide constants, mirroring `pyroomacoustics.parameters`.

Only the constants the ported kernels actually read are present, with upstream's
names and default values so `constants.get("frac_delay_length")` means the same
thing here as it does there.
"""

from __future__ import annotations

import os

_constants: dict = {}
_constants_default = {
    "c": 343.0,  # speed of sound at 20 C in dry air
    "ffdist": 10.0,  # distance to the far field
    "fc_hp": 300.0,  # cut-off frequency of standard high-pass filter
    "frac_delay_length": 81,  # length of the fractional delay filters for RIR gen
    "room_isinside_max_iter": 20,
    "sinc_lut_granularity": 20,  # points per integer interval in the sinc LUT
    "num_threads": os.cpu_count() or 1,
    "octave_bands_n_fft": 512,
    "octave_bands_base_freq": 125.0,
    "octave_bands_keep_dc": True,
}


class Constants:
    """A class to provide easy access package wide to user settable constants."""

    def set(self, name, val):
        _constants[name] = val

    def get(self, name):
        try:
            return _constants[name]
        except KeyError:
            try:
                return _constants_default[name]
            except KeyError:
                raise NameError(name + ": no such constant")


constants = Constants()


def get_num_threads():
    """Upstream's thread-count probe; the RIR kernels are threaded through it."""
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count() or 1


_constants_default["num_threads"] = get_num_threads()