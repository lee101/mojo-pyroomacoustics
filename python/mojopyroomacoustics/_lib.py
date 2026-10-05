"""Loads (and if necessary builds) the compiled Mojo library.

The shared library is the only implementation of every kernel in this package.
If it is missing and cannot be built, `load()` raises rather than falling back to
NumPy: a silent pure-Python fallback would let a test pass without exercising
the port at all.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
BUILD = os.path.join(ROOT, "build", "build.sh")
LIB = os.environ.get("MOJO_PYROOMACUSTICS_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pyroomacoustics.so"
)

I = ctypes.c_int64
F = ctypes.c_double

# Every symbol src/ported.mojo @exports, with its ctypes signature.
_SIGNATURES: dict[str, tuple[list, object]] = {
    # utilities
    "mpa_build_sinc_lut": ([I, I, I], None),
    "mpa_build_hann": ([I, I], None),
    "mpa_convmtx_dot": ([I] * 5, None),
    "mpa_low_pass_dirac": ([I, I, I, I, I, F], None),
    "mpa_fractional_delay_bank": ([I, I, I, I, I, F], None),
    "mpa_levinson": ([I] * 7, None),
    "mpa_lpc_from_autocorr": ([I] * 5, None),
    "mpa_autocorr_time": ([I] * 4, None),
    "mpa_goertzel": ([I] * 4, None),
    # beamforming
    "mpa_distance": ([I] * 7, None),
    "mpa_unit_vec2D": ([I, F], None),
    "mpa_linear_2D_array": ([I, I, I, F, F, I], None),
    "mpa_circular_2D_array": ([I, I, I, F, F], None),
    "mpa_square_2D_array": ([I, I, I, I, F, F, I, I], None),
    "mpa_H": ([I] * 4, None),
    "mpa_mdot3": ([I] * 9, None),
    "mpa_steering_vector_2D_from_point": ([I] * 7 + [F, F, I, I, I], None),
    "mpa_steering_vector_2D": ([I] * 5 + [F, F, F, I, I, I], None),
    "mpa_response": ([I] * 5, None),
    "mpa_far_field_weights": ([I] * 5 + [F], None),
    "mpa_far_field_projection": ([I] * 5 + [F], None),
    "mpa_rake_delay_and_sum_weights": ([I] * 5, None),
    "mpa_cov_gram": ([I] * 4, None),
    "mpa_rake_mvdr_solution": ([I] * 5, None),
    "mpa_response_sinr": ([I] * 5, None),
    # RIR
    "mpa_rir_builder": ([I] * 10, None),
    "mpa_rir_builder_parallel": ([I] * 11, None),
    "mpa_delay_sum": ([I] * 6, None),
    "mpa_delay_sum_parallel": ([I] * 7, None),
    "mpa_fractional_delay": ([I] * 7, None),
    "mpa_ism_image_amplitudes": ([I] * 6 + [F, I, I], None),
    "mpa_ism_rir_length": ([I] * 4, I),
    "mpa_ism_rir_assemble": ([I] * 11, None),
}


class BuildError(RuntimeError):
    pass


class MissingLibrary(RuntimeError):
    """Raised when the compiled library is absent and cannot be produced."""


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_PYROOMACUSTICS_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    manifest = os.path.join(ROOT, "pixi.toml")
    if os.path.exists(pixi) and os.path.exists(manifest):
        return [pixi, "run", "--manifest-path", manifest, "mojo"]
    raise BuildError("mojo not found; set MOJO_PYROOMACUSTICS_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/ported.mojo` into `dist/libmojo-pyroomacoustics.so` if stale."""
    source = os.path.join(SRC, "ported.mojo")
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= os.path.getmtime(source):
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        source,
        "-o",
        LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    """The loaded shared library. This is the only implementation; no fallback."""
    global _lib
    if _lib is None:
        path = LIB
        if not os.path.exists(path):
            path = build()
        if not os.path.exists(path):
            raise MissingLibrary(f"compiled library not found at {path}")
        _lib = ctypes.CDLL(path)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def require() -> ctypes.CDLL:
    """Like `lib()` but fails loudly when the library is missing.

    Tests call this: a test that silently degrades to a pure-Python path proves
    nothing about the port.
    """
    try:
        return lib()
    except (BuildError, MissingLibrary, OSError) as exc:
        raise MissingLibrary(
            f"dist/libmojo-pyroomacoustics.so is required to test this port "
            f"and could not be built: {exc}"
        ) from exc


class _Address(int):
    """An ABI address that keeps the buffer it points into alive.

    Buffers are handed to the kernel as plain integers, so nothing in the
    ctypes call graph holds a reference to the NumPy array. That is safe while
    the caller keeps the array in a local, but every call site of the form
    `addr(f64(x))` builds a *temporary* that the returned integer does not
    keep alive: the array is freed the instant `addr(...)` returns, and the
    kernel then reads freed memory. This subclass carries a reference to the
    array for exactly as long as the address is alive, which closes that hole
    without changing any call site.

    `ctypes` accepts an `int` subclass for a `c_int64` argument, so the ABI is
    unchanged.
    """

    def __new__(cls, buffer: np.ndarray):
        return super().__new__(cls, buffer.ctypes.data)

    def __init__(self, buffer: np.ndarray):
        self._buffer = buffer


def addr(a: np.ndarray) -> _Address:
    """Address of a C-contiguous array, as the ABI expects.

    The returned object keeps `a` alive for as long as it exists, so it is safe
    to pass as a temporary: `addr(f64(x))` cannot free `x` before the kernel
    has read it.
    """
    if not a.flags["C_CONTIGUOUS"]:
        raise ValueError(
            f"the buffer must be C-contiguous; got strides {a.strides} for a "
            f"buffer of shape {a.shape}"
        )
    return _Address(a)


def f64(a) -> np.ndarray:
    """A C-contiguous float64 array. Copies only when it has to.

    The result may be a temporary the caller does not name, which is why
    `addr` keeps its buffer alive.
    """
    return np.ascontiguousarray(a, dtype=np.float64)


def c128(a) -> np.ndarray:
    """A C-contiguous complex128 array, viewed over its own float64 storage.

    Complex values live in Mojo as interleaved (re, im) pairs, so a complex
    array is passed as a float64 array of twice the length and viewed back on
    the other side. No copy.
    """
    arr = np.ascontiguousarray(a, dtype=np.complex128)
    return arr.view(np.float64)


def as_c128(a: np.ndarray) -> np.ndarray:
    """Inverse of `c128`: reinterpret a float64 buffer as complex128."""
    return a.view(np.complex128)


def workers() -> int:
    """Thread count to hand the parallel RIR kernels.

    `constants.get("num_threads")` upstream is this; it is read through the
    shared module so both sides agree.
    """
    from .parameters import constants

    return max(1, int(constants.get("num_threads")))


def main() -> int:
    """`python -m mojopyroomacoustics._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())